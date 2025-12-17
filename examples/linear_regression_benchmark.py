import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from tqdm import tqdm
import mlx.optimizers as opt
import mlx.nn as nn
import mlx.core as mx
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass
import dataclasses
import logging

from SimpleMoEs import (
    GenericMoE, MLPExpert, gate_factory,
    batch_iter, evaluate_model, set_seed, Trainer, learning_schedule,
    DenseRegressor,smooth, setup_logger,
) 

def print_param_tree(params, prefix=""):
    if isinstance(params, dict):
        for k, v in params.items():
            print_param_tree(v, prefix + k + ".")
    else:
        print(f"{prefix[:-1]:40s}")

def make_linear_dataset(n=5000, d=64, noise_std=0.0, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n, d)).astype(np.float32)
    w = rng.normal(0, 1.0, size=(d,)).astype(np.float32)
    y = X @ w + rng.normal(0, noise_std, size=(n,)).astype(np.float32)

    X = mx.array(X)
    y = mx.array(y)

    # 标准化 label
    mu = mx.mean(y)
    sigma = mx.std(y) + 1e-8
    y = (y - mu) / sigma
    return X, y

#make datas
def make_composition_dataset(
    n=50000, d=1024, num_paths=4, std=1.0, noise_std=0.05, seed=42,
    h=512
):

    rng = np.random.default_rng(seed)
    per = n // num_paths

    # -------- 1) d -> h 的线性投影 --------
    W1 = rng.normal(0, 0.2, size=(d, h)).astype(np.float32)
    W2 = rng.normal(0, 0.2, size=(d, h)).astype(np.float32)
    W3 = rng.normal(0, 0.2, size=(d, h)).astype(np.float32)
    b1 = rng.normal(0, 0.2, size=(h,)).astype(np.float32)
    b2 = rng.normal(0, 0.2, size=(h,)).astype(np.float32)
    b3 = rng.normal(0, 0.2, size=(h,)).astype(np.float32)

    # -------- 2) h -> h 的“中间算子” --------
    V1 = rng.normal(0, 0.2, size=(h, h)).astype(np.float32)
    V2 = rng.normal(0, 0.2, size=(h, h)).astype(np.float32)
    V3 = rng.normal(0, 0.2, size=(h, h)).astype(np.float32)

    # -------- 3) 三个基础算子：自动判断输入维度 --------
    def f1(x):
        if x.shape[1] == d:
            return np.tanh(x @ W1 + b1)
        else:  # h-dim
            return np.tanh(x @ V1)

    def f2(x):
        if x.shape[1] == d:
            return np.maximum(0, x @ W2 + b2)     # ReLU(d->h)
        else:
            return np.maximum(0, x @ V2)           # ReLU(h->h)

    def f3(x):
        if x.shape[1] == d:
            return np.sin(x @ W3 + b3)             # sin(d->h)
        else:
            return np.sin(x @ V3)                  # sin(h->h)

    # -------- 4) 生成不同“推理路径”的样本 --------
    Xs, Ys = [], []
    for i in range(num_paths):
        X = rng.normal(0, std, size=(per, d)).astype(np.float32)

        if i == 0:
            Y = f2(f1(X)).sum(axis=1)
        elif i == 1:
            Y = f3(f1(X)).sum(axis=1)
        elif i == 2:
            Y = f1(f3(X)).sum(axis=1)
        else:
            Y = f2(f3(X)).sum(axis=1)

        # 观测噪声
        Y = Y + rng.normal(0, noise_std, size=Y.shape).astype(np.float32)

        Xs.append(X.astype(np.float32))
        Ys.append(Y.astype(np.float32))

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)

    idx = rng.permutation(X.shape[0])
    return mx.array(X[idx]), mx.array(Y[idx])

def build_model(
        input_dim,
        num_classes,
        dim_model,
        num_experts,
        dim_hidden,
        gate_name="topk",
        k=1,
        temperature=1.0,
        num_exp_layers=1,
        entropy_coeff=1e-2
):
    model = GenericMoE(
        dim_input=input_dim,
        dim_model=dim_model,
        dim_output=num_classes,
        num_experts=num_experts,
        dim_hidden=dim_hidden,
        entropy_coeff=entropy_coeff,
        experts=[MLPExpert(input_dim=dim_model, hidden_dim=dim_hidden, output_dim=dim_model, num_layers=num_exp_layers) for _ in range(num_experts)],
        gate_name=gate_name,
        k=k,
        temperature=temperature
    )

    return model

@dataclass
class SuperParamters:
    batch_size: int
    d_model: int
    d_hidden: int
    d_in: int
    num_exp: int
    out_dim: int
    num_epochs: int
    init_lr: float
    warmup_steps: int
    decay_steps: int
    end_lr: float
    k_num: int
    choose_gate_name: str
    tau: float
    num_exp_layers: int
    entropy_coe: float

def build_optimizer_from_cfg(cfg: SuperParamters, use_schedule: bool):
    if use_schedule:
        lr_fn = learning_schedule(
            init=cfg.init_lr,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.decay_steps,
            end=cfg.end_lr,
        )
        return opt.Adam(learning_rate=lr_fn)
    else:
        return opt.Adam(learning_rate=cfg.init_lr)
    
def run_experiment(
        Xtr, ytr, X_test, y_test,
        cfg: SuperParamters,
        use_schedule: bool,
        tag: str = "fixed",
        model_tag: str = "moe",   # 新增：'dense' 或 'moe'
        logger: logging.Logger | None = None,
):
    batch_size = cfg.batch_size
    d_model = cfg.d_model
    d_hidden = cfg.d_hidden
    d_in = cfg.d_in
    num_exp = cfg.num_exp
    out_dim = cfg.out_dim
    num_epochs = cfg.num_epochs
    choose_gate_name = cfg.choose_gate_name
    tau = cfg.tau
    num_exp_layers = cfg.num_exp_layers
    entropy_coe = cfg.entropy_coe

    # === 构建模型：Dense 和 MoE 分支 ===
    if model_tag == "dense":
        model = DenseRegressor(
            dim_input=d_in,
            dim_hidden=d_model,
            dim_output=out_dim,
            num_layers=2,
        )
    else:
        model = build_model(
            input_dim=d_in,
            num_classes=out_dim,
            dim_model=d_model,
            num_experts=num_exp,
            dim_hidden=d_hidden,
            gate_name=choose_gate_name,
            k=cfg.k_num,
            temperature=tau,
            num_exp_layers=num_exp_layers,
            entropy_coeff=entropy_coe,
        )

    optimizer = build_optimizer_from_cfg(cfg, use_schedule=use_schedule)
    trainer = Trainer(model, optimizer, "regression", logger)

    train_mse_list, train_rmse_list = [], []
    test_mse_list, test_rmse_list = [], []

    print("=====================Paramsters List=====================")
    print_param_tree(model.parameters())

    for epoch in tqdm(range(num_epochs), "Epoch"):
        model.train(True)

        train_mse_step, train_rmse_step, train_aux = trainer.train_epoch(
            batch_iter(Xtr, ytr, batch_size=batch_size, shuffle=True),
            log_every=50,
        )

        # 1) 在训练集上评估 ---- 注：此处不使用，因为训练集过大，计算量过大
        #train_mse_eval, train_rmse_eval = evaluate_model(
        #    model, Xtr, ytr, batch_size=batch_size, task_type="regression"
        #)
        # 2) 在测试集上评估
        test_mse_eval, test_rmse_eval = evaluate_model(
            model, X_test, y_test, batch_size=batch_size, task_type="regression"
        )

        train_mse_list.append(train_mse_step)
        train_rmse_list.append(train_rmse_step)
        test_mse_list.append(test_mse_eval)
        test_rmse_list.append(test_rmse_eval)

        logger.info(
            f"[{tag}] Epoch {epoch+1}/{num_epochs} "
            f"- Train(step) MSE {train_mse_step:.4f} RMSE {train_rmse_step:.4f} "
            #f"- Train(eval) MSE {train_mse_eval:.4f} RMSE {train_rmse_eval:.4f} "
            f"- Test(eval)  MSE {test_mse_eval:.4f}  RMSE {test_rmse_eval:.4f} "
            f"- Aux {train_aux:.4f}"
        )

    stats = {
        "train_mse": train_mse_list,
        "train_rmse": train_rmse_list,
        "test_mse": test_mse_list,
        "test_rmse": test_rmse_list,
    }

    return model, stats

def plot_losses(all_stats, smooth_k: int = 20):
    # -------- 1) Train MSE (log) --------
    plt.figure(figsize=(10, 5))
    for name, stats in all_stats.items():
        train_rmse = stats["train_rmse"]
        train_rmse_sm = smooth(train_rmse, k=smooth_k)
        plt.plot(train_rmse_sm, label=name)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Train RMSE (smoothed, log scale)")
    plt.title("Train RMSE on composition task (log scale)")
    plt.legend()
    plt.tight_layout()
    current_time = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    plt.savefig(f"./SimpleMoEs/pics/compose_func_train_mse_{current_time}.png")

    # -------- 2) Test MSE (log) --------
    plt.figure(figsize=(10, 5))
    for name, stats in all_stats.items():
        test_rmse = stats["test_rmse"]
        test_rmse_sm = smooth(test_rmse, k=smooth_k)
        plt.plot(test_rmse_sm, label=name)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Test RMSE (smoothed, log scale)")
    plt.title("Test RMSE on composition task (log scale)")
    plt.legend()
    plt.tight_layout()
    current_time = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    plt.show()
    plt.savefig(f"./SimpleMoEs/pics/compose_func_test_mse_{current_time}.png")

    

def main():
    log = setup_logger("compose_func_1", "INFO", "./SimpleMoEs/logs/compose_func_1.log")
    set_seed(33550337)
    #Xtr, ytr = make_composition_dataset(n=70000, d=1024, num_paths=4, seed=33550337)
    #X_test, y_test = make_composition_dataset(n=10000, d=1024, num_paths=4, seed=42)
    Xtr, ytr = make_linear_dataset(n=20000, d=1024, noise_std=0.0)
    X_test, y_test = make_linear_dataset(n=5000, d=1024, noise_std=0.0, seed=24)

    mu = mx.mean(ytr)
    sigma = mx.std(ytr) + 1e-8

    ytr = (ytr - mu) / sigma
    y_test = (y_test - mu) / sigma

    base_cfg = SuperParamters(
        batch_size = 256,
        d_model = 512,
        d_hidden = 2048,
        d_in = 1024,
        num_exp = 4,        # 默认值，下面会覆盖
        out_dim = 1,
        num_epochs = 100,   # 建议先从 200/300 试，2000 太狠了
        init_lr=1e-3,
        warmup_steps=1000, 
        decay_steps=12000,    
        end_lr=1e-5,
        k_num = 1,
        choose_gate_name = "topk",  # 默认，下面再换
        tau = 1.0,
        num_exp_layers = 1,
        entropy_coe = 1e-2,
    )

    # 五个实验：name, cfg(拷贝后改字段), model_tag
    
    exps = []

    # 1. Dense model
    cfg_dense = dataclasses.replace(
        base_cfg,
        num_exp=1,
        choose_gate_name="dense",   # 自定义标记
        entropy_coe=0.0,            # Dense 不需要 balance_loss
    )
    exps.append(("Dense", cfg_dense, "dense"))

    # 2. TopkGate, 1 expert
    cfg_topk1 = dataclasses.replace(
        base_cfg,
        num_exp=1,
        choose_gate_name="topk",
        entropy_coe=0.0,
    )
    exps.append(("TopkGate-1E", cfg_topk1, "moe"))

    '''# 3. SoftGate, 4 experts
    cfg_topk4 = dataclasses.replace(
        base_cfg,
        num_exp=4,
        choose_gate_name="topk",
        entropy_coe=1e-2,
    )
    exps.append(("TopkGate-4E", cfg_topk4, "moe"))

    # 4. Quantum_layer_Gate, 1 expert
    cfg_q1 = dataclasses.replace(
        base_cfg,
        num_exp=1,
        choose_gate_name="quantum",
        entropy_coe=1e-2,
    )
    exps.append(("Quantum-1E", cfg_q1, "moe"))

    # 5. Quantum_layer_Gate, 4 experts
    cfg_q4 = dataclasses.replace(
        base_cfg,
        num_exp=4,
        choose_gate_name="quantum",
        entropy_coe=1e-2,
    )
    exps.append(("Quantum-4E", cfg_q4, "moe"))'''
    
    cfg_soft1 = dataclasses.replace(
        base_cfg,
        num_exp=1,
        choose_gate_name="softmax",
        entropy_coe=0.0,
    )
    exps.append(("SoftGate-1E", cfg_soft1, "moe"))

    all_stats = {}
    models = {}

    print("===== 对照实验开始 =====")
    for name, cfg, model_tag in exps:
        print(f"\n>>> Running experiment: {name}")
        model, stats = run_experiment(
            Xtr, ytr, X_test, y_test,
            cfg=cfg,
            use_schedule=False,
            tag=name,
            model_tag=model_tag,
            logger = log,
        )
        models[name] = model
        all_stats[name] = stats

    # 画图
    plot_losses(all_stats)

    # 例如保存某个模型
    #mx.save_safetensors("./SimpleMoEs/models/model_dense.safetensors",
                        #models["Dense"].parameters())
    #mx.save_safetensors("./SimpleMoEs/models/model_moe_1e.safetensors",
                        #models["TopkGate-1E"].parameters())
    #mx.save_safetensors("./SimpleMoEs/models/model_moe_4e.safetensors",
                        #models["TopkGate-4E"].parameters())
    #mx.save_safetensors("./SimpleMoEs/models/model_quantum_1e.safetensors",
                        #models["Quantum-1E"].parameters())
    #mx.save_safetensors("./SimpleMoEs/models/model_quantum_4e.safetensors",
                        #models["Quantum-4E"].parameters())


if __name__ == "__main__":
    main()

