import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from tqdm import tqdm
import mlx.optimizers as opt
from torchvision.datasets import MNIST
import mlx.nn as nn
import mlx.core as mx
import matplotlib.pyplot as plt

from SimpleMoEs import (
    GenericMoE, MLPExpert, gate_factory,
    batch_iter, evaluate_moe, set_seed, train_step_classification
)

def load_mnist():
    print("Downloading MNIST dataset...")
    train = MNIST(root="./data", train=True,  download=True, transform=None)
    test  = MNIST(root="./data", train=False, download=True, transform=None)

    X_train = np.array(train.data).astype(np.float32) / 255.0
    y_train = np.array(train.targets).astype(np.int64)
    X_test  = np.array(test.data).astype(np.float32) / 255.0
    y_test  = np.array(test.targets).astype(np.int64)

    mean = mx.array([0.1307], dtype=mx.float32)
    std  = mx.array([0.3081], dtype=mx.float32)

    mean_flat = mx.repeat(mean, repeats=28*28)
    std_flat  = mx.repeat(std, repeats=28*28)

    X_train = mx.array(X_train.reshape(-1, 28*28*1))
    X_test  = mx.array(X_test.reshape(-1, 28*28*1))
    y_train = mx.array(y_train)
    y_test  = mx.array(y_test)

    X_train = (X_train - mean_flat) / (std_flat + 1e-16)     #标准化数据，避免数据带来logits快速增长
    X_test  = (X_test  - mean_flat) / (std_flat + 1e-16)

    return X_train, y_train, X_test, y_test

# Build MoE model for CIFAR-10 classification, using MLP experts
def build_model(
    input_dim: int, num_classes: int,
    dim_model: int, num_experts: int, dim_hidden: int,
    gate_name: str, k: int, temperature: float, num_exp_layers: int,
    entropy_coeff: float
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
        k = k,
        temperature = temperature
    )
    return model

def main():
    set_seed(42)
    Xtr, ytr, Xte, yte = load_mnist()
    print(Xtr.shape, ytr.shape, Xte.shape, yte.shape)

    batch_size = 256
    d_model = 512
    d_hidden = 256
    in_dim = Xtr.shape[-1]
    num_exp = 10
    out_dim = 10
    num_epochs = 50
    lr = 1e-3
    k_num = 2
    choose_gate_name = "topk"
    tau = 1.0
    num_exp_layer = 1
    entropy_coe = 1e-2

    model = build_model(
        input_dim=in_dim, num_classes=out_dim,
        dim_model=d_model, num_experts=num_exp, dim_hidden=d_hidden,
        gate_name=choose_gate_name, k=k_num, temperature=tau, num_exp_layers=num_exp_layer,
        entropy_coeff=entropy_coe
    )
    optimizer = opt.Adam(learning_rate=lr)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        model.train(True)
        pbar = tqdm(batch_iter(Xtr, ytr, batch_size=batch_size, shuffle=True), total=Xtr.shape[0]//batch_size)
        for xb, yb in pbar:
            loss_val,acc_val, aux_val = train_step_classification(model, optimizer, xb, yb, aux_weight=1e-2)
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss_val:.4f} - Acc: {acc_val:.4f} - Aux: {aux_val:.4f}")

        train_loss, train_acc = evaluate_moe(model, Xtr, ytr, batch_size=batch_size)
        test_loss, test_acc = evaluate_moe(model, Xte, yte, batch_size=batch_size)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    # Plot training curves
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.show()
    plt.savefig(f"SimpleMoEs/pics/cifar10_moe_{num_exp}exps_{choose_gate_name}_{k_num}gates.png")

main()


    
