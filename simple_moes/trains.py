# Training process for Mixture of Experts (MoE) models

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from typing import List, Type, Any, Optional, Union
from .losses import cross_entropy, mse_loss, balance_loss, accuracy
from .tools import snapshot_params, params_delta_norm, params_l2_norm
import logging

def linear_warmup(init, warmup_steps):
    def fn(step):
        if step > warmup_steps:
            return init
        return init * (step/warmup_steps)
    return fn

def learning_schedule(init, warmup_steps, decay_steps, end=0.0):
    warmup_fn = linear_warmup(init, warmup_steps)
    cosine_fn = opt.cosine_decay(init, decay_steps, end)

    def schedule(step):
        if step < warmup_steps:
            return warmup_fn
        else:
            return cosine_fn(step - warmup_steps)
    return schedule

def train_step_classification(model: nn.Module, optimizer: nn.Module, x: mx.array, y: mx.array) -> Union[mx.array, float, mx.array]:
    def loss_aux():
        logits, aux = model(x, return_balance=True)
        ce_loss = cross_entropy(logits, y)
        total_loss = ce_loss +  aux
        return total_loss, (logits, aux, ce_loss)
    
    (total_loss, (logits, aux, ce_loss)), grads = nn.value_and_grad(model, loss_aux)()

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    acc = accuracy(logits, y)
    return total_loss.item(), acc, aux.item()

def train_step_regression(model: nn.Module, optimizer: nn.Module, x: mx.array, y: mx.array) -> Union[mx.array, float, mx.array]:
    def loss_aux():
        preds, aux = model(x, return_balance=True)
        '''#测试梯度消失问题的代码
        p_mean = float(mx.mean(preds).item())
        p_std = float(mx.std(preds).item())
        probs = model.moelayer._last_probs
        g_max = float(mx.mean(mx.max(probs, axis=-1)))
        g_std = float(mx.std(probs).item())
        #这里是结束'''
        y_ = y
        if y_.ndim == 1 and preds.ndim == 2 and preds.shape[1] == 1:
            y_ = y[:, None]
        mse = mse_loss(preds, y_)
        total_loss = mse + aux
        return total_loss, (mse, aux)#p_mean, p_std, g_max, g_std
    
    loss_and_grad = nn.value_and_grad(model, loss_aux)
    (total_loss, (mse, aux)), grads = loss_and_grad()

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    rmse = float(mx.sqrt(mse))

    return mse.item(), rmse, aux.item() #p_mean, p_std, g_max, g_std

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: nn.Module,
            task_type: str = "regression",
            #aux_weight: float = 1.0,
            logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.task_type = task_type
        #self.aux_weight = aux_weight
        self.logger = logger

    def train_step(
            self,
            x: mx.array,
            y: mx.array
    ):
        if self.task_type == "classification":
            return train_step_classification(self.model, self.optimizer, x, y)
        elif self.task_type == "regression":
            return train_step_regression(self.model, self.optimizer, x, y)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
    def train_epoch(
            self,
            data_loader,
            log_every: int = 100,
    ):
        total_loss = 0.0
        total_metric = 0.0
        total_aux = 0.0
        n_steps = 0

        for step, (x, y) in enumerate(data_loader):

            before = snapshot_params(self.model)

            loss, metric, aux = self.train_step(x, y)

            mx.eval(self.model.parameters(), self.optimizer.state)
            '''if step < 3:   # 只看前 3 个 batch 就够了
                delta = params_delta_norm(self.model, before)
                total_norm = params_l2_norm(self.model)
                self.logger.info(
                    f"[debug] step {step} Δ||params|| = {delta:.6e}, "
                    f"||params|| = {total_norm:.6e}"
                    #f"[debug] preds mean/std: {p_mean:.6e}/{p_std:.6e}, "
                    #f"[debug] gate max/std: {g_max:.6e}/{g_std:.6e}"
                )'''

            total_loss += loss
            total_metric += metric
            total_aux += aux
            n_steps += 1

            if (step + 1) % log_every == 0:
                self.logger.info(f"Step {step + 1}: loss = {total_loss / n_steps:.4f}, metric = {total_metric / n_steps:.4f}, aux = {total_aux / n_steps:.4f}")

        return (
            total_loss / n_steps,
            total_metric / n_steps,
            total_aux / n_steps
        )
