# Training process for Mixture of Experts (MoE) models

import mlx.core as mx
import mlx.nn as nn
from typing import List, Type, Any, Optional, Union
from .losses import cross_entropy, mse_loss, balance_loss, accuracy

def train_step_classification(model: nn.Module, optimizer: nn.Module, x: mx.array, y: mx.array, aux_weight: float = 1.0) -> Union[mx.array, float, mx.array]:
    def loss_aux():
        logits, aux = model(x, return_balance=True)
        ce_loss = cross_entropy(logits, y)
        total_loss = ce_loss + aux_weight * aux
        return total_loss, (logits, aux, ce_loss)
    
    (total_loss, (logits, aux, ce_loss)), grads = nn.value_and_grad(model, loss_aux)()

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    acc = accuracy(logits, y)
    return total_loss.item(), acc, aux.item()

def train_step_regression(model: nn.Module, optimizer: nn.Module, x: mx.array, y: mx.array, aux_weight: float = 1.0) -> Union[mx.array, float, mx.array]:
    def loss_aux():
        preds, aux = model(x, return_balance=True)
        mse = mse_loss(preds, y)
        total_loss = mse + aux_weight * aux
        return total_loss, (mse, aux)
    
    (total_loss, (mse, aux)), grads = nn.value_and_grad(model, loss_aux)()

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    rmse = float(mx.sqrt(mse))

    return mse.item(), rmse, aux.item()

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: nn.Module,
            task_type: str = "regression",
            aux_weight: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.task_type = task_type
        self.aux_weight = aux_weight

    def train_step(
            self,
            x: mx.array,
            y: mx.array
    ):
        if self.task_type == "classification":
            return train_step_classification(self.model, self.optimizer, x, y, self.aux_weight)
        elif self.task_type == "regression":
            return train_step_regression(self.model, self.optimizer, x, y, self.aux_weight)
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
            loss, metric, aux = self.train_step(x, y)
            total_loss += loss
            total_metric += metric
            total_aux += aux
            n_steps += 1

            if (step + 1) % log_every == 0:
                print(f"Step {step + 1}: loss = {total_loss / n_steps:.4f}, metric = {total_metric / n_steps:.4f}, aux = {total_aux / n_steps:.4f}")

        return (
            total_loss / n_steps,
            total_metric / n_steps,
            total_aux / n_steps
        )
