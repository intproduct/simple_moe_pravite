# Training process for Mixture of Experts (MoE) models

import mlx.core as mx
import mlx.nn as nn
from typing import List, Type, Any, Optional, Union
from SimpleMoEs.losses import cross_entropy, mse_loss, balance_loss, accuracy

def train_step_classification(model: Type[nn.Module], optimizer: Type[nn.Module], x: mx.array, y: mx.array, aux_weight: float = 1.0) -> Union[mx.array, float, mx.array]:
    def loss_aux():
        logits, aux = model(x, return_balance=True)
        ce_loss = cross_entropy(logits, y) + aux_weight * aux
        return ce_loss, (logits, aux)
    
    (ce_loss, (logits, aux)), grads = nn.value_and_grad(model, loss_aux)()

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    acc = accuracy(logits, y)
    return ce_loss.item(), acc, aux.item()

def train_step_regression(model: Type[nn.Module], optimizer: Type[nn.Module], x: mx.array, y: mx.array, aux_weight: float = 1.0) -> Union[mx.array, float, mx.array]:
    def loss_aux():
        preds, aux = model(x, return_balance=True)
        mse = mse_loss(preds, y) + aux_weight * aux
        return mse, (preds, aux)
    
    (mse, (preds, aux)), grads = nn.value_and_grad(model, loss_aux)()

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    rmse = float(mx.sqrt(mse))
    return mse.item(), rmse, aux.item()
        