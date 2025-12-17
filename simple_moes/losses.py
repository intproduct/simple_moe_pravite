# Loss functions

import mlx.core as mx
import mlx.nn as nn
from .gate import log_softmax
from typing import List, Type, Any, Optional

def cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    log_probs = log_softmax(logits, temperature=1.0, axis=-1)              
    idx = targets.astype(mx.int32).reshape(-1, 1)         
    picked = mx.take_along_axis(log_probs, idx, axis=1)
    loss = (-picked).mean()
    return loss

def mse_loss(pred: mx.array, target: mx.array) -> mx.array:
    pred = pred.reshape(-1)
    target = target.reshape(-1).astype(pred.dtype)
    return ((pred - target) ** 2).mean()

def balance_loss(probs: mx.array, mask: Optional[mx.array], coeff: float) -> mx.array:
    probs = probs.astype(mx.float32)
    if mask is not None:
       masked = probs * mask    #Select the top k elements at their positions, other positions are 0
       probs = masked / (mx.sum(masked, axis=-1, keepdims=True) + 1e-9)
    probs = mx.clip(probs, 1e-9, 1.0)
    ent = -mx.sum(probs * mx.log(probs), axis=-1).mean()
    return -coeff * ent

def accuracy(logits: mx.array, targets: mx.array) -> mx.array:
    preds = mx.argmax(logits, axis=-1)
    correct = (preds == targets).astype(mx.float32)
    acc = correct.mean()
    return float(acc)
