# Gating functions or Routings for Mixture of Experts

import mlx.core as mx
import mlx.nn as nn
from typing import List, Type, Any, Optional, Union

def softmax_with_temperature(logits: mx.array, temperature: float, axis = -1) -> mx.array:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    type_logits = logits.dtype
    logits = (logits / mx.array(temperature)).astype(mx.float32)
    return mx.softmax(logits, axis=axis).astype(type_logits)

def log_softmax(x:mx.array, temperature:float, axis=-1)->mx.array:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    x_type = x.dtype
    x = (x / temperature).astype(mx.float32)  
    x_max = mx.max(x, axis=axis, keepdims=True) 
    x_exp = mx.exp(x - x_max)
    x_exp_sum = mx.sum(x_exp, axis=axis, keepdims=True)
    return (mx.log(x_exp) - mx.log(x_exp_sum)).astype(x_type)

def topk_routing(probs:mx.array, k:int)->Union[mx.array, mx.array]:
    probs = probs.astype(mx.float32)
    idx_sorted = mx.argsort(probs, axis=-1)  #Sort index for each element in a row, the biggest element's index at the last
    topk_idx = idx_sorted[:, -k:]     #get the index of the top k elements(the last k elements in the sorted index)                        

    B, N = probs.shape
    mask = mx.zeros((B, N), dtype=probs.dtype)
    rows = mx.arange(B).reshape(-1, 1)
    mask[rows, topk_idx] = 1.0

    masked = probs * mask    #Select the top k elements at their positions, other positions are 0
    norm = masked / (mx.sum(masked, axis=-1, keepdims=True) + 1e-9)

    return norm.astype(probs.dtype), mask.astype(probs.dtype)

class SoftGate(nn.Module):
    def __init__(self,data_dim:int, num_experts:int, temperature:float=1.0):
        super().__init__()
        self.fc = nn.Linear(data_dim, num_experts)
        self.temperature = temperature

    def __call__(self, x:mx.array)->mx.array:
        logits = self.fc(x)
        probs = softmax_with_temperature(logits, self.temperature, axis=-1)
        return probs, None    #这里返回None是为了和topk gate的返回值保持一致，后者要返回掩码
    
class TopKGate(nn.Module):
    def __init__(self, data_dim:int, num_experts:int, k:int=2, temperature:float=1.0):
        super().__init__()
        self.fc = nn.Linear(data_dim, num_experts)
        self.k = k
        self.temperature = temperature

    def __call__(self, x:mx.array)->Union[mx.array, mx.array]:
        logits = self.fc(x)
        probs = softmax_with_temperature(logits, self.temperature, axis=-1)
        topk_probs, mask = topk_routing(probs, self.k)
        return topk_probs, mask

def gate_factory(gate_type: str, data_dim: int, num_experts: int, **kwargs) -> nn.Module:
    gate_type = gate_type.lower() #统一转换为小写
    if gate_type == "topk":
        return TopKGate(data_dim, num_experts, k=kwargs.get("k", 2), temperature=kwargs.get("temperature", 1.0))
    if gate_type == "softmax":
        return SoftGate(data_dim, num_experts, temperature=kwargs.get("temperature", 1.0))
    else:
        raise ValueError(f"Unsupported gate type: {gate_type}")
