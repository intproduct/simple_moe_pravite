# Expert models for Mixture of Experts (MoE) architectures.

import mlx.core as mx
import mlx.nn as nn
from typing import List, Type, Any, Optional
from .losses import balance_loss
from .gate import gate_factory
from .heads import ClassificationHead, RegressionHead, LinearHead


class MoElayer(nn.Module):
    def __init__(self, dim_input: int, dim_output: int, num_experts: int,
                 dim_hidden: int, entropy_coeff: int,
                 experts: List[nn.Module], gate_fn: nn.Module,
                 expert_args: Optional[List[dict]] = None):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_experts = num_experts
        self.dim_hidden = dim_hidden
        self.expert_layers = experts
        self.gate = gate_fn
        self.coeff = entropy_coeff
        self.expert_args = expert_args

        for i, e in enumerate(experts):
            setattr(self, f"expert_{i}", e)

    def __call__(self, x: mx.array, return_balance: bool = False) -> mx.array:
        probs, mask = self.gate(x)
        #self._last_probs = probs  # For analysis purposes
        #mask = mx.stop_gradient(mask)
        if return_balance:
            ba = balance_loss(probs, mask, self.coeff)
        probs = probs[..., None]
        outs = [exp(x) for exp in self.expert_layers]
        outs = mx.stack(outs, axis=-2)

        y = mx.sum(outs * probs, axis=-2)
        
        if return_balance:
            return y, ba
        else:
            return y
        
class GenericMoE(nn.Module):
    def __init__(self, dim_input: int, dim_model: int, dim_output: int, num_experts: int,
                 dim_hidden: int, entropy_coeff: float,
                 experts: List[Type[nn.Module]], gate_name: str,
                 expert_args: Optional[List[dict]] = None,
                 **gate_kwargs):
        super().__init__()
        self.input_layer = nn.Linear(dim_input, dim_model)

        gate = gate_factory(gate_name, dim_model, num_experts, **gate_kwargs) #标准情况下调用top2/softmax 温度为1.0

        self.moelayer = MoElayer(
            dim_input=dim_model,
            dim_output=dim_model,
            num_experts=num_experts,
            dim_hidden=dim_hidden,
            entropy_coeff=entropy_coeff,    
            experts=experts,
            gate_fn=gate)
        
        self.head = LinearHead(dim_model, dim_output) #实现上完全一样，似乎不改也行

    def __call__(self, x: mx.array, return_balance: bool = False) -> mx.array:
        x = self.input_layer(x)
        if return_balance:
            x, balance = self.moelayer(x, return_balance=True)
            logits = self.head(x)
            return logits, balance
        else:
            x = self.moelayer(x)
            logits = self.head(x)
            return logits
        
class DenseRegressor(nn.Module):
    #Dense baseline：
    def __init__(self, dim_input: int, dim_hidden: int, dim_output: int, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = dim_input
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, dim_hidden))
            layers.append(nn.ReLU())
            in_dim = dim_hidden
        layers.append(nn.Linear(in_dim, dim_output))
        self.net = nn.Sequential(*layers)

    def __call__(self, x: mx.array, return_balance: bool = False):
        y = self.net(x)
        if return_balance:
            aux = mx.array(0.0, dtype=mx.float32)  # Dense 没有 balance_loss
            return y, aux
        return y
