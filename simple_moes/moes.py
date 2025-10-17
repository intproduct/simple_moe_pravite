# Expert models for Mixture of Experts (MoE) architectures.

import mlx.core as mx
import mlx.nn as nn
from typing import List, Type, Any, Optional
from SimpleMoEs.losses import balance_loss
from SimpleMoEs.gate import gate_factory
from SimpleMoEs.heads import ClassificationHead, RegressionHead


class MoElayer(nn.Module):
    def __init__(self, dim_input: int, dim_output: int, num_experts: int,
                 dim_hidden: int, entropy_coeff: int,
                 experts: List[Type[nn.Module]], gate_fn: nn.Module,
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

    def __call__(self, x: mx.array, return_balance: bool = False) -> mx.array:
        probs, mask = self.gate(x)
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
                 expert_args: Optional[List[dict]] = None):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_input, dim_model)
        )

        self.gate = gate_factory(gate_name, dim_model, num_experts) #标准情况下调用top2/softmax 温度为1.0

        self.moelayer = MoElayer(
            dim_input=dim_model,
            dim_output=dim_model,
            num_experts=num_experts,
            dim_hidden=dim_hidden,
            entropy_coeff=entropy_coeff,    
            experts=experts,
            gate_fn=self.gate)
        
        self.head = ClassificationHead(dim_model, dim_output)

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