# SimpleMoEs/__init__.py
"""
SimpleMoEs: a tiny Mixture-of-Experts (MoE) library on MLX.

Public API layout:
- experts:     expert building blocks (MLP, CNN, RNN)
- gate:        gating functions and factories (soft/TopK, temperature-aware)
- heads:       task heads (classification / regression)
- losses:      CE / MSE / balance loss / accuracy
- moes:        MoE layer and a ready-to-use GenericMoE model
- tools:       training/eval helpers and small utilities
- trains:      per-step training routines for cls/reg
"""

__version__ = "0.1.0"

# Experts
from .experts import (
    MLPExpert,
    CNNExpert,
    RNNExpert,
)

# Gating & routing
from .gate import (
    softmax_with_temperature,
    log_softmax,
    topk_routing,
    SoftGate,
    TopKGate,
    gate_factory,
)

# Heads
from .heads import (
    ClassificationHead,
    RegressionHead,
)

# Losses & metrics
from .losses import (
    cross_entropy,
    mse_loss,
    balance_loss,
    accuracy,
)

# MoE composites
from .moes import (
    MoElayer,
    GenericMoE,
)

# Utilities
from .tools import (
    smooth,
    set_seed,
    batch_iter,
    evaluate,
    evaluate_moe,
    MoEInspector,
)

# Train steps
from .trains import (
    train_step_classification,
    train_step_regression,
    Trainer
)

__all__ = [
    # experts
    "MLPExpert", "CNNExpert", "RNNExpert",
    # gate
    "softmax_with_temperature", "log_softmax", "topk_routing",
    "SoftGate", "TopKGate", "gate_factory",
    # heads
    "ClassificationHead", "RegressionHead",
    # losses
    "cross_entropy", "mse_loss", "balance_loss", "accuracy",
    # moes
    "MoElayer", "GenericMoE",
    # tools
    "smooth", "set_seed", "batch_iter", "evaluate", "evaluate_moe", "MoEInspector"
    # train steps
    "train_step_classification", "train_step_regression", "Trainer"
]
