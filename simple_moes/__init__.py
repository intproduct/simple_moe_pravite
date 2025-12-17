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
    normalize_state_general,
    measure_sigma_z_probs_general,
    pauli_z,
    normalize,
    shape_init,
    mps_init,
    mpo_init,
    get_quantum_layer,
    evolution,
    measurement,
    Quantum_layer_Gate,
)

# Heads
from .heads import (
    ClassificationHead,
    RegressionHead,
    LinearHead,
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
    DenseRegressor,
)

# Utilities
from .tools import (
    smooth,
    set_seed,
    batch_iter,
    evaluate,
    evaluate_classification,
    evaluate_regression,
    evaluate_model,
    MoEInspector,
    flatten_params,
    params_l2_norm,
    params_delta_norm,
    snapshot_params
)

# Train steps
from .trains import (
    linear_warmup,
    learning_schedule,
    train_step_classification,
    train_step_regression,
    Trainer
)

from .debug import setup_logger

__all__ = [
    # experts
    "MLPExpert", "CNNExpert", "RNNExpert",
    # gate
    "softmax_with_temperature", "log_softmax", "topk_routing",
    "SoftGate", "TopKGate", "gate_factory","normalize_state_general", 
    "measure_sigma_z_probs_general", "pauli_z", "normalize", 
    "shape_init", "mps_init", "mpo_init", "get_quantum_layer", 
    "evolution", "measurement", "Quantum_layer_Gate",
    # heads
    "ClassificationHead", "RegressionHead", "LinearHead",
    # losses
    "cross_entropy", "mse_loss", "balance_loss", "accuracy",
    # moes
    "MoElayer", "GenericMoE", "DenseRegressor",
    # tools
    "smooth", "set_seed", "batch_iter", "evaluate", "evaluate_classification", "evaluate_regression", "evaluate_model", "MoEInspector","flatten_params", "params_l2_norm", "params_delta_norm", "snapshot_params",
    # train steps
    "linear_warmup", "learning_schedule", "train_step_classification", "train_step_regression", "Trainer",
    # debug
    "setup_logger",
]
