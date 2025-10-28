
import mlx.core as mx
import mlx.nn as nn
from typing import List, Type, Any, Optional

class ClassificationHead(nn.Module):
    def __init__(self, input_dim:int, num_classes:int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, num_classes)
            )

    def __call__(self, x:mx.array)->mx.array:
        logits = self.fc(x)
        return logits
    
class RegressionHead(nn.Module):
    def __init__(self, input_dim:int, output_dim:int=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
            )

    def __call__(self, x:mx.array)->mx.array:
        out = self.fc(x)
        return out
