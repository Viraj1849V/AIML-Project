"""
Baseline 3: Static Multimodal Baseline (MLP)

A static feedforward neural network operating on window-aggregated multimodal features
(mean and standard deviation of all 23 features across the temporal window = 46 dims).
No temporal recurrent modeling is used, providing a strict baseline for temporal models.
"""

import torch
import torch.nn as nn
from typing import List


class StaticMultimodalMLP(nn.Module):
    def __init__(self,
                 input_dim: int = 46,  # 23 mean + 23 std
                 hidden_dims: List[int] = None,
                 num_classes: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x:
            Either (B, 46) pre-aggregated vector,
            Or (B, T, 23) sequence tensor (mean and std will be computed).
        Output:
            logits: (B, num_classes)
        """
        if x.dim() == 3:
            mean_f = torch.mean(x, dim=1)
            std_f = torch.std(x, dim=1)
            x_stat = torch.cat([mean_f, std_f], dim=-1)
        else:
            x_stat = x

        return self.network(x_stat)
