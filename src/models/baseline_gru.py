"""
Baseline 4: Standard Multimodal Temporal GRU / LSTM

Processes sequential 23-D multimodal vectors using a multi-layer recurrent neural network
with standard unweighted concatenation. Serves as direct temporal baseline against
adaptive reliability fusion.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class StandardTemporalRNN(nn.Module):
    def __init__(self,
                 input_dim: int = 23,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.2,
                 rnn_cell: str = "gru",
                 bidirectional: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_cell_type = rnn_cell.lower()
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if self.rnn_cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_directions, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: (B, T, 23)
        Output:
            logits: (B, num_classes)
        """
        rnn_out, _ = self.rnn(x)  # (B, T, hidden_dim * directions)
        final_h = rnn_out[:, -1, :]  # (B, hidden_dim * directions)
        logits = self.classifier(final_h)
        return logits
