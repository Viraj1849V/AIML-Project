"""
Adaptive Reliability-Aware Temporal Multimodal Fusion Network (Adaptive GRU)

Core Research Architecture:
1. Feature Branch Encoders (Eye, Mouth, Head)
2. Dynamic Reliability Gating Network conditioned on branch representations and quality indicators
3. Learned reliability weights (w_eye, w_mouth, w_head)
4. Reliability-weighted multimodal feature fusion
5. Recurrent temporal GRU modeling
6. Inspectable reliability weights for explainability & robustness evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from src.features.aggregator import EYE_INDICES, MOUTH_INDICES, HEAD_INDICES, QUALITY_INDICES


class BranchEncoder(nn.Module):
    """Encodes raw domain features into a normalized latent embedding."""
    def __init__(self, in_features: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReliabilityGatingNetwork(nn.Module):
    """
    Predicts dynamic reliability weights for Eye, Mouth, and Head branches.
    Conditioned on branch embeddings and explicit quality/tracking signals.
    """
    def __init__(self,
                 eye_dim: int = 32,
                 mouth_dim: int = 24,
                 head_dim: int = 24,
                 quality_dim: int = 4,
                 hidden_dim: int = 32,
                 temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        in_dim = eye_dim + mouth_dim + head_dim + quality_dim

        self.gating_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # [w_eye, w_mouth, w_head]
        )

    def forward(self,
                eye_emb: torch.Tensor,
                mouth_emb: torch.Tensor,
                head_emb: torch.Tensor,
                quality: torch.Tensor) -> torch.Tensor:
        """
        Inputs: (B, T, D_branch)
        Output: weights (B, T, 3) normalized via Softmax
        """
        # Concatenate across feature dimension
        concat_input = torch.cat([eye_emb, mouth_emb, head_emb, quality], dim=-1)
        logits = self.gating_mlp(concat_input)  # (B, T, 3)
        weights = F.softmax(logits / self.temperature, dim=-1)
        return weights


class AdaptiveReliabilityGRU(nn.Module):
    """
    Full Adaptive Temporal Fusion Network.
    """
    def __init__(self,
                 eye_in_dim: int = 8,
                 mouth_in_dim: int = 5,
                 head_in_dim: int = 6,
                 quality_in_dim: int = 4,
                 eye_embed_dim: int = 32,
                 mouth_embed_dim: int = 24,
                 head_embed_dim: int = 24,
                 gating_hidden_dim: int = 32,
                 rnn_hidden_dim: int = 64,
                 rnn_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.2,
                 temperature: float = 1.0):
        super().__init__()

        # Branch Encoders
        self.eye_encoder = BranchEncoder(eye_in_dim, eye_embed_dim)
        self.mouth_encoder = BranchEncoder(mouth_in_dim, mouth_embed_dim)
        self.head_encoder = BranchEncoder(head_in_dim, head_embed_dim)

        # Reliability Gating
        self.gating = ReliabilityGatingNetwork(
            eye_dim=eye_embed_dim,
            mouth_dim=mouth_embed_dim,
            head_dim=head_embed_dim,
            quality_dim=quality_in_dim,
            hidden_dim=gating_hidden_dim,
            temperature=temperature
        )

        fused_dim = eye_embed_dim + mouth_embed_dim + head_embed_dim

        # Temporal Recurrent Fusion (GRU)
        self.gru = nn.GRU(
            input_size=fused_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )

        # Final Drowsiness Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self,
                x: torch.Tensor,
                return_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Input:
            x: (B, T, 23) full multimodal sequence tensor
        Output:
            logits: (B, num_classes)
            weights: (B, T, 3) reliability weights [w_eye, w_mouth, w_head] (if return_weights=True)
        """
        # Slice branch features
        eye_raw = x[:, :, EYE_INDICES]
        mouth_raw = x[:, :, MOUTH_INDICES]
        head_raw = x[:, :, HEAD_INDICES]
        quality = x[:, :, QUALITY_INDICES]

        # 1. Project through branch encoders
        e_eye = self.eye_encoder(eye_raw)      # (B, T, 32)
        e_mouth = self.mouth_encoder(mouth_raw)# (B, T, 24)
        e_head = self.head_encoder(head_raw)   # (B, T, 24)

        # 2. Predict dynamic reliability weights
        weights = self.gating(e_eye, e_mouth, e_head, quality)  # (B, T, 3)

        w_eye = weights[:, :, 0:1]      # (B, T, 1)
        w_mouth = weights[:, :, 1:2]    # (B, T, 1)
        w_head = weights[:, :, 2:3]     # (B, T, 1)

        # 3. Weighted feature fusion
        weighted_eye = w_eye * e_eye
        weighted_mouth = w_mouth * e_mouth
        weighted_head = w_head * e_head

        fused = torch.cat([weighted_eye, weighted_mouth, weighted_head], dim=-1)  # (B, T, 80)

        # 4. Temporal modeling
        gru_out, _ = self.gru(fused)  # (B, T, rnn_hidden_dim)
        final_h = gru_out[:, -1, :]   # (B, rnn_hidden_dim)

        # 5. Classification
        logits = self.classifier(final_h)  # (B, num_classes)

        if return_weights:
            return logits, weights
        return logits
