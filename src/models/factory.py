"""
Model Factory Utility
"""

import torch.nn as nn
from typing import Dict, Any

from src.models.baseline_eye import ClassicalEyeBaseline
from src.models.baseline_static import StaticMultimodalMLP
from src.models.baseline_gru import StandardTemporalRNN
from src.models.adaptive_gru import AdaptiveReliabilityGRU


def build_model(model_name: str, config: Dict[str, Any] = None) -> Any:
    """Instantiates a model architecture from its identifier."""
    name = model_name.lower()
    m_cfg = (config or {}).get("models", {}).get(name, {})

    if name == "baseline_eye":
        return ClassicalEyeBaseline(
            classifier_type=m_cfg.get("classifier", "logistic_regression")
        )
    elif name == "baseline_static":
        return StaticMultimodalMLP(
            input_dim=46,
            hidden_dims=m_cfg.get("hidden_dims", [64, 32]),
            num_classes=3,
            dropout=m_cfg.get("dropout", 0.2)
        )
    elif name in ["baseline_gru", "baseline_lstm"]:
        cell = "lstm" if name == "baseline_lstm" else "gru"
        return StandardTemporalRNN(
            input_dim=m_cfg.get("input_dim", 23),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            num_layers=m_cfg.get("num_layers", 2),
            num_classes=3,
            dropout=m_cfg.get("dropout", 0.2),
            rnn_cell=cell,
            bidirectional=m_cfg.get("bidirectional", False)
        )
    elif name == "adaptive_gru":
        branch_cfg = m_cfg.get("branch_encoders", {})
        temp_cfg = m_cfg.get("temporal_fusion", {})
        gating_cfg = m_cfg.get("reliability_gating", {})
        return AdaptiveReliabilityGRU(
            eye_in_dim=8,
            mouth_in_dim=5,
            head_in_dim=6,
            quality_in_dim=4,
            eye_embed_dim=branch_cfg.get("eye_embed_dim", 32),
            mouth_embed_dim=branch_cfg.get("mouth_embed_dim", 24),
            head_embed_dim=branch_cfg.get("head_embed_dim", 24),
            gating_hidden_dim=gating_cfg.get("hidden_dim", 32),
            rnn_hidden_dim=temp_cfg.get("hidden_dim", 64),
            rnn_layers=temp_cfg.get("num_layers", 2),
            num_classes=3,
            dropout=temp_cfg.get("dropout", 0.2),
            temperature=gating_cfg.get("temperature", 1.0)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available: baseline_eye, baseline_static, baseline_gru, baseline_lstm, adaptive_gru")
