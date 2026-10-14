"""
Evaluation Metrics & Ablation Benchmarking Suite

Computes:
- Accuracy, Precision, Recall, Macro F1, Weighted F1
- Multi-class Confusion Matrix
- Inference Latency (ms) and Throughput (FPS)
- Inspectable Adaptive Reliability Weights under normal & occluded conditions
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Any, Tuple, Optional, List


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist()
    }


def benchmark_inference_speed(model: Any, sample_input: torch.Tensor, device: torch.device,
                              num_warmup: int = 20, num_runs: int = 100) -> Tuple[float, float]:
    """
    Measures mean inference latency in milliseconds and throughput in FPS.
    """
    if hasattr(model, "eval"):
        model.eval()
        x = sample_input.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Timing
            t0 = time.perf_counter()
            for _ in range(num_runs):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        total_time = t1 - t0
        latency_ms = (total_time / num_runs) * 1000.0
        # Sample input has shape (B, T, D) -> frames per run = B * T
        batch_size = sample_input.shape[0]
        seq_len = sample_input.shape[1] if sample_input.dim() == 3 else 1
        fps = (num_runs * batch_size * seq_len) / total_time
        return float(latency_ms), float(fps)
    else:
        # Classical scikit-learn model
        x_np = sample_input.numpy() if isinstance(sample_input, torch.Tensor) else sample_input
        # Warmup
        for _ in range(num_warmup):
            _ = model.predict(x_np)

        t0 = time.perf_counter()
        for _ in range(num_runs):
            _ = model.predict(x_np)
        t1 = time.perf_counter()

        total_time = t1 - t0
        latency_ms = (total_time / num_runs) * 1000.0
        batch_size = x_np.shape[0]
        seq_len = x_np.shape[1] if x_np.ndim == 3 else 1
        fps = (num_runs * batch_size * seq_len) / total_time
        return float(latency_ms), float(fps)


def inspect_adaptive_weights(model: nn.Module, dataloader: Any, device: torch.device) -> Dict[str, Any]:
    """
    Evaluates average learned reliability weights (w_eye, w_mouth, w_head)
    under normal conditions and simulated occlusion stress tests.
    """
    model.eval()
    all_weights = []
    class_weights_dict = {0: [], 1: [], 2: []}

    # Occlusion simulation collectors
    eye_occlusion_weights = []
    mouth_occlusion_weights = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            y = batch["label"].cpu().numpy()

            # 1. Normal conditions
            _, weights = model(x, return_weights=True)
            w_np = weights.cpu().numpy()  # (B, T, 3)
            mean_w_sample = np.mean(w_np, axis=1)  # (B, 3)
            all_weights.append(mean_w_sample)

            for i, label in enumerate(y):
                if label in class_weights_dict:
                    class_weights_dict[label].append(mean_w_sample[i])

            # 2. Simulated Eye Occlusion: set eye features to zero, set eye visibility flag (idx 21) to 0.0
            x_eye_occ = x.clone()
            x_eye_occ[:, :, 0:8] = 0.0  # Zero out eye features
            x_eye_occ[:, :, 21] = 0.0   # Eye visibility = 0.0
            _, w_eye_occ = model(x_eye_occ, return_weights=True)
            eye_occlusion_weights.append(np.mean(w_eye_occ.cpu().numpy(), axis=1))

            # 3. Simulated Mouth Occlusion: set mouth features to zero, mouth visibility flag (idx 22) to 0.0
            x_mouth_occ = x.clone()
            x_mouth_occ[:, :, 8:13] = 0.0
            x_mouth_occ[:, :, 22] = 0.0
            _, w_mouth_occ = model(x_mouth_occ, return_weights=True)
            mouth_occlusion_weights.append(np.mean(w_mouth_occ.cpu().numpy(), axis=1))

    mean_normal = np.mean(np.concatenate(all_weights, axis=0), axis=0)
    mean_eye_occ = np.mean(np.concatenate(eye_occlusion_weights, axis=0), axis=0)
    mean_mouth_occ = np.mean(np.concatenate(mouth_occlusion_weights, axis=0), axis=0)

    per_class_summary = {}
    for c, w_list in class_weights_dict.items():
        if w_list:
            per_class_summary[c] = np.mean(np.array(w_list), axis=0).tolist()

    return {
        "normal_weights": {
            "w_eye": float(mean_normal[0]),
            "w_mouth": float(mean_normal[1]),
            "w_head": float(mean_normal[2])
        },
        "eye_occlusion_weights": {
            "w_eye": float(mean_eye_occ[0]),
            "w_mouth": float(mean_eye_occ[1]),
            "w_head": float(mean_eye_occ[2])
        },
        "mouth_occlusion_weights": {
            "w_eye": float(mean_mouth_occ[0]),
            "w_mouth": float(mean_mouth_occ[1]),
            "w_head": float(mean_mouth_occ[2])
        },
        "per_class_weights": per_class_summary
    }
