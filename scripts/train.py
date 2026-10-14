"""
Model Training Script for Driver Drowsiness Detection

Supports:
- baseline_eye: Classical Eye-only Classifier (Logistic Regression / SVM)
- baseline_static: Multimodal Static MLP
- baseline_gru: Multimodal Temporal GRU
- adaptive_gru: Adaptive Reliability-Aware Temporal Fusion Network
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_project_root, load_yaml_config
from src.utils.seeding import seed_everything
from src.data.dataset import get_dataloaders
from src.data.split_generator import load_fold_subjects
from src.models.factory import build_model
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import compute_classification_metrics, benchmark_inference_speed, inspect_adaptive_weights


def train_single_model(model_name: str, fold_idx: int, cfg: Dict[str, Any],
                       npz_path: Path, splits_dir: Path, device: torch.device) -> Dict[str, Any]:
    print(f"\n============================================================")
    print(f"Training: {model_name} | Fold {fold_idx} | Device: {device}")
    print(f"============================================================")

    fold_subjects = load_fold_subjects(splits_dir, fold_idx=fold_idx)
    train_subjs = fold_subjects["train"]
    val_subjs = fold_subjects["val"]
    test_subjs = fold_subjects["test"]

    print(f"Subjects - Train: {train_subjs} | Val: {val_subjs} | Test: {test_subjs}")

    # Determine feature mode
    feat_mode = "static" if model_name == "baseline_static" else "all"

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        npz_path=npz_path,
        train_subjects=train_subjs,
        val_subjects=val_subjs,
        test_subjects=test_subjs,
        batch_size=cfg["training"].get("batch_size", 32),
        feature_mode=feat_mode
    )

    model = build_model(model_name, cfg)

    checkpoints_dir = get_project_root() / "results" / "checkpoints"
    save_path = checkpoints_dir / f"{model_name}_fold{fold_idx}_best.pt"

    if model_name == "baseline_eye":
        # Classical model training
        npz_file = np.load(str(npz_path))
        X_all = npz_file["sequences"]
        y_all = npz_file["labels"]
        subjs_all = npz_file["subjects"]

        train_mask = np.isin(subjs_all, train_subjs)
        test_mask = np.isin(subjs_all, test_subjs)

        model.fit(X_all[train_mask], y_all[train_mask])
        y_pred = model.predict(X_all[test_mask])
        y_test = y_all[test_mask]

        metrics = compute_classification_metrics(y_test, y_pred)
        sample_input = torch.tensor(X_all[test_mask][:1], dtype=torch.float32)
        latency_ms, fps = benchmark_inference_speed(model, sample_input, device=device)
        metrics["latency_ms"] = latency_ms
        metrics["fps"] = fps

        print(f"Results for {model_name} (Fold {fold_idx}):")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Macro F1:  {metrics['f1_macro']*100:.2f}%")
        print(f"  Latency:   {metrics['latency_ms']:.3f} ms | FPS: {metrics['fps']:.1f}")
        return metrics

    else:
        # PyTorch Neural Network Training
        trainer = ModelTrainer(
            model=model,
            device=device,
            learning_rate=cfg["training"].get("learning_rate", 0.001),
            weight_decay=cfg["training"].get("weight_decay", 0.0001),
            early_stopping_patience=cfg["training"].get("early_stopping_patience", 10),
            class_weights=class_weights if cfg["training"]["loss"].get("use_class_weights", True) else None,
            label_smoothing=cfg["training"]["loss"].get("label_smoothing", 0.05)
        )

        epochs = cfg["training"].get("epochs", 40)
        history = trainer.fit(train_loader, val_loader, epochs=epochs, save_path=save_path)

        # Final evaluation on unseen test split
        _, test_acc, test_f1, y_pred, y_test = trainer.evaluate(test_loader)
        metrics = compute_classification_metrics(y_test, y_pred)

        # Benchmark latency
        sample_batch = next(iter(test_loader))
        latency_ms, fps = benchmark_inference_speed(model, sample_batch["x"][:1], device=device)
        metrics["latency_ms"] = latency_ms
        metrics["fps"] = fps

        if model_name == "adaptive_gru":
            # Inspect adaptive reliability weights
            weights_report = inspect_adaptive_weights(model, test_loader, device=device)
            metrics["reliability_weights"] = weights_report
            print("\n  Learned Reliability Weights (Test Split):")
            print(f"    Normal:    w_eye={weights_report['normal_weights']['w_eye']:.3f}, "
                  f"w_mouth={weights_report['normal_weights']['w_mouth']:.3f}, "
                  f"w_head={weights_report['normal_weights']['w_head']:.3f}")
            print(f"    Eye Occ:   w_eye={weights_report['eye_occlusion_weights']['w_eye']:.3f}, "
                  f"w_mouth={weights_report['eye_occlusion_weights']['w_mouth']:.3f}, "
                  f"w_head={weights_report['eye_occlusion_weights']['w_head']:.3f}")

        print(f"\nFinal Test Results for {model_name} (Fold {fold_idx}):")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Macro F1:  {metrics['f1_macro']*100:.2f}%")
        print(f"  Recall:    {metrics['recall_macro']*100:.2f}%")
        print(f"  Latency:   {metrics['latency_ms']:.3f} ms | FPS: {metrics['fps']:.1f}")
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Train driver drowsiness models")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Model config path")
    parser.add_argument("--model", type=str, default="adaptive_gru",
                        help="Model to train: baseline_eye, baseline_static, baseline_gru, adaptive_gru, or all")
    parser.add_argument("--fold", type=str, default="1", help="Fold index (1..5) or 'all'")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    seed_everything(cfg.get("seed", 42))

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    npz_path = get_project_root() / "data" / "processed" / "uta_rldd_features.npz"
    splits_dir = get_project_root() / "data" / "splits"

    if not npz_path.exists():
        print(f"[ERROR] Processed feature file not found: {npz_path}")
        print("Please run 'python scripts/preprocess.py' first.")
        sys.exit(1)

    models_to_run = ["baseline_eye", "baseline_static", "baseline_gru", "adaptive_gru"] if args.model == "all" else [args.model]

    fold_list = [1, 2, 3] if args.fold == "all" else [int(args.fold)]

    all_results = {}
    for m in models_to_run:
        all_results[m] = []
        for f in fold_list:
            res = train_single_model(m, fold_idx=f, cfg=cfg, npz_path=npz_path, splits_dir=splits_dir, device=device)
            all_results[m].append(res)

    print("\n" + "=" * 60)
    print("Training Complete across requested models and folds.")


if __name__ == "__main__":
    main()
