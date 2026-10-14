"""
Comprehensive Evaluation & Ablation Study Script

Runs the standardized evaluation and feature ablation suite across:
A. Eye only classical baseline
B. Eye + Mouth GRU ablation
C. Static Multimodal Fusion (MLP)
D. Temporal Multimodal GRU (unweighted)
E. Adaptive Reliability-Aware Temporal GRU

Outputs:
- Markdown and CSV results table to results/metrics/ablation_results.csv
- Confusion matrix and reliability weight visualizations
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Any

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


def run_ablation_experiment(cfg: Dict[str, Any], device: torch.device, num_folds: int = 3, epochs: int = 25) -> pd.DataFrame:
    npz_path = get_project_root() / "data" / "processed" / "uta_rldd_features.npz"
    splits_dir = get_project_root() / "data" / "splits"
    metrics_dir = get_project_root() / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {"name": "Eye Baseline (Classical)", "model_type": "baseline_eye", "feat_mode": "eye_only"},
        {"name": "Eye + Mouth (Temporal GRU)", "model_type": "baseline_gru", "feat_mode": "eye_mouth", "in_dim": 13},
        {"name": "Static Fusion (Multimodal MLP)", "model_type": "baseline_static", "feat_mode": "static"},
        {"name": "Standard Temporal GRU", "model_type": "baseline_gru", "feat_mode": "all", "in_dim": 23},
        {"name": "Adaptive Reliability GRU", "model_type": "adaptive_gru", "feat_mode": "all"}
    ]

    records = []

    print("\n" + "=" * 75)
    print(f"Executing Controlled Ablation Suite ({num_folds}-Fold Subject-Independent CV)")
    print("=" * 75)

    for exp in experiments:
        exp_name = exp["name"]
        m_type = exp["model_type"]
        feat_mode = exp["feat_mode"]

        fold_accs = []
        fold_precs = []
        fold_recs = []
        fold_f1s = []
        fold_fps = []

        print(f"\n---> Evaluating: {exp_name}")

        for f_idx in range(1, num_folds + 1):
            subjs = load_fold_subjects(splits_dir, fold_idx=f_idx)
            train_subjs, val_subjs, test_subjs = subjs["train"], subjs["val"], subjs["test"]

            if m_type == "baseline_eye":
                # Classical model
                model = build_model("baseline_eye", cfg)
                npz_file = np.load(str(npz_path))
                X_all = npz_file["sequences"]
                y_all = npz_file["labels"]
                subjs_all = npz_file["subjects"]

                train_mask = np.isin(subjs_all, train_subjs)
                test_mask = np.isin(subjs_all, test_subjs)

                model.fit(X_all[train_mask], y_all[train_mask])
                y_pred = model.predict(X_all[test_mask])
                y_test = y_all[test_mask]

                m = compute_classification_metrics(y_test, y_pred)
                _, fps = benchmark_inference_speed(model, torch.tensor(X_all[test_mask][:1], dtype=torch.float32), device=device)

            else:
                train_loader, val_loader, test_loader, class_weights = get_dataloaders(
                    npz_path=npz_path,
                    train_subjects=train_subjs,
                    val_subjects=val_subjs,
                    test_subjects=test_subjs,
                    batch_size=cfg["training"].get("batch_size", 32),
                    feature_mode=feat_mode
                )

                if m_type == "baseline_gru" and exp.get("in_dim") is not None:
                    # Specific ablation dimension
                    from src.models.baseline_gru import StandardTemporalRNN
                    model = StandardTemporalRNN(
                        input_dim=exp["in_dim"],
                        hidden_dim=64,
                        num_layers=2,
                        num_classes=3,
                        dropout=0.2
                    )
                else:
                    model = build_model(m_type, cfg)

                trainer = ModelTrainer(
                    model=model,
                    device=device,
                    learning_rate=cfg["training"].get("learning_rate", 0.001),
                    weight_decay=cfg["training"].get("weight_decay", 0.0001),
                    early_stopping_patience=cfg["training"].get("early_stopping_patience", 10),
                    class_weights=class_weights if cfg["training"]["loss"].get("use_class_weights", True) else None
                )

                trainer.fit(train_loader, val_loader, epochs=epochs)
                _, _, _, y_pred, y_test = trainer.evaluate(test_loader)
                m = compute_classification_metrics(y_test, y_pred)

                sample_batch = next(iter(test_loader))
                _, fps = benchmark_inference_speed(model, sample_batch["x"][:1], device=device)

            fold_accs.append(m["accuracy"])
            fold_precs.append(m["precision_macro"])
            fold_recs.append(m["recall_macro"])
            fold_f1s.append(m["f1_macro"])
            fold_fps.append(fps)

        mean_acc = np.mean(fold_accs) * 100
        mean_prec = np.mean(fold_precs) * 100
        mean_rec = np.mean(fold_recs) * 100
        mean_f1 = np.mean(fold_f1s) * 100
        mean_fps = np.mean(fold_fps)

        print(f"  [Result] Accuracy: {mean_acc:.1f}% | Precision: {mean_prec:.1f}% | Recall: {mean_rec:.1f}% | Macro F1: {mean_f1:.1f}% | FPS: {mean_fps:.0f}")

        records.append({
            "Model / Configuration": exp_name,
            "Accuracy (%)": f"{mean_acc:.2f}",
            "Precision (%)": f"{mean_prec:.2f}",
            "Recall (%)": f"{mean_rec:.2f}",
            "Macro F1 (%)": f"{mean_f1:.2f}",
            "Inference FPS": f"{mean_fps:.1f}"
        })

    df = pd.DataFrame(records)

    # Save to CSV
    csv_path = metrics_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved ablation table to: {csv_path}")

    # Print markdown table
    print("\n" + "=" * 75)
    print("FINAL EXPERIMENTAL RESULTS TABLE")
    print("=" * 75)
    print(df.to_markdown(index=False))
    print("=" * 75)

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate models and generate ablation benchmark")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Model config path")
    parser.add_argument("--folds", type=int, default=3, help="Number of cross-validation folds")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs per fold")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    seed_everything(cfg.get("seed", 42))

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    run_ablation_experiment(cfg=cfg, device=device, num_folds=args.folds, epochs=args.epochs)


if __name__ == "__main__":
    main()
