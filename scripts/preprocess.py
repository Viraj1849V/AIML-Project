"""
Batch Preprocessing & Feature Caching Script

Extracts sliding window feature sequences (23-D) from raw video files
and saves cached compressed NPZ tensors for fast, reproducible training.
Also creates subject-independent fold partition files.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_data_root, get_project_root, load_yaml_config
from src.preprocessing.pipeline import VideoPreprocessor
from src.data.split_generator import create_subject_independent_splits
from scripts.check_dataset import match_class_label


def main():
    parser = argparse.ArgumentParser(description="Extract and cache facial feature sequences")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="Path to dataset config")
    parser.add_argument("--data-dir", type=str, default=None, help="Custom dataset root path")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory for cached NPZ files")
    parser.add_argument("--max-subjects", type=int, default=None, help="Optional limit on subjects to process")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else get_data_root()
    output_dir = (get_project_root() / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = (get_project_root() / "data" / "splits").resolve()
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Video Preprocessing & Feature Sequence Extraction")
    print("=" * 60)
    print(f"Source Directory:  {data_dir}")
    print(f"Output Directory:  {output_dir}")
    print(f"Target Sampling:   {cfg['preprocessing']['sample_fps']} FPS")
    print(f"Sequence Length:   {cfg['preprocessing']['sequence_length']} frames")
    print(f"Stride:            {cfg['preprocessing']['stride_train']} frames")

    preprocessor = VideoPreprocessor(
        target_fps=cfg["preprocessing"]["sample_fps"],
        sequence_length=cfg["preprocessing"]["sequence_length"],
        stride=cfg["preprocessing"]["stride_train"],
        min_detection_confidence=cfg["preprocessing"]["face_detection"]["min_detection_confidence"]
    )

    # Discover subject folders
    candidate_folders = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    candidate_folders.sort(key=lambda p: p.name)

    if args.max_subjects:
        candidate_folders = candidate_folders[:args.max_subjects]

    if not candidate_folders:
        print("[ERROR] No subject directories found in data path.")
        sys.exit(1)

    all_sequences: List[np.ndarray] = []
    all_labels: List[int] = []
    all_subjects: List[str] = []
    all_metadata: List[Dict[str, Any]] = []

    print(f"\nProcessing {len(candidate_folders)} subjects...")
    for s_idx, subj_dir in enumerate(candidate_folders, 1):
        subj_name = subj_dir.name
        print(f"\n[{s_idx}/{len(candidate_folders)}] Processing Subject: {subj_name}")

        video_files = list(subj_dir.glob("*.mp4")) + list(subj_dir.glob("*.avi")) + list(subj_dir.glob("*.mov"))
        for vf in video_files:
            detected_class = match_class_label(vf.stem)

            if detected_class is None:
                print(f"  [!] Skipped unrecognized video: {vf.name}")
                continue

            class_name = cfg["dataset"]["classes"].get(detected_class, f"Class {detected_class}")
            print(f"  Extracting features from {vf.name} (Class {detected_class}: {class_name})...")

            try:
                res = preprocessor.process_video(vf, subject_id=subj_name, class_label=detected_class)
                n_seqs = len(res["sequences"])
                print(f"    -> Extracted {n_seqs} sequence windows")

                if n_seqs > 0:
                    all_sequences.append(res["sequences"])
                    all_labels.extend(res["labels"])
                    all_subjects.extend([subj_name] * n_seqs)
                    all_metadata.extend(res["metadata"])
            except Exception as e:
                print(f"  [ERROR] Failed processing {vf.name}: {e}")

    preprocessor.close()

    if not all_sequences:
        print("\n[ERROR] No valid sequences could be extracted.")
        sys.exit(1)

    # Concatenate all sequences
    X = np.concatenate(all_sequences, axis=0)  # Shape: (N, T, 23)
    y = np.array(all_labels, dtype=np.int64)
    subjects_arr = np.array(all_subjects)

    print("\n" + "=" * 60)
    print("Extraction Summary:")
    print(f"  Total Sequence Windows: {len(X)}")
    print(f"  Tensor Shape:           {X.shape} (N_samples, T_timesteps, D_features)")
    print(f"  Class Distribution:     {np.bincount(y)}")

    # Save to compressed NPZ
    out_npz = output_dir / "uta_rldd_features.npz"
    np.savez_compressed(
        str(out_npz),
        sequences=X,
        labels=y,
        subjects=subjects_arr,
        feature_names=np.array(preprocessor.extractor.eye_tracker.__class__.__name__)
    )
    print(f"  Saved cached dataset to: {out_npz} ({out_npz.stat().st_size / (1024*1024):.2f} MB)")

    # Generate subject-independent fold splits
    unique_subjs = sorted(list(set(all_subjects)))
    num_folds = cfg.get("splitting", {}).get("num_folds", 5)
    splits = create_subject_independent_splits(unique_subjs, splits_dir=splits_dir, num_folds=num_folds)
    print(f"\nGenerated {len(splits)} subject-independent folds in {splits_dir}")
    for fold in splits:
        print(f"  Fold {fold['fold']}: Train {fold['train']}, Val {fold['val']}, Test {fold['test']}")

    print("\n[SUCCESS] Preprocessing and caching complete!")


if __name__ == "__main__":
    main()
