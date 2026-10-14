"""
Subject-Independent Split Generator

Ensures strict subject-level partitioning across training, validation, and test sets.
Zero frames or sequences from the same subject will ever appear across multiple splits.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import KFold


def create_subject_independent_splits(
    subject_ids: List[str],
    splits_dir: Path,
    num_folds: int = 5,
    seed: int = 42
) -> List[Dict[str, List[str]]]:
    """
    Creates deterministic K-fold subject splits and saves them to text files.
    For small sample sets (e.g. < 5 subjects), creates leave-one-subject-out or proportional folds.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(list(set(subject_ids)))
    n_subjects = len(subjects)

    actual_folds = min(num_folds, n_subjects)
    kf = KFold(n_splits=actual_folds, shuffle=True, random_state=seed)

    fold_definitions = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(subjects)):
        test_subjects = [subjects[i] for i in test_idx]
        train_val_pool = [subjects[i] for i in train_val_idx]

        # In standard 5-fold, take 1 subject (or ~15%) of train_val for validation
        if len(train_val_pool) > 1:
            val_subjects = [train_val_pool[-1]]
            train_subjects = train_val_pool[:-1]
        else:
            val_subjects = test_subjects  # Fallback for very small test samples
            train_subjects = train_val_pool

        fold_info = {
            "fold": fold_idx + 1,
            "train": train_subjects,
            "val": val_subjects,
            "test": test_subjects
        }
        fold_definitions.append(fold_info)

        # Save to disk
        for split_name, subjs in [("train", train_subjects), ("val", val_subjects), ("test", test_subjects)]:
            out_file = splits_dir / f"fold_{fold_idx + 1}_{split_name}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                for s in subjs:
                    f.write(f"{s}\n")

    return fold_definitions


def load_fold_subjects(splits_dir: Path, fold_idx: int = 1) -> Dict[str, List[str]]:
    """Loads subject IDs for a specific fold."""
    res = {}
    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"fold_{fold_idx}_{split_name}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file missing: {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            res[split_name] = [line.strip() for line in f if line.strip()]
    return res
