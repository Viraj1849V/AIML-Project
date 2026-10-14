"""
PyTorch Dataset & DataLoader for Multimodal Drowsiness Sequences

Loads preprocessed 23-D temporal feature sequences and provides subject-filtered
splits for cross-validation without subject leakage.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

from src.features.aggregator import EYE_INDICES, MOUTH_INDICES, HEAD_INDICES, QUALITY_INDICES, FEATURE_NAMES


class DrowsinessSequenceDataset(Dataset):
    """
    Dataset representing temporal sequences of multi-branch facial features.
    """

    def __init__(self,
                 npz_path: Union[str, Path],
                 split_subjects: Optional[List[str]] = None,
                 feature_mode: str = "all",  # "all", "eye_only", "eye_mouth", "static"
                 transform_mean: Optional[np.ndarray] = None,
                 transform_std: Optional[np.ndarray] = None):
        super().__init__()
        npz_file = np.load(str(npz_path), allow_pickle=True)
        raw_sequences = npz_file["sequences"]  # Shape: (N, T, 23)
        raw_labels = npz_file["labels"]        # Shape: (N,)
        raw_subjects = npz_file["subjects"]    # Shape: (N,)

        # Subject-based filtering
        if split_subjects is not None:
            mask = np.isin(raw_subjects, split_subjects)
            self.sequences = raw_sequences[mask]
            self.labels = raw_labels[mask]
            self.subjects = raw_subjects[mask]
        else:
            self.sequences = raw_sequences
            self.labels = raw_labels
            self.subjects = raw_subjects

        self.feature_mode = feature_mode
        self.transform_mean = transform_mean
        self.transform_std = transform_std

        # Compute or apply normalization
        if self.transform_mean is not None and self.transform_std is not None:
            std_safe = np.where(self.transform_std < 1e-6, 1.0, self.transform_std)
            self.sequences = (self.sequences - self.transform_mean) / std_safe

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates feature-wise mean and std across all time steps."""
        flat = self.sequences.reshape(-1, self.sequences.shape[-1])
        mean = np.mean(flat, axis=0)
        std = np.std(flat, axis=0)
        return mean, std

    def get_class_weights(self) -> torch.Tensor:
        """Calculates balanced class weights: N / (num_classes * count_c)."""
        counts = np.bincount(self.labels, minlength=3)
        total = len(self.labels)
        weights = total / (len(counts) * np.maximum(counts, 1).astype(np.float32))
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]  # (T, 23)
        label = self.labels[idx]

        eye_feat = seq[:, EYE_INDICES]        # (T, 8)
        mouth_feat = seq[:, MOUTH_INDICES]    # (T, 5)
        head_feat = seq[:, HEAD_INDICES]      # (T, 6)
        quality_feat = seq[:, QUALITY_INDICES]# (T, 4)

        # Slice features based on mode
        if self.feature_mode == "eye_only":
            x = eye_feat
        elif self.feature_mode == "eye_mouth":
            x = np.concatenate([eye_feat, mouth_feat], axis=-1)
        elif self.feature_mode == "static":
            # For static baseline: aggregate mean and std across time window T -> (23 * 2 = 46)
            mean_feat = np.mean(seq, axis=0)
            std_feat = np.std(seq, axis=0)
            x = np.concatenate([mean_feat, std_feat], axis=-1)
        else:
            x = seq  # Full 23-D sequence

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "eye": torch.tensor(eye_feat, dtype=torch.float32),
            "mouth": torch.tensor(mouth_feat, dtype=torch.float32),
            "head": torch.tensor(head_feat, dtype=torch.float32),
            "quality": torch.tensor(quality_feat, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "subject": str(self.subjects[idx])
        }


def get_dataloaders(
    npz_path: Union[str, Path],
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
    batch_size: int = 32,
    feature_mode: str = "all"
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Constructs train, validation, and test DataLoaders for a specific fold split.
    Normalizes validation and test sets using training set statistics only.
    """
    # Initialize train dataset to obtain feature mean and std
    train_ds_raw = DrowsinessSequenceDataset(npz_path, split_subjects=train_subjects, feature_mode=feature_mode)
    mean, std = train_ds_raw.get_stats()
    class_weights = train_ds_raw.get_class_weights()

    # Re-instantiate with normalization
    train_ds = DrowsinessSequenceDataset(npz_path, split_subjects=train_subjects, feature_mode=feature_mode,
                                         transform_mean=mean, transform_std=std)
    val_ds = DrowsinessSequenceDataset(npz_path, split_subjects=val_subjects, feature_mode=feature_mode,
                                       transform_mean=mean, transform_std=std)
    test_ds = DrowsinessSequenceDataset(npz_path, split_subjects=test_subjects, feature_mode=feature_mode,
                                        transform_mean=mean, transform_std=std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_weights
