"""
PyTorch Training Engine

Handles:
- Training and validation epochs
- Balanced loss weighting & label smoothing
- AdamW optimizer and learning rate scheduling
- Early stopping based on validation loss / macro F1
- Checkpoint persistence
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score


class ModelTrainer:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001,
                 early_stopping_patience: int = 10,
                 class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.05):
        self.model = model.to(device)
        self.device = device
        self.early_stopping_patience = early_stopping_patience

        # Criterion with class weighting
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-5
        )

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in dataloader:
            x = batch["x"].to(self.device)
            y = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            logits = out[0] if isinstance(out, tuple) else out

            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())

        avg_loss = total_loss / max(1, len(all_targets))
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        return avg_loss, acc, macro_f1

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in dataloader:
            x = batch["x"].to(self.device)
            y = batch["label"].to(self.device)

            out = self.model(x)
            logits = out[0] if isinstance(out, tuple) else out

            loss = self.criterion(logits, y)
            total_loss += loss.item() * len(y)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())

        avg_loss = total_loss / max(1, len(all_targets))
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        return avg_loss, acc, macro_f1, np.array(all_preds), np.array(all_targets)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 50,
            save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Trains model with early stopping.
        """
        best_val_f1 = -1.0
        best_weights = None
        patience_counter = 0
        history = {
            "train_loss": [], "train_acc": [], "train_f1": [],
            "val_loss": [], "val_acc": [], "val_f1": []
        }

        for epoch in range(1, epochs + 1):
            t_loss, t_acc, t_f1 = self.train_epoch(train_loader)
            v_loss, v_acc, v_f1, _, _ = self.evaluate(val_loader)
            self.scheduler.step()

            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)
            history["train_f1"].append(t_f1)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)
            history["val_f1"].append(v_f1)

            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_weights, str(save_path))
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                break

        # Restore best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return history
