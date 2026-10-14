"""
Baseline 1: Classical Eye-Only Drowsiness Classifier

Uses classical handcrafted eye metrics:
- Mean EAR
- Minimum EAR
- Windowed PERCLOS
- Blink frequency
Evaluated using Logistic Regression / Support Vector Classifier.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, Any, Tuple


class ClassicalEyeBaseline:
    """
    Classical eye-only baseline classifier.
    """

    def __init__(self, classifier_type: str = "logistic_regression", C: float = 1.0):
        self.classifier_type = classifier_type
        if classifier_type == "svm":
            self.model = SVC(C=C, kernel="rbf", probability=True, random_state=42)
        else:
            self.model = LogisticRegression(C=C, max_iter=1000, random_state=42)

    def extract_features(self, sequences: np.ndarray) -> np.ndarray:
        """
        Extracts summary eye features from (N, T, 23) or (N, T, 8) tensor:
        0: ear_left, 1: ear_right, 2: ear_avg, 3: eye_openness,
        4: is_blink_event, 5: blink_duration, 6: blink_frequency, 7: perclos
        Returns (N, 4) summary matrix:
        [mean_ear, min_ear, final_perclos, max_blink_freq]
        """
        N = sequences.shape[0]
        feats = np.zeros((N, 4), dtype=np.float32)

        for i in range(N):
            seq = sequences[i]
            ear_avg = seq[:, 2]
            blink_freq = seq[:, 6]
            perclos = seq[:, 7]

            feats[i, 0] = float(np.mean(ear_avg))
            feats[i, 1] = float(np.min(ear_avg))
            feats[i, 2] = float(perclos[-1])      # Final PERCLOS over window
            feats[i, 3] = float(np.max(blink_freq))# Peak blink rate

        return feats

    def fit(self, X_seq: np.ndarray, y: np.ndarray):
        X_feats = self.extract_features(X_seq)
        self.model.fit(X_feats, y)
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        X_feats = self.extract_features(X_seq)
        return self.model.predict(X_feats)

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        X_feats = self.extract_features(X_seq)
        return self.model.predict_proba(X_feats)
