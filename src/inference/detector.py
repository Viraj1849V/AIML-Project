"""
Real-Time Inference Engine with Rolling Temporal Buffer & Persistence Gating

Maintains:
- 30-frame FIFO sliding window of 23-D feature vectors
- Online inference via Adaptive Reliability GRU
- Temporal persistence filter to prevent single-frame false alarms
- Feature contribution / signal level estimation for explainable feedback
"""

import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from src.features.aggregator import MultimodalFeatureExtractor, FEATURE_NAMES
from src.preprocessing.face_detector import FaceMeshDetector
from src.models.adaptive_gru import AdaptiveReliabilityGRU


CLASS_NAMES = {
    0: "ALERT",
    1: "LOW VIGILANCE",
    2: "DROWSY"
}

ALERT_LEVELS = {
    "NORMAL": 0,
    "ADVISORY": 1,
    "DROWSINESS_ALERT": 2,
    "SEVERE_DROWSINESS": 3
}


class RealtimeDrowsinessDetector:
    def __init__(self,
                 checkpoint_path: Optional[Path] = None,
                 sequence_length: int = 30,
                 target_fps: int = 15,
                 persistence_seconds: float = 2.0,
                 severe_seconds: float = 4.0,
                 device: str = "cuda"):
        self.sequence_length = sequence_length
        self.target_fps = target_fps
        self.persistence_frames = int(persistence_seconds * target_fps)
        self.severe_frames = int(severe_seconds * target_fps)

        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

        # Visual modules
        self.face_detector = FaceMeshDetector(min_detection_confidence=0.5)
        self.feature_extractor = MultimodalFeatureExtractor(fps=target_fps, window_size=sequence_length)

        # Load model
        self.model = AdaptiveReliabilityGRU(
            eye_in_dim=8, mouth_in_dim=5, head_in_dim=6, quality_in_dim=4,
            eye_embed_dim=32, mouth_embed_dim=24, head_embed_dim=24,
            gating_hidden_dim=32, rnn_hidden_dim=64, rnn_layers=2, num_classes=3
        ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(str(checkpoint_path), map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[OK] Loaded trained weights from: {checkpoint_path}")
        else:
            print("[INFO] Operating with initialized Adaptive GRU architecture.")

        self.model.eval()

        # Rolling temporal buffer: (sequence_length, 23)
        self.feature_buffer = []

        # Persistence counters
        self.consecutive_drowsy_frames = 0
        self.consecutive_low_vigilance_frames = 0
        self.active_alert_level = ALERT_LEVELS["NORMAL"]

        # Latency tracking
        self.prev_time = time.perf_counter()
        self.current_fps = 0.0

    def reset(self):
        """Resets temporal buffers and state."""
        self.feature_extractor.reset()
        self.feature_buffer.clear()
        self.consecutive_drowsy_frames = 0
        self.consecutive_low_vigilance_frames = 0
        self.active_alert_level = ALERT_LEVELS["NORMAL"]

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single video frame and returns full detection telemetry.
        """
        now = time.perf_counter()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 1e-4:
            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt)

        h, w = frame_bgr.shape[:2]

        # 1. Detect facial landmarks
        landmarks_2d, face_detected, face_conf = self.face_detector.process(frame_bgr)

        # 2. Extract 23-D multimodal feature vector
        feat_vec, feat_dict = self.feature_extractor.process_frame(
            landmarks_2d=landmarks_2d,
            frame_shape=(h, w),
            face_detected=face_detected,
            face_confidence=face_conf
        )

        # 3. Append to rolling window buffer
        self.feature_buffer.append(feat_vec)
        if len(self.feature_buffer) > self.sequence_length:
            self.feature_buffer.pop(0)

        # Default telemetry
        predicted_class = 0
        confidence = 1.0
        probabilities = [1.0, 0.0, 0.0]
        reliability_weights = {"w_eye": 0.33, "w_mouth": 0.33, "w_head": 0.34}

        # 4. Run model inference if buffer has sufficient history
        if len(self.feature_buffer) >= self.sequence_length:
            seq_tensor = torch.tensor(np.array([self.feature_buffer]), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits, weights = self.model(seq_tensor, return_weights=True)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                w_last = weights[0, -1].cpu().numpy()  # Weights at current time step

            probabilities = [float(p) for p in probs]
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
            reliability_weights = {
                "w_eye": float(w_last[0]),
                "w_mouth": float(w_last[1]),
                "w_head": float(w_last[2])
            }

        # 5. Temporal Persistence Filter
        if predicted_class == 2:  # Drowsy
            self.consecutive_drowsy_frames += 1
            self.consecutive_low_vigilance_frames = 0
        elif predicted_class == 1:  # Low Vigilance
            self.consecutive_low_vigilance_frames += 1
            self.consecutive_drowsy_frames = max(0, self.consecutive_drowsy_frames - 1)
        else:  # Alert
            self.consecutive_drowsy_frames = max(0, self.consecutive_drowsy_frames - 2)
            self.consecutive_low_vigilance_frames = max(0, self.consecutive_low_vigilance_frames - 1)

        # Determine alert level
        alert_msg = "NORMAL"
        alert_level = ALERT_LEVELS["NORMAL"]

        if self.consecutive_drowsy_frames >= self.severe_frames:
            alert_msg = "CRITICAL: SEVERE DROWSINESS DETECTED"
            alert_level = ALERT_LEVELS["SEVERE_DROWSINESS"]
        elif self.consecutive_drowsy_frames >= self.persistence_frames:
            alert_msg = "WARNING: PERSISTENT DROWSINESS DETECTED"
            alert_level = ALERT_LEVELS["DROWSINESS_ALERT"]
        elif self.consecutive_low_vigilance_frames >= (self.persistence_frames * 1.5):
            alert_msg = "ADVISORY: EARLY FATIGUE DETECTED"
            alert_level = ALERT_LEVELS["ADVISORY"]

        self.active_alert_level = alert_level

        # Compute descriptive qualitative signals for explainability
        ear_avg = feat_dict.get("ear_avg", 0.0)
        perclos = feat_dict.get("perclos", 0.0)
        mar = feat_dict.get("mar", 0.0)
        is_nod = feat_dict.get("is_nodding_event", 0.0)

        eye_signal = "HIGH CLOSURE" if (perclos > 0.35 or ear_avg < 0.20) else ("MODERATE" if ear_avg < 0.25 else "NORMAL")
        mouth_signal = "YAWNING" if mar > 0.55 else ("ACTIVE" if mar > 0.40 else "NORMAL")
        head_signal = "NODDING" if is_nod > 0.5 else ("TILTED" if abs(feat_dict.get("pitch", 0.0)) > 15 else "STABLE")

        return {
            "predicted_class": predicted_class,
            "class_name": CLASS_NAMES.get(predicted_class, "UNKNOWN"),
            "confidence": confidence,
            "probabilities": probabilities,
            "reliability_weights": reliability_weights,
            "signals": {
                "eye_closure": eye_signal,
                "mouth_state": mouth_signal,
                "head_movement": head_signal,
                "ear": ear_avg,
                "perclos": perclos,
                "mar": mar,
                "pitch": feat_dict.get("pitch", 0.0)
            },
            "alert": {
                "message": alert_msg,
                "level": alert_level,
                "consecutive_drowsy": self.consecutive_drowsy_frames,
                "persistence_target": self.persistence_frames
            },
            "tracking": {
                "face_detected": face_detected,
                "landmarks": landmarks_2d,
                "fps": self.current_fps
            }
        }

    def close(self):
        self.face_detector.close()
