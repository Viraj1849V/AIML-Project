"""
Tracking Quality & Visibility Assessment Module

Computes frame-level and window-level quality metrics:
- Face detection presence and confidence
- Eye visibility and occlusion flags (glasses/hair)
- Mouth visibility and occlusion flags (hands/objects)
- Landmark spatial jitter / stability
- Composite tracking reliability score
"""

import numpy as np
from typing import Dict, List, Optional


class QualityTracker:
    """
    Evaluates reliability of visual signals across frames.
    Supplies quality indicators directly to the adaptive reliability gating module.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.prev_landmarks: Optional[np.ndarray] = None
        self.jitter_history: List[float] = []

    def reset(self):
        """Resets tracker state."""
        self.prev_landmarks = None
        self.jitter_history.clear()

    def evaluate(self, face_detected: bool,
                 face_confidence: float,
                 eye_vis: float,
                 mouth_vis: float,
                 landmarks_2d: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Computes frame quality metrics.
        """
        if not face_detected or landmarks_2d is None:
            self.prev_landmarks = None
            jitter = 1.0  # High jitter when tracking is lost
            composite_quality = 0.0
            face_conf = 0.0
        else:
            face_conf = float(np.clip(face_confidence, 0.0, 1.0))

            # Compute landmark jitter (mean Euclidean displacement of stable facial anchors)
            # Anchors: nose tip, eye outer corners
            anchor_indices = [1, 33, 263]
            valid_anchors = [idx for idx in anchor_indices if idx < len(landmarks_2d)]

            if self.prev_landmarks is not None and len(valid_anchors) == len(anchor_indices):
                curr_anchors = landmarks_2d[valid_anchors]
                prev_anchors = self.prev_landmarks[valid_anchors]
                disp = np.linalg.norm(curr_anchors - prev_anchors, axis=1)
                # Normalize jitter by face scale (e.g. eye distance)
                eye_dist = np.linalg.norm(landmarks_2d[33] - landmarks_2d[263]) if len(landmarks_2d) > 263 else 50.0
                jitter = float(np.mean(disp) / max(10.0, eye_dist))
            else:
                jitter = 0.0

            self.prev_landmarks = landmarks_2d.copy()

            # Composite quality: weighted combination of face confidence, visibility, and low jitter
            jitter_penalty = max(0.0, 1.0 - jitter * 5.0)
            composite_quality = float(0.4 * face_conf + 0.3 * ((eye_vis + mouth_vis) / 2.0) + 0.3 * jitter_penalty)

        self.jitter_history.append(jitter)
        if len(self.jitter_history) > self.window_size:
            self.jitter_history.pop(0)

        return {
            "face_detected": 1.0 if face_detected else 0.0,
            "face_confidence": face_conf,
            "eye_visibility": float(np.clip(eye_vis, 0.0, 1.0)),
            "mouth_visibility": float(np.clip(mouth_vis, 0.0, 1.0)),
            "landmark_jitter": float(np.clip(jitter, 0.0, 1.0)),
            "composite_quality": float(np.clip(composite_quality, 0.0, 1.0))
        }
