"""
Unified Multimodal Feature Aggregator

Extracts and collates:
- Eye Branch (8 features)
- Mouth Branch (5 features)
- Head Branch (6 features)
- Quality Control (4 features)
Total = 23-dimensional feature vector per frame.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from src.features.eye import EyeFeatureTracker
from src.features.mouth import MouthFeatureTracker
from src.features.head_pose import HeadPoseTracker
from src.features.quality import QualityTracker


FEATURE_NAMES = [
    # Eye Branch (0-7)
    "ear_left", "ear_right", "ear_avg", "eye_openness",
    "is_blink_event", "blink_duration", "blink_frequency", "perclos",
    # Mouth Branch (8-12)
    "mar", "mouth_openness", "is_yawn_event", "yawn_duration", "yawn_frequency",
    # Head Branch (13-18)
    "pitch", "yaw", "roll", "pitch_velocity", "yaw_velocity", "roll_velocity",
    # Quality Control (19-22)
    "face_detected", "face_confidence", "eye_visibility", "mouth_visibility"
]

EYE_INDICES = list(range(0, 8))
MOUTH_INDICES = list(range(8, 13))
HEAD_INDICES = list(range(13, 19))
QUALITY_INDICES = list(range(19, 23))


class MultimodalFeatureExtractor:
    """
    Orchestrates the individual tracking modules and returns a structured 23-D vector.
    """

    def __init__(self, fps: int = 15, window_size: int = 30):
        self.fps = fps
        self.window_size = window_size
        self.eye_tracker = EyeFeatureTracker(fps=fps, window_size=window_size)
        self.mouth_tracker = MouthFeatureTracker(fps=fps, window_size=window_size)
        self.head_tracker = HeadPoseTracker(fps=fps, window_size=window_size)
        self.quality_tracker = QualityTracker(window_size=window_size)

    def reset(self):
        """Resets all internal trackers."""
        self.eye_tracker.reset()
        self.mouth_tracker.reset()
        self.head_tracker.reset()
        self.quality_tracker.reset()

    def process_frame(self,
                      landmarks_2d: Optional[np.ndarray],
                      frame_shape: Tuple[int, int],
                      face_detected: bool = True,
                      face_confidence: float = 1.0,
                      left_eye_indices: Optional[List[int]] = None,
                      right_eye_indices: Optional[List[int]] = None,
                      mouth_indices: Optional[List[int]] = None,
                      pnp_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extracts 23-dimensional feature vector for a single frame.
        """
        if left_eye_indices is None:
            left_eye_indices = [362, 385, 387, 263, 373, 380]
        if right_eye_indices is None:
            right_eye_indices = [33, 160, 158, 133, 153, 144]
        if mouth_indices is None:
            # Inner mouth landmarks
            mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178]
        if pnp_indices is None:
            # Nose, Chin, Eye outer corners, Mouth corners
            pnp_indices = [1, 199, 263, 33, 291, 61]

        is_valid = face_detected and (landmarks_2d is not None) and (len(landmarks_2d) > 0)

        # Slice branch landmarks
        if is_valid:
            max_idx = max(max(left_eye_indices), max(right_eye_indices),
                          max(mouth_indices), max(pnp_indices))
            if len(landmarks_2d) <= max_idx:
                is_valid = False

        left_eye_pts = landmarks_2d[left_eye_indices] if is_valid else None
        right_eye_pts = landmarks_2d[right_eye_indices] if is_valid else None
        mouth_pts = landmarks_2d[mouth_indices] if is_valid else None
        pnp_pts = landmarks_2d[pnp_indices] if is_valid else None

        # Update trackers
        eye_metrics = self.eye_tracker.update(left_eye_pts, right_eye_pts, is_face_valid=is_valid)
        mouth_metrics = self.mouth_tracker.update(mouth_pts, is_face_valid=is_valid)
        head_metrics = self.head_tracker.update(pnp_pts, frame_shape, is_face_valid=is_valid)
        quality_metrics = self.quality_tracker.evaluate(
            face_detected=is_valid,
            face_confidence=face_confidence,
            eye_vis=eye_metrics["eye_visibility"],
            mouth_vis=mouth_metrics["mouth_visibility"],
            landmarks_2d=landmarks_2d if is_valid else None
        )

        # Assemble unified feature dictionary
        full_dict = {**eye_metrics, **mouth_metrics, **head_metrics, **quality_metrics}

        # Pack into ordered 23-D NumPy array
        vector = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        for i, name in enumerate(FEATURE_NAMES):
            vector[i] = full_dict.get(name, 0.0)

        return vector, full_dict
