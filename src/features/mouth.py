"""
Mouth Feature Extraction Module

Implements:
- Mouth Aspect Ratio (MAR)
- Mouth openness
- Temporal yawn-like event detection (distinguishing yawning from speech/laughing via duration)
- Yawn duration and frequency over rolling temporal window
"""

import numpy as np
from typing import Dict, List, Optional


def compute_mar(mouth_landmarks: np.ndarray) -> float:
    """
    Computes Mouth Aspect Ratio (MAR).
    Expects landmarks corresponding to inner mouth:
    [left_corner, upper_left, upper_mid, upper_right, right_corner, lower_right, lower_mid, lower_left]
    indices: [0, 1, 2, 3, 4, 5, 6, 7]
    Formula:
        MAR = (||p2 - p6|| + ||p1 - p7|| + ||p3 - p5||) / (2 * ||p0 - p4||)
    """
    if mouth_landmarks is None or len(mouth_landmarks) < 8:
        return 0.0

    # Vertical distances
    dist_v1 = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[6])  # Upper mid to lower mid
    dist_v2 = np.linalg.norm(mouth_landmarks[1] - mouth_landmarks[7])  # Upper left to lower left
    dist_v3 = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[5])  # Upper right to lower right

    # Horizontal distance (corner to corner)
    dist_h = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[4])

    if dist_h < 1e-6:
        return 0.0

    mar = float((dist_v1 + dist_v2 + dist_v3) / (2.0 * dist_h))
    return max(0.0, min(mar, 2.0))


class MouthFeatureTracker:
    """
    Tracks mouth dynamics over time.
    Separates genuine yawns from brief speech openings based on temporal duration:
    - Normal speech opening: 0.1 - 0.5 seconds (brief)
    - Yawn: High MAR (> 0.55) sustained for >= 1.5 seconds
    """

    def __init__(self, mar_yawn_threshold: float = 0.58, min_yawn_duration_sec: float = 1.2,
                 fps: int = 15, window_size: int = 30):
        self.mar_yawn_threshold = mar_yawn_threshold
        self.min_yawn_frames = int(min_yawn_duration_sec * fps)
        self.fps = fps
        self.window_size = window_size

        self.consecutive_open_frames = 0
        self.mar_history: List[float] = []
        self.yawn_timestamps: List[int] = []
        self.total_yawns = 0
        self.current_frame = 0

    def reset(self):
        """Resets tracker state."""
        self.consecutive_open_frames = 0
        self.mar_history.clear()
        self.yawn_timestamps.clear()
        self.total_yawns = 0
        self.current_frame = 0

    def update(self, mouth_pts: Optional[np.ndarray], is_face_valid: bool = True) -> Dict[str, float]:
        """
        Updates tracker with new frame mouth landmarks.
        Returns feature dictionary for current frame.
        """
        self.current_frame += 1

        if not is_face_valid or mouth_pts is None:
            mar = 0.0
            mouth_visibility = 0.0
        else:
            mar = compute_mar(mouth_pts)
            mouth_visibility = 1.0

        is_open = (mar >= self.mar_yawn_threshold)
        is_yawn_event = 0.0
        yawn_duration = 0.0

        if is_open:
            self.consecutive_open_frames += 1
            if self.consecutive_open_frames == self.min_yawn_frames:
                # Trigger yawn registration once duration threshold is crossed
                is_yawn_event = 1.0
                self.yawn_timestamps.append(self.current_frame)
                self.total_yawns += 1
            elif self.consecutive_open_frames > self.min_yawn_frames:
                is_yawn_event = 1.0  # Continuing yawn
        else:
            if self.consecutive_open_frames >= self.min_yawn_frames:
                yawn_duration = float(self.consecutive_open_frames / self.fps)
            self.consecutive_open_frames = 0

        # Maintain rolling window history
        self.mar_history.append(mar)
        if len(self.mar_history) > self.window_size:
            self.mar_history.pop(0)

        # Windowed yawn frequency (yawns per minute)
        recent_yawns = sum(1 for t in self.yawn_timestamps if t >= (self.current_frame - self.window_size))
        window_duration_mins = max(1e-3, (len(self.mar_history) / (self.fps * 60.0)))
        yawn_freq = float(recent_yawns / window_duration_mins)

        # Normalized mouth openness relative to range [0.15, 0.75]
        mouth_openness = float(np.clip((mar - 0.15) / (0.75 - 0.15), 0.0, 1.0))

        return {
            "mar": mar,
            "mouth_openness": mouth_openness,
            "is_yawn_event": is_yawn_event,
            "yawn_duration": yawn_duration,
            "yawn_frequency": yawn_freq,
            "mouth_visibility": mouth_visibility
        }
