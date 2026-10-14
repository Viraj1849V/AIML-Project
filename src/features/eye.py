"""
Eye Feature Extraction Module

Implements:
- Eye Aspect Ratio (EAR) for left and right eyes
- Average EAR
- Normalized eye openness
- Blink event detection and duration tracking
- Rolling blink frequency over a temporal window
- PERCLOS (% of time eyes closed over temporal window)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_ear(eye_landmarks: np.ndarray) -> float:
    """
    Computes Eye Aspect Ratio (EAR) given 6 (x, y) landmark coordinates.
    Formula (Soukupová & Čech, 2016):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    if eye_landmarks is None or len(eye_landmarks) < 6:
        return 0.0

    # Vertical landmark Euclidean distances
    dist_v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    dist_v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    # Horizontal landmark Euclidean distance
    dist_h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    if dist_h < 1e-6:
        return 0.0

    ear = float((dist_v1 + dist_v2) / (2.0 * dist_h))
    return max(0.0, min(ear, 1.0))  # Bound between 0 and 1


class EyeFeatureTracker:
    """
    Tracks eye state over a sequence of frames, computing temporal statistics:
    - Blink events and blink duration
    - Windowed blink frequency
    - PERCLOS (Percentage of Eye Closure)
    """

    def __init__(self, ear_closed_threshold: float = 0.21, min_blink_frames: int = 2,
                 max_blink_frames: int = 15, fps: int = 15, window_size: int = 30):
        self.ear_closed_threshold = ear_closed_threshold
        self.min_blink_frames = min_blink_frames
        self.max_blink_frames = max_blink_frames
        self.fps = fps
        self.window_size = window_size

        self.consecutive_closed_frames = 0
        self.ear_history: List[float] = []
        self.closed_history: List[int] = []
        self.blink_timestamps: List[int] = []
        self.total_blinks = 0
        self.current_frame = 0

    def reset(self):
        """Resets tracker state."""
        self.consecutive_closed_frames = 0
        self.ear_history.clear()
        self.closed_history.clear()
        self.blink_timestamps.clear()
        self.total_blinks = 0
        self.current_frame = 0

    def update(self, left_eye_pts: Optional[np.ndarray],
               right_eye_pts: Optional[np.ndarray],
               is_face_valid: bool = True) -> Dict[str, float]:
        """
        Updates tracker with new frame landmarks.
        Returns feature dictionary for current frame.
        """
        self.current_frame += 1

        if not is_face_valid or left_eye_pts is None or right_eye_pts is None:
            # Missing or invalid face: record NaN / fallback and decrement confidence
            left_ear = 0.0
            right_ear = 0.0
            avg_ear = 0.0
            is_closed = 0
            eye_visibility = 0.0
        else:
            left_ear = compute_ear(left_eye_pts)
            right_ear = compute_ear(right_eye_pts)
            avg_ear = float((left_ear + right_ear) / 2.0)
            is_closed = 1 if avg_ear < self.ear_closed_threshold else 0
            eye_visibility = 1.0

        # Update closed frame counter
        is_blink_event = 0.0
        blink_duration = 0.0

        if is_closed:
            self.consecutive_closed_frames += 1
        else:
            if self.min_blink_frames <= self.consecutive_closed_frames <= self.max_blink_frames:
                # Registered a completed blink
                is_blink_event = 1.0
                blink_duration = float(self.consecutive_closed_frames / self.fps)
                self.blink_timestamps.append(self.current_frame)
                self.total_blinks += 1
            self.consecutive_closed_frames = 0

        # Maintain rolling window history
        self.ear_history.append(avg_ear)
        self.closed_history.append(is_closed)
        if len(self.ear_history) > self.window_size:
            self.ear_history.pop(0)
            self.closed_history.pop(0)

        # Compute PERCLOS (% of frames in window where eyes were closed)
        perclos = float(sum(self.closed_history) / max(1, len(self.closed_history)))

        # Compute Blink frequency in window (blinks per minute)
        recent_blinks = sum(1 for t in self.blink_timestamps if t >= (self.current_frame - self.window_size))
        window_duration_mins = max(1e-3, (len(self.ear_history) / (self.fps * 60.0)))
        blink_freq = float(recent_blinks / window_duration_mins)

        # Normalized eye openness (0.0 to 1.0 scaled relative to typical range [0.15, 0.35])
        eye_openness = float(np.clip((avg_ear - 0.15) / (0.35 - 0.15), 0.0, 1.0))

        return {
            "ear_left": left_ear,
            "ear_right": right_ear,
            "ear_avg": avg_ear,
            "eye_openness": eye_openness,
            "is_blink_event": is_blink_event,
            "blink_duration": blink_duration,
            "blink_frequency": blink_freq,
            "perclos": perclos,
            "eye_visibility": eye_visibility
        }
