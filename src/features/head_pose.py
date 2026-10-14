"""
Head Pose and Orientation Tracking Module

Estimates:
- 3D Head Pose (Pitch, Yaw, Roll) via Perspective-n-Point (solvePnP)
- Temporal derivatives: angular velocities (dPitch/dt, dYaw/dt, dRoll/dt)
- Downward head nodding detection (characteristic micro-sleep head drops)
- Head movement magnitude and duration of unusual orientation
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


# Standard 3D anthropometric face model coordinates (in mm)
# Points: [Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner]
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (landmark 1)
    [0.0, -330.0, -65.0],     # Chin (landmark 199)
    [-225.0, 170.0, -135.0],  # Left eye outer corner (landmark 263)
    [225.0, 170.0, -135.0],   # Right eye outer corner (landmark 33)
    [-150.0, -150.0, -125.0], # Left mouth corner (landmark 291)
    [150.0, -150.0, -125.0]   # Right mouth corner (landmark 61)
], dtype=np.float64)


def estimate_head_pose_angles(image_points: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    Estimates Euler angles (Pitch, Yaw, Roll) in degrees.
    image_points: 6x2 array of 2D pixel coordinates corresponding to FACE_3D_MODEL.
    frame_shape: (height, width)
    """
    if image_points is None or len(image_points) < 6:
        return 0.0, 0.0, 0.0

    height, width = frame_shape
    focal_length = width
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assuming no lens distortion

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL,
        image_points.astype(np.float64),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)

    # Decompose rotation matrix into Euler angles
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = 0.0

    # Convert radians to degrees
    pitch_deg = float(np.degrees(pitch))
    yaw_deg = float(np.degrees(yaw))
    roll_deg = float(np.degrees(roll))

    return pitch_deg, yaw_deg, roll_deg


class HeadPoseTracker:
    """
    Tracks head orientation, nodding events, and angular dynamics over time.
    """

    def __init__(self, fps: int = 15, window_size: int = 30):
        self.fps = fps
        self.window_size = window_size

        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.has_prev = False

        self.pitch_history: List[float] = []
        self.nod_timestamps: List[int] = []
        self.current_frame = 0

    def reset(self):
        """Resets tracker state."""
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.has_prev = False
        self.pitch_history.clear()
        self.nod_timestamps.clear()
        self.current_frame = 0

    def update(self, key_landmarks_2d: Optional[np.ndarray], frame_shape: Tuple[int, int],
               is_face_valid: bool = True) -> Dict[str, float]:
        """
        Updates head pose and returns orientation/velocity metrics.
        """
        self.current_frame += 1

        if not is_face_valid or key_landmarks_2d is None:
            pitch, yaw, roll = 0.0, 0.0, 0.0
            d_pitch, d_yaw, d_roll = 0.0, 0.0, 0.0
            head_visibility = 0.0
        else:
            pitch, yaw, roll = estimate_head_pose_angles(key_landmarks_2d, frame_shape)
            head_visibility = 1.0

            if self.has_prev:
                dt = max(1e-3, 1.0 / self.fps)
                d_pitch = (pitch - self.prev_pitch) / dt
                d_yaw = (yaw - self.prev_yaw) / dt
                d_roll = (roll - self.prev_roll) / dt
            else:
                d_pitch, d_yaw, d_roll = 0.0, 0.0, 0.0
                self.has_prev = True

            self.prev_pitch = pitch
            self.prev_yaw = yaw
            self.prev_roll = roll

        self.pitch_history.append(pitch)
        if len(self.pitch_history) > self.window_size:
            self.pitch_history.pop(0)

        # Detect nodding (cyclic downward pitch oscillation exceeding -15 degrees)
        is_nodding_event = 0.0
        if len(self.pitch_history) >= 10:
            pitch_min = min(self.pitch_history[-10:])
            pitch_range = max(self.pitch_history[-10:]) - pitch_min
            if pitch_min < -12.0 and pitch_range > 10.0:
                is_nodding_event = 1.0
                if self.current_frame not in self.nod_timestamps:
                    self.nod_timestamps.append(self.current_frame)

        # Movement magnitude
        movement_magnitude = float(np.sqrt(d_pitch ** 2 + d_yaw ** 2 + d_roll ** 2))

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "pitch_velocity": float(d_pitch),
            "yaw_velocity": float(d_yaw),
            "roll_velocity": float(d_roll),
            "is_nodding_event": is_nodding_event,
            "movement_magnitude": movement_magnitude,
            "head_visibility": head_visibility
        }
