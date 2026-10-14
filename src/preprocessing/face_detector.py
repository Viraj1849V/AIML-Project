"""
Facial Landmark & Mesh Detector Wrapper

Uses MediaPipe Tasks FaceLandmarker (with fallback support) to extract
478 3D/2D facial landmarks in pixel coordinates.
Includes confidence scores and graceful error handling.
"""

import os
import cv2
import urllib.request
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


class FaceMeshDetector:
    """
    High-performance facial landmark detector based on MediaPipe FaceLandmarker.
    """

    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._landmarker = None
        self._legacy_mesh = None
        self._init_detector()

    def _ensure_task_model(self) -> Path:
        assets_dir = Path(__file__).resolve().parent / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        model_path = assets_dir / "face_landmarker.task"
        if not model_path.exists() or model_path.stat().st_size < 1000000:
            print("[INFO] Downloading MediaPipe face_landmarker.task model asset...")
            urllib.request.urlretrieve(TASK_MODEL_URL, str(model_path))
            print(f"[OK] Downloaded model asset to {model_path} ({model_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return model_path

    def _init_detector(self):
        # Attempt 1: Modern MediaPipe Tasks API (v0.10+ / v1.0+)
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options

            model_path = self._ensure_task_model()
            options = vision.FaceLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=self.max_num_faces,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_face_blendshapes=False
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self.mp = mp
            return
        except Exception as e:
            pass

        # Attempt 2: Legacy mp.solutions API
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self._legacy_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=self.static_image_mode,
                    max_num_faces=self.max_num_faces,
                    refine_landmarks=self.refine_landmarks,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self.mp = mp
                return
        except Exception:
            pass

        print("[WARN] Neither MediaPipe Tasks nor Legacy FaceMesh could be initialized.")

    def process(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], bool, float]:
        """
        Processes a BGR image frame.
        Returns:
            landmarks_2d: (N, 2) NumPy array of (x, y) pixel coordinates (or None)
            face_detected: bool
            confidence: float [0.0, 1.0]
        """
        if frame_bgr is None:
            return None, False, 0.0

        h, w = frame_bgr.shape[:2]

        # Use Tasks API if available
        if self._landmarker is not None:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
                results = self._landmarker.detect(mp_image)

                if not results.face_landmarks:
                    return None, False, 0.0

                primary_face = results.face_landmarks[0]
                num_pts = len(primary_face)
                coords = np.zeros((num_pts, 2), dtype=np.float32)
                for i, lm in enumerate(primary_face):
                    coords[i, 0] = lm.x * w
                    coords[i, 1] = lm.y * h

                confidence = 0.95
                return coords, True, confidence
            except Exception:
                return None, False, 0.0

        # Use Legacy API if available
        if self._legacy_mesh is not None:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = self._legacy_mesh.process(frame_rgb)
                if not results.multi_face_landmarks:
                    return None, False, 0.0

                primary_face = results.multi_face_landmarks[0]
                num_pts = len(primary_face.landmark)
                coords = np.zeros((num_pts, 2), dtype=np.float32)
                for i, lm in enumerate(primary_face.landmark):
                    coords[i, 0] = lm.x * w
                    coords[i, 1] = lm.y * h

                return coords, True, 0.90
            except Exception:
                return None, False, 0.0

        return None, False, 0.0

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._legacy_mesh is not None:
            self._legacy_mesh.close()
            self._legacy_mesh = None
