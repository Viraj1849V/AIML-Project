"""
Video Preprocessing & Feature Extraction Pipeline

Decodes videos, samples frames at target FPS, extracts facial landmarks and
aggregates 23-D multi-branch feature sequences.
Caches processed sequences into compressed NPZ format with full metadata.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator, Any

from src.preprocessing.face_detector import FaceMeshDetector
from src.features.aggregator import MultimodalFeatureExtractor, FEATURE_NAMES


class VideoPreprocessor:
    """
    Processes video files into sampled frame feature sequences.
    """

    def __init__(self,
                 target_fps: int = 15,
                 sequence_length: int = 30,
                 stride: int = 15,
                 min_detection_confidence: float = 0.5):
        self.target_fps = target_fps
        self.sequence_length = sequence_length
        self.stride = stride
        self.detector = FaceMeshDetector(min_detection_confidence=min_detection_confidence)
        self.extractor = MultimodalFeatureExtractor(fps=target_fps, window_size=sequence_length)

    def process_video(self, video_path: Path, subject_id: str, class_label: int) -> Dict[str, Any]:
        """
        Processes an entire video file and extracts sliding window sequences.
        Returns dictionary containing:
            sequences: np.ndarray (N, sequence_length, 23)
            labels: np.ndarray (N,)
            metadata: List[Dict] with subject, video, timestamps, and sequence IDs
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(native_fps / self.target_fps)))

        self.extractor.reset()
        frame_features: List[np.ndarray] = []
        frame_timestamps: List[float] = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                h, w = frame.shape[:2]
                landmarks_2d, detected, conf = self.detector.process(frame)
                feat_vec, _ = self.extractor.process_frame(
                    landmarks_2d=landmarks_2d,
                    frame_shape=(h, w),
                    face_detected=detected,
                    face_confidence=conf
                )
                frame_features.append(feat_vec)
                frame_timestamps.append(frame_idx / native_fps)

            frame_idx += 1

        cap.release()

        # Build sliding window temporal sequences
        total_frames = len(frame_features)
        sequences: List[np.ndarray] = []
        labels: List[int] = []
        meta_records: List[Dict] = []

        if total_frames >= self.sequence_length:
            for start_idx in range(0, total_frames - self.sequence_length + 1, self.stride):
                end_idx = start_idx + self.sequence_length
                seq_window = np.array(frame_features[start_idx:end_idx], dtype=np.float32)
                sequences.append(seq_window)
                labels.append(class_label)
                meta_records.append({
                    "subject": subject_id,
                    "video": video_path.name,
                    "class": class_label,
                    "start_sec": frame_timestamps[start_idx],
                    "end_sec": frame_timestamps[end_idx - 1],
                    "sequence_id": f"{subject_id}_{video_path.stem}_{start_idx}"
                })

        sequences_arr = np.array(sequences, dtype=np.float32) if sequences else np.empty((0, self.sequence_length, len(FEATURE_NAMES)), dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.int64) if labels else np.empty((0,), dtype=np.int64)

        return {
            "sequences": sequences_arr,
            "labels": labels_arr,
            "metadata": meta_records,
            "feature_names": FEATURE_NAMES
        }

    def close(self):
        self.detector.close()
