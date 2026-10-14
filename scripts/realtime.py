"""
Real-Time Webcam & Video Driver Drowsiness Detection Application

Runs:
1. Video frame capture (Webcam or Video file)
2. Face landmark detection (MediaPipe FaceLandmarker)
3. 23-D feature sequence tracking (Eye, Mouth, Head, Quality)
4. Rolling temporal buffer
5. Adaptive Reliability-Aware GRU inference
6. Real-time HUD visualization with dynamic weights & alert persistence
"""

import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_project_root, load_yaml_config
from src.inference.detector import RealtimeDrowsinessDetector
from src.inference.visualizer import draw_hud


def main():
    parser = argparse.ArgumentParser(description="Real-Time Driver Drowsiness Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file instead of webcam")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/adaptive_gru_fold1_best.pt",
                        help="Trained model checkpoint path")
    parser.add_argument("--persistence", type=float, default=2.0, help="Alert persistence duration in seconds")
    parser.add_argument("--no-gui", action="store_true", help="Run in console/headless mode without OpenCV window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for testing)")
    args = parser.parse_args()

    ckpt_path = get_project_root() / args.checkpoint
    if not ckpt_path.exists():
        ckpt_path = None  # Fallback to initialized weights if fold checkpoint not found

    print("=" * 65)
    print("Adaptive Multi-Feature Driver Drowsiness Detection: Live Mode")
    print("=" * 65)
    print(f"Video Source:    {'Webcam (' + str(args.camera) + ')' if args.video is None else args.video}")
    print(f"Checkpoint:      {ckpt_path if ckpt_path else 'Default Initialized Weights'}")
    print(f"Persistence:     {args.persistence:.1f} seconds")
    print("Press 'q' or 'ESC' in the window to quit.\n")

    detector = RealtimeDrowsinessDetector(
        checkpoint_path=ckpt_path,
        sequence_length=30,
        target_fps=15,
        persistence_seconds=args.persistence,
        severe_seconds=4.0
    )

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.video if args.video else args.camera}")
        print("Tip: If you do not have an active physical webcam, test on a video file with --video data/raw/Subject01/10.mp4")
        sys.exit(1)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] End of video stream reached.")
                break

            frame_count += 1
            telemetry = detector.process_frame(frame)

            if not args.no_gui:
                hud_frame = draw_hud(frame, telemetry)
                cv2.imshow("Adaptive Driver Drowsiness Detector", hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
            else:
                # Console mode output
                if frame_count % 15 == 0:
                    state = telemetry["class_name"]
                    conf = telemetry["confidence"]
                    w = telemetry["reliability_weights"]
                    sig = telemetry["signals"]
                    alert = telemetry["alert"]["message"]
                    print(f"[Frame {frame_count:4d}] State: {state:14s} ({conf*100:4.1f}%) | "
                          f"Signals: Eye={sig['eye_closure']:12s} Yawn={sig['mouth_state']:8s} | "
                          f"Weights: E={w['w_eye']:.2f} M={w['w_mouth']:.2f} H={w['w_head']:.2f} | "
                          f"Alert: {alert}")

            if args.max_frames and frame_count >= args.max_frames:
                break

    finally:
        cap.release()
        if not args.no_gui:
            cv2.destroyAllWindows()
        detector.close()

    elapsed = time.time() - start_time
    avg_fps = frame_count / max(1e-3, elapsed)
    print(f"\nSession terminated. Processed {frame_count} frames in {elapsed:.2f}s (Average {avg_fps:.1f} FPS).")


if __name__ == "__main__":
    main()
