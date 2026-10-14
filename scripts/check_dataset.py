"""
Dataset Verification & Setup Utility for UTA-RLDD

This script:
1. Validates local dataset structure against official UTA-RLDD conventions.
2. Checks DATA_ROOT / UTA_RLDD_ROOT environment variables or local paths.
3. Reports missing subjects and class video files.
4. Provides clear instructions on how to request and obtain the dataset legally.
5. NEVER silently downloads tens of gigabytes without explicit authorization.
6. Includes --create-mock-sample to create lightweight synthetic samples for pipeline verification.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_data_root, get_project_root


import re

def match_class_label(filename_or_stem: str) -> Optional[int]:
    """Robustly maps video filenames to class indices (0: Alert, 1: Low Vigilance, 2: Drowsy)."""
    stem = Path(filename_or_stem).stem.lower()
    if stem in ["0", "alert"]:
        return 0
    if stem in ["5", "1", "low_vigilance", "lowvigilant", "lowvigilance"]:
        return 1
    if stem in ["10", "2", "drowsy"]:
        return 2

    tokens = re.split(r'[^a-zA-Z0-9]+', stem)
    if "10" in tokens or "drowsy" in tokens:
        return 2
    if "5" in tokens or "lowvigilance" in tokens or "lowvigilant" in tokens:
        return 1
    if "0" in tokens or "alert" in tokens:
        return 0
    return None



def explain_dataset_access():
    print("\n" + "=" * 70)
    print("UTA-RLDD (UTA Real-Life Drowsiness Dataset) Acquisition Guide")
    print("=" * 70)
    print("Dataset Overview:")
    print("  - Total Subjects: 60 healthy adult participants")
    print("  - Video Count: 180 multi-stage RGB video sequences (~10 minutes each)")
    print("  - Total Storage: ~12 to 18 GB uncompressed video")
    print("  - Target Classes: 0 (Alert), 1 / 5 (Low Vigilance), 2 / 10 (Drowsy)")
    print("\nHow to Obtain:")
    print("  1. Visit official project portal: https://sites.google.com/view/utarldd/home")
    print("  2. Fill out the academic / research end-user agreement (EULA).")
    print("  3. Once access is granted, download the dataset archives.")
    print("  4. Extract the subjects into either:")
    print(f"     a) Default local path: {get_project_root() / 'data' / 'raw'}")
    print("     b) Any external location, and set environment variable:")
    print("        [Windows PowerShell]: $env:DATA_ROOT = 'D:\\Datasets\\UTA-RLDD'")
    print("        [Linux/macOS]:        export DATA_ROOT=/path/to/UTA-RLDD")
    print("\nExpected Directory Layout:")
    print("  <DATA_ROOT>/")
    print("     |-- Subject01/ (or 01/)")
    print("     |    |-- 0.mp4  (Alert)")
    print("     |    |-- 5.mp4  (Low Vigilance)")
    print("     |    +-- 10.mp4 (Drowsy)")
    print("     |-- Subject02/ ...")
    print("     +-- Subject60/")
    print("=" * 70 + "\n")


def check_dataset_integrity(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Inspects target data directory for subjects and class files."""
    stats = {
        "data_dir": str(data_dir),
        "exists": data_dir.exists(),
        "total_subjects_found": 0,
        "complete_subjects": [],
        "incomplete_subjects": [],
        "missing_subjects": [],
        "total_videos_found": 0,
    }

    if not data_dir.exists():
        return False, stats

    # Identify subject folders
    candidate_folders = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    # Sort folders logically if named numerically or SubjectXX
    def parse_subject_num(name: str):
        digits = "".join([c for c in name if c.isdigit()])
        return int(digits) if digits else 999999

    candidate_folders.sort(key=lambda p: parse_subject_num(p.name))

    for subj_dir in candidate_folders:
        video_files = list(subj_dir.glob("*.mp4")) + list(subj_dir.glob("*.avi")) + list(subj_dir.glob("*.mov"))
        class_found = {0: False, 1: False, 2: False}

        for vf in video_files:
            cls_idx = match_class_label(vf.stem)
            if cls_idx is not None and cls_idx in class_found:
                class_found[cls_idx] = True

        stats["total_videos_found"] += len(video_files)
        if all(class_found.values()):
            stats["complete_subjects"].append(subj_dir.name)
        elif any(class_found.values()):
            stats["incomplete_subjects"].append((subj_dir.name, class_found))

    stats["total_subjects_found"] = len(stats["complete_subjects"]) + len(stats["incomplete_subjects"])
    is_ready = len(stats["complete_subjects"]) >= 2
    return is_ready, stats


def create_mock_sample_dataset(output_dir: Path, num_subjects: int = 3, num_frames_per_video: int = 90):
    """
    Creates synthetic sample videos with simulated blinking, yawning, and head movement
    for rapid end-to-end pipeline validation without downloading the full dataset.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("ERROR: OpenCV and NumPy are required to generate synthetic test videos.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating synthetic mock dataset at: {output_dir}")
    print(f"Generating {num_subjects} subjects with 3 drowsiness classes each ({num_frames_per_video} frames @ 15 FPS)...")

    fps = 15
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for s_idx in range(1, num_subjects + 1):
        subj_name = f"Subject{s_idx:02d}"
        subj_dir = output_dir / subj_name
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Classes: 0 = Alert, 5 = Low Vigilance, 10 = Drowsy
        class_configs = {
            "0.mp4": {"state": "ALERT", "blink_prob": 0.05, "yawn": False, "head_nod": False},
            "5.mp4": {"state": "LOW_VIGILANCE", "blink_prob": 0.20, "yawn": True, "head_nod": False},
            "10.mp4": {"state": "DROWSY", "blink_prob": 0.50, "yawn": True, "head_nod": True}
        }

        for filename, cfg in class_configs.items():
            video_path = subj_dir / filename
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            for f in range(num_frames_per_video):
                # Create synthetic facial canvas
                frame = np.ones((height, width, 3), dtype=np.uint8) * 60  # Dark gray background

                # Simulated head center with optional nodding
                cx = width // 2
                cy = height // 2
                if cfg["head_nod"]:
                    cy += int(15 * np.sin(2 * np.pi * f / 30))  # 0.5 Hz cyclic nodding

                # Draw face oval
                cv2.ellipse(frame, (cx, cy), (65, 85), 0, 0, 360, (190, 160, 140), -1)

                # Eye states: open or closed
                is_closed = (np.sin(2 * np.pi * f / 12) > (1.0 - 2.0 * cfg["blink_prob"]))
                eye_h = 2 if is_closed else 8
                # Left eye
                cv2.ellipse(frame, (cx - 24, cy - 15), (12, eye_h), 0, 0, 360, (40, 30, 20), -1)
                # Right eye
                cv2.ellipse(frame, (cx + 24, cy - 15), (12, eye_h), 0, 0, 360, (40, 30, 20), -1)

                # Mouth state: normal or yawning
                is_yawning = cfg["yawn"] and (30 <= f <= 65)
                mouth_h = 22 if is_yawning else 5
                cv2.ellipse(frame, (cx, cy + 35), (18, mouth_h), 0, 0, 360, (20, 20, 80), -1)

                # Simulated nose
                cv2.circle(frame, (cx, cy + 10), 4, (140, 110, 90), -1)

                # Watermark text
                cv2.putText(frame, f"{subj_name} {cfg['state']} f:{f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                writer.write(frame)

            writer.release()
            print(f"  [+] Wrote {video_path.relative_to(get_project_root())}")

    print(f"\n[OK] Synthetic verification dataset successfully created in: {output_dir}")
    print("You can now test preprocessing and baseline model training immediately!")


def main():
    parser = argparse.ArgumentParser(description="UTA-RLDD Dataset Checker & Mock Setup")
    parser.add_argument("--data-dir", type=str, default=None, help="Custom dataset root path")
    parser.add_argument("--explain", action="store_true", help="Print dataset acquisition instructions")
    parser.add_argument("--create-mock-sample", action="store_true", help="Generate synthetic sample dataset for testing")
    parser.add_argument("--mock-subjects", type=int, default=3, help="Number of synthetic subjects to create")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else get_data_root()

    if args.explain:
        explain_dataset_access()
        return

    if args.create_mock_sample:
        create_mock_sample_dataset(data_dir, num_subjects=args.mock_subjects)
        return

    print("=" * 60)
    print("UTA-RLDD Dataset Status Checker")
    print("=" * 60)
    print(f"Target Directory: {data_dir}")
    print(f"DATA_ROOT Env:    {os.environ.get('DATA_ROOT', 'Not Set')}")

    is_ready, stats = check_dataset_integrity(data_dir)

    print(f"\nDirectory Exists:       {stats['exists']}")
    print(f"Total Subjects Found:   {stats['total_subjects_found']}")
    print(f"Complete Subjects (3/3): {len(stats['complete_subjects'])}")
    print(f"Incomplete Subjects:    {len(stats['incomplete_subjects'])}")
    print(f"Total Video Files:      {stats['total_videos_found']}")

    if is_ready:
        print("\n[SUCCESS] Dataset structure is valid and ready for preprocessing.")
        print(f"Found complete subjects: {stats['complete_subjects'][:5]}...")
    else:
        print("\n[NOTICE] UTA-RLDD dataset was not found or is incomplete at this location.")
        print("To proceed:")
        print("  1. Run with --explain for instructions to acquire official UTA-RLDD.")
        print("  2. OR run with --create-mock-sample to create a lightweight verification dataset.")
        print("     Example: python scripts/check_dataset.py --create-mock-sample")


if __name__ == "__main__":
    main()
