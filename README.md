# Adaptive Multi-Feature Temporal Driver Drowsiness Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic machine learning research project investigating whether **adaptive reliability-aware temporal fusion** of multi-facial signals (Eye, Mouth, Head Pose) outperforms conventional eye-only and static multimodal approaches in detecting driver drowsiness under challenging visual conditions.

---

## 1. Research Motivation & Highlights

Traditional drowsiness detection systems rely excessively on eye closures (e.g. static EAR thresholding), rendering them vulnerable to:
- Lighting changes, shadows, and glares
- Occlusions (sunglasses, prescription glasses, hand-to-face gestures)
- Morphological variance in eye shapes across drivers
- Inability to detect cognitive fatigue and head nodding without eye changes

This project implements a software-only, camera-based system that:
1. **Extracts Multi-Facial Signals:**
   - **Eye Branch:** Left/Right/Average EAR, eye openness, windowed blink frequency, PERCLOS.
   - **Mouth Branch:** Mouth Aspect Ratio (MAR), mouth openness, yawn event duration and frequency.
   - **Head Branch:** 3D Head Pose (pitch, yaw, roll via solvePnP), angular velocity, nodding events.
2. **Estimates Dynamic Feature Reliability:** A gating network estimates dynamic weights $(w_{eye}, w_{mouth}, w_{head})$ conditioned on facial tracking confidence and spatial features.
3. **Models Temporal Persistence:** Recurrent temporal fusion (GRU) captures progressive fatigue over rolling windows ($T=30$ frames), preventing single-frame false alarms.
4. **Subject-Independent Validation:** Zero data leakage across subjects using strict 5-fold cross-validation.

---

## 2. Directory Structure

```
AIML Project/
├── README.md
├── requirements.txt
├── research_log.md
│
├── configs/
│   ├── base.yaml          # System & real-time webcam parameters
│   ├── dataset.yaml       # UTA-RLDD settings & preprocessing specifications
│   └── model.yaml         # Architecture & training hyperparameters
│
├── data/
│   ├── raw/               # Raw video files (SubjectXX/0.mp4, 5.mp4, 10.mp4)
│   ├── processed/         # Cached sequence tensors (.npz)
│   └── splits/            # Subject-independent fold definitions
│
├── src/
│   ├── data/              # PyTorch Dataset & DataLoader loaders
│   ├── preprocessing/     # Video frame decoding & MediaPipe landmark extraction
│   ├── features/          # EAR, MAR, Head Pose, and Quality Metric extractors
│   ├── models/            # Baselines, GRU, and Adaptive Fusion networks
│   ├── training/          # Training loop, early stopping, and loss functions
│   ├── evaluation/        # Metrics, confusion matrix, and ablation suites
│   ├── inference/         # Rolling temporal buffer & real-time persistence alert
│   └── utils/             # Config loaders, random seeding, visualization
│
├── experiments/           # Experiment logs & ablation comparisons
├── notebooks/             # Exploratory analysis notebooks
├── scripts/
│   ├── check_dataset.py   # Dataset validation & mock sample generator
│   ├── preprocess.py      # Batch feature extraction & caching
│   ├── train.py           # Training script for baselines and adaptive model
│   ├── evaluate.py        # Evaluation & ablation metric reporting
│   └── realtime.py        # Live webcam drowsiness detector
│
└── results/
    ├── metrics/           # CSV / JSON metric logs
    ├── plots/             # Loss curves, ROC/PR curves, confusion matrices
    └── checkpoints/       # Saved PyTorch model weights (.pt)
```

---

## 3. Installation & Quickstart

### Step 1: Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.4 acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install computer vision, mediapipe, and project dependencies
pip install -r requirements.txt
```

### Step 2: Dataset Verification or Synthetic Mock Generation
To check if the official UTA-RLDD dataset is installed:
```powershell
python scripts/check_dataset.py
```
To generate a lightweight synthetic test dataset to verify the pipeline immediately:
```powershell
python scripts/check_dataset.py --create-mock-sample
```

### Step 3: Preprocess Videos to Cached Feature Tensors
```powershell
python scripts/preprocess.py
```

### Step 4: Train Baseline & Adaptive Models
```powershell
# Train eye-only classical baseline
python scripts/train.py --model baseline_eye

# Train standard multimodal GRU
python scripts/train.py --model baseline_gru

# Train adaptive reliability-aware fusion model
python scripts/train.py --model adaptive_gru
```

### Step 5: Evaluate & Run Ablation Benchmark
```powershell
python scripts/evaluate.py
```

### Step 6: Launch Real-Time Live Webcam Detection
```powershell
python scripts/realtime.py
```

---

## 4. Evaluation Protocol

All models are evaluated on identical subject-independent test splits using:
- **Macro F1-Score & Accuracy**
- **Precision & Recall per class** (Alert, Low Vigilance, Drowsy)
- **False-Alert Frequency & Detection Latency**
- **Inference Latency & FPS** on CPU and GPU
