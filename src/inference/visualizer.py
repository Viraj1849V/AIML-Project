"""
HUD Overlay Visualizer for Real-Time Drowsiness Feedback

Renders:
- Driver state banner & confidence
- Contributing facial signal badges (Eye Closure, Yawning, Nodding)
- Dynamic reliability weights (w_eye, w_mouth, w_head)
- Temporal persistence alert bar
- Live FPS counter
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional


def draw_hud(frame: np.ndarray, telemetry: Dict[str, Any]) -> np.ndarray:
    """
    Draws professional real-time analytics overlay on OpenCV frame.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    state_name = telemetry["class_name"]
    conf = telemetry["confidence"]
    alert_lvl = telemetry["alert"]["level"]
    alert_msg = telemetry["alert"]["message"]
    rel_weights = telemetry["reliability_weights"]
    signals = telemetry["signals"]
    fps = telemetry["tracking"]["fps"]

    # Color scheme (BGR)
    if alert_lvl >= 2:
        theme_color = (40, 40, 230)      # Crimson Red
        bg_bar_color = (20, 20, 140)
    elif state_name == "LOW VIGILANCE" or alert_lvl == 1:
        theme_color = (30, 170, 240)     # Amber Orange
        bg_bar_color = (20, 90, 140)
    else:
        theme_color = (50, 200, 50)      # Bright Green
        bg_bar_color = (20, 80, 20)

    # 1. Top Banner
    cv2.rectangle(canvas, (0, 0), (w, 75), (25, 25, 25), -1)
    cv2.rectangle(canvas, (0, 72), (w, 75), theme_color, -1)

    # State text & confidence
    cv2.putText(canvas, f"DRIVER STATE: {state_name}", (15, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, theme_color, 2)
    cv2.putText(canvas, f"Model Confidence: {conf * 100:.1f}%", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # FPS counter
    cv2.putText(canvas, f"{fps:.1f} FPS", (w - 110, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 220, 160), 1)

    # 2. Contributing Signals Panel (Right side)
    panel_x = w - 240
    panel_y = 90
    panel_w = 225
    panel_h = 160

    # Semi-transparent background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (70, 70, 70), 1)

    cv2.putText(canvas, "FEATURE SIGNALS", (panel_x + 10, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)

    # Signal bars
    cv2.putText(canvas, f"Eye Closure: {signals['eye_closure']}", (panel_x + 10, panel_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(canvas, f"Yawning:     {signals['mouth_state']}", (panel_x + 10, panel_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(canvas, f"Head Pose:   {signals['head_movement']}", (panel_x + 10, panel_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # Numerical metrics
    cv2.putText(canvas, f"EAR: {signals['ear']:.2f} | MAR: {signals['mar']:.2f}", (panel_x + 10, panel_y + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
    cv2.putText(canvas, f"PERCLOS: {signals['perclos'] * 100:.1f}%", (panel_x + 10, panel_y + 148),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    # 3. Dynamic Reliability Weights Panel (Bottom Left)
    rel_x = 15
    rel_y = h - 90
    cv2.rectangle(canvas, (rel_x, rel_y), (rel_x + 280, rel_y + 75), (30, 30, 30), -1)
    cv2.rectangle(canvas, (rel_x, rel_y), (rel_x + 280, rel_y + 75), (70, 70, 70), 1)
    cv2.putText(canvas, "ADAPTIVE RELIABILITY WEIGHTS", (rel_x + 10, rel_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # Horizontal weight meters
    labels = ["Eye", "Mouth", "Head"]
    vals = [rel_weights["w_eye"], rel_weights["w_mouth"], rel_weights["w_head"]]
    colors = [(220, 120, 80), (80, 180, 220), (120, 220, 120)]

    for idx, (lbl, val, col) in enumerate(zip(labels, vals, colors)):
        y_pos = rel_y + 35 + idx * 14
        cv2.putText(canvas, f"{lbl:5s}:", (rel_x + 10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        bar_len = int(val * 140)
        cv2.rectangle(canvas, (rel_x + 65, y_pos - 8), (rel_x + 65 + bar_len, y_pos), col, -1)
        cv2.putText(canvas, f"{val:.2f}", (rel_x + 215, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # 4. Critical Alert Flasher
    if alert_lvl >= 2:
        cv2.rectangle(canvas, (0, h - 50), (w, h), (30, 30, 220), -1)
        cv2.putText(canvas, f"!!! {alert_msg} !!!", (w // 2 - 200, h - 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
    elif alert_lvl == 1:
        cv2.rectangle(canvas, (0, h - 35), (w, h), (20, 120, 200), -1)
        cv2.putText(canvas, alert_msg, (w // 2 - 150, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return canvas
