"""
config.py — Shared constants for the Traffic Police CV system.

Single source of truth for the model path and class colour mapping.
Both the web app (video_processor.py) and the local camera script
(detect_camera.py) import from here so there is no risk of the two
entry-points loading different weights or drifting on box colours.
"""

from pathlib import Path

# ── Model ─────────────────────────────────────────────────────────────────────
# YOLOv8n trained from scratch on the Roboflow-exported YOLO dataset.
# 4 classes: bus, car, motorcycle, truck.
# Held-out test set results:
#   Precision  75.39%  |  Recall  59.90%
#   mAP50      68.20%  |  mAP50-95  48.07%
REPO_ROOT  = Path(__file__).parent          # traffic_police_system/
MODEL_PATH = REPO_ROOT / "checkpoints" / "best.pt"

# YOLO class index → name (matches the order exported by Roboflow)
YOLO_CLASS_NAMES = {
    0: "bus",
    1: "car",
    2: "motorcycle",
    3: "truck",
}

# ── Colours ───────────────────────────────────────────────────────────────────
# BGR tuples for OpenCV drawing.
#   car        → yellow   BGR (0, 255, 255)
#   motorcycle → blue     BGR (255, 0,   0)
#   bus        → black    BGR (0,   0,   0)
#   truck      → red      BGR (0,   0, 220)
CLASS_COLORS_BGR = {
    "car":        (0,   255, 255),   # yellow
    "motorcycle": (255,   0,   0),   # blue
    "bus":        (0,     0,   0),   # black
    "truck":      (0,     0, 220),   # red
}

# Text colour to use on each box background so the label stays readable.
# White on dark boxes (bus=black, motorcycle=blue); black on bright boxes.
CLASS_TEXT_COLORS = {
    "car":        (0,   0,   0),     # black text on yellow
    "motorcycle": (255, 255, 255),   # white text on blue
    "bus":        (255, 255, 255),   # white text on black
    "truck":      (255, 255, 255),   # white text on red
}
