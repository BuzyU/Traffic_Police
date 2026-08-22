"""
detect_camera.py — Live webcam/video detection using the trained YOLOv8n model.

Opens a native OpenCV window (not a browser tab) and runs per-frame inference.
Tracking is enabled via YOLO's built-in ByteTrack so the per-class count shown
in the corner is "unique vehicles tracked so far in this session", not just
"detections in this single frame".  The on-screen label says "tracked" to make
that clear.

Usage
-----
# Default webcam (index 0), confidence threshold 0.4
python detect_camera.py

# Different camera index
python detect_camera.py --source 1

# Video file
python detect_camera.py --source path/to/video.mp4

# Custom confidence threshold
python detect_camera.py --source 0 --conf 0.35

Press  q  to quit cleanly.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set  # Python 3.8-compatible generics
import cv2
import numpy as np
import os

# Suppress ByteTrack's internal "not enough matching points" optical-flow
# warning — it fires when the scene has low texture (e.g. dark backgrounds)
# and is cosmetic: it doesn't affect detection results.
warnings.filterwarnings("ignore", message=".*not enough matching points.*")
warnings.filterwarnings("ignore", message=".*No matching points.*")

# ── path setup ────────────────────────────────────────────────────────────────
# detect_camera.py lives at the repo root.
# traffic_police_system/ is the package root for config, part2_detection, etc.
REPO_ROOT = Path(__file__).resolve().parent
SYS_ROOT  = REPO_ROOT / "traffic_police_system"
if str(SYS_ROOT) not in sys.path:
    sys.path.insert(0, str(SYS_ROOT))

# Tell YOLO where to write its config cache (avoids AppData permission issues).
os.environ["YOLO_CONFIG_DIR"] = str(SYS_ROOT / "runs" / "config")

from ultralytics import YOLO

# Import shared constants — same model path and same colour map as the web app.
from config import CLASS_COLORS_BGR, CLASS_TEXT_COLORS, MODEL_PATH, YOLO_CLASS_NAMES


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live vehicle detection from webcam or video file."
    )
    p.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or path to a video file (default: 0)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS — lower keeps more overlapping boxes (default: 0.45)",
    )
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def resolve_source(raw: str):
    """Return an int camera index or a string file path."""
    try:
        return int(raw)
    except ValueError:
        return raw


def draw_hud(frame: np.ndarray, counts: Dict[str, int], fps: float) -> None:
    """
    Draw a semi-transparent HUD in the top-left corner showing per-class
    cumulative unique-vehicle counts and current FPS.
    """
    card_w, card_h = 240, 170

    # FIX: draw rectangle on overlay copy first, then blend back into frame.
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + card_w, 10 + card_h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(
        frame,
        "TRACKED  (unique this session)",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    y = 55
    for cls_name in ["car", "motorcycle", "bus", "truck"]:
        color = CLASS_COLORS_BGR[cls_name]
        cnt   = counts.get(cls_name, 0)
        # Colour swatch with grey border so the black bus swatch is still visible.
        cv2.rectangle(frame, (18, y - 10), (32, y + 2), color, -1)
        cv2.rectangle(frame, (18, y - 10), (32, y + 2), (160, 160, 160), 1)
        cv2.putText(
            frame,
            f"{cls_name.capitalize()}: {cnt}",
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += 26

    # FPS line
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (18, y + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 255, 120),
        1,
        cv2.LINE_AA,
    )


def draw_boxes(frame: np.ndarray, results) -> None:
    """Draw per-detection bounding boxes and labels on frame."""
    if not results or results[0].boxes is None:
        return

    boxes   = results[0].boxes
    xyxy    = boxes.xyxy.cpu().numpy()
    confs   = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy()
    h_frame = frame.shape[0]

    # ByteTrack IDs may be absent if tracking lost all targets for a frame.
    if boxes.id is not None:
        track_ids = boxes.id.cpu().numpy().astype(int).tolist()
    else:
        track_ids = [None] * len(xyxy)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3])
        cls_name  = YOLO_CLASS_NAMES.get(int(cls_ids[i]), "car")
        conf      = float(confs[i])
        color     = CLASS_COLORS_BGR[cls_name]
        text_color = CLASS_TEXT_COLORS.get(cls_name, (255, 255, 255))
        tid        = track_ids[i]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label text
        if tid is not None:
            label = f"{cls_name.capitalize()} ID:{tid} {conf:.2f}"
        else:
            label = f"{cls_name.capitalize()} {conf:.2f}"

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # FIX: clamp badge top so it never exceeds the frame boundary (y<0 crash).
        badge_top    = max(0, y1 - lh - 8)
        badge_bottom = badge_top + lh + 8

        cv2.rectangle(frame, (x1, badge_top), (x1 + lw + 6, badge_bottom), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, badge_bottom - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )


# ── main loop ─────────────────────────────────────────────────────────────────
def main() -> None:
    args   = parse_args()
    source = resolve_source(args.source)
    conf   = args.conf
    iou    = args.iou

    # Validate model file before doing anything else.
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Copy best.pt into traffic_police_system/checkpoints/ and try again.")
        sys.exit(1)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"Model: {MODEL_PATH.name}  |  Source: {source}")
    print(f"Conf threshold: {conf}  |  IoU threshold: {iou}")
    print("Tip: lower --conf catches more vehicles (try 0.15-0.25); raise to cut false positives.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source '{source}'")
        sys.exit(1)

    # Cumulative unique-vehicle counts for this session.
    # Keyed by class name; tracks set of ByteTrack IDs already counted.
    seen_ids: Dict[str, Set[int]]  = {cls: set() for cls in YOLO_CLASS_NAMES.values()}
    cumulative: Dict[str, int]     = {cls: 0     for cls in YOLO_CLASS_NAMES.values()}

    fps_display                    = 0.0
    last_fps_print                 = time.time()
    frame_times: List[float]       = []

    print("Running — press  q  in the display window to quit.")

    while True:
        t0          = time.time()
        ret, frame  = cap.read()

        if not ret:
            # Loop video files automatically; webcam/stream just ends.
            if isinstance(source, str) and Path(source).is_file():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("\nStream ended or camera disconnected.")
            break

        # Inference with ByteTrack tracking.
        results = model.track(
            frame,
            persist=True,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        # Update cumulative unique counts from track IDs.
        if (results
                and results[0].boxes is not None
                and results[0].boxes.id is not None):
            b        = results[0].boxes
            cls_arr  = b.cls.cpu().numpy()
            id_arr   = b.id.cpu().numpy().astype(int)
            for cls_id, tid in zip(cls_arr, id_arr):
                cls_name = YOLO_CLASS_NAMES.get(int(cls_id), "car")
                if tid not in seen_ids[cls_name]:
                    seen_ids[cls_name].add(int(tid))
                    cumulative[cls_name] += 1

        # Draw per-detection boxes on the frame.
        draw_boxes(frame, results)

        # Measure FPS over a rolling 30-frame window.
        t1 = time.time()
        frame_times.append(t1 - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps_display = 1.0 / (sum(frame_times) / len(frame_times))

        # Print FPS to console once per second (useful if window is minimised).
        now = time.time()
        if now - last_fps_print >= 1.0:
            print(f"FPS: {fps_display:.1f}", end="\r", flush=True)
            last_fps_print = now

        # Draw HUD overlay with cumulative counts and FPS.
        draw_hud(frame, cumulative, fps_display)

        cv2.imshow("Traffic Police — Vehicle Detection (q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release cleanly — no hanging process.
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession totals: {cumulative}")


if __name__ == "__main__":
    main()
