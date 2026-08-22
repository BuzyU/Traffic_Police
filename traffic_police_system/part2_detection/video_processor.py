"""
video_processor.py — Core Video Processing Loop.

Integrates:
  1. Classical CV localizer + trained classifier
  2. Centroid tracking with persistent vehicle IDs
  3. Real geometry violations (wrong-way, stop-line, speeding)
  4. Heuristic accident detection (overlap + deceleration)
  5. Live annotated frame rendering
"""

from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import torch
from pathlib import Path
import threading
import time
import os
from part1_foundations.localize import CLASS_COLORS_BGR
from part2_detection.tracker import YOLOTrackerWrapper
from part2_detection.violations import ViolationEngine
from part3_advanced.accident_detector import AccidentDetector

# Set YOLO config dir to avoid AppData permissions
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).parent.parent / "runs" / "config")
from ultralytics import YOLO

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"

YOLO_CLASS_NAMES = {
    0: "bus",
    1: "car",
    2: "motorcycle",
    3: "truck"
}


class VideoProcessor:
    def __init__(
        self,
        video_source: Any = 0,
        model_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        yolo_path = model_path or (CKPT_DIR / "best.pt")
        if not yolo_path.exists():
            print(f"[VideoProcessor] WARNING: Trained YOLO model not found at {yolo_path}, falling back to yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(str(yolo_path))

        self.tracker = YOLOTrackerWrapper(max_disappeared=25)
        self.violations_engine = ViolationEngine()
        self.accident_detector = AccidentDetector()

        self.video_source = video_source
        self.cap: Optional[cv2.VideoCapture] = None

        self.is_running = False
        self.current_frame_raw: Optional[np.ndarray] = None
        self.current_frame_annotated: Optional[np.ndarray] = None
        self.frame_number = 0
        self.source_status = "initializing"

        # Real-time state
        self.latest_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        self.active_tracked_count = 0
        self.counts_timeline: List[Dict[str, Any]] = []

        self.lock = threading.Lock()

    def set_calibration(self, stop_line=None, allowed_direction=None):
        with self.lock:
            if stop_line is not None:
                self.violations_engine.set_stop_line(stop_line[0], stop_line[1])
            if allowed_direction is not None:
                self.violations_engine.set_allowed_direction(allowed_direction[0], allowed_direction[1])

    def start(self):
        self.is_running = True
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            with self.lock:
                self.source_status = f"error: cannot open source '{self.video_source}'"
            self.is_running = False
            return
        
        with self.lock:
            self.source_status = "ok"
            
        thread = threading.Thread(target=self._process_loop, daemon=True)
        thread.start()

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

    def _process_loop(self):
        fps_target = 30.0
        frame_interval = 1.0 / fps_target
        
        last_t = time.time()

        while self.is_running and self.cap and self.cap.isOpened():
            t_start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                # Loop video if source is a file
                if isinstance(self.video_source, str) and Path(self.video_source).exists():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            self.frame_number += 1
            annotated_frame = self.process_single_frame(frame, self.frame_number)

            t_end = time.time()
            self.current_fps = 1.0 / max((t_end - last_t), 1e-4)
            last_t = t_end

            with self.lock:
                self.current_frame_raw = frame
                self.current_frame_annotated = annotated_frame

            elapsed = time.time() - t_start
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

    def process_single_frame(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        # 1. YOLO inference + tracking
        results = self.model.track(
            frame, 
            persist=True, 
            conf=0.25, 
            verbose=False, 
            device=0 if self.device.type == "cuda" else "cpu"
        )
        
        raw_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        if results and results[0].boxes:
            for cls_id in results[0].boxes.cls.cpu().numpy():
                c_name = YOLO_CLASS_NAMES.get(int(cls_id), "car")
                raw_counts[c_name] += 1

        # 2. Update tracking wrapper
        tracked_vehicles = self.tracker.update(results, frame_num, YOLO_CLASS_NAMES, CLASS_COLORS_BGR)

        # 3. Check violations
        new_violations = self.violations_engine.check_violations(tracked_vehicles, frame_num)

        # 4. Check accidents
        new_accidents = self.accident_detector.check_accidents(tracked_vehicles, frame_num)

        # 5. Draw visualization on annotated frame
        annotated = frame.copy()

        # Draw calibration lines
        self.violations_engine.draw_overlays(annotated)

        # Draw tracked vehicle bounding boxes and labels
        for vid, vehicle in tracked_vehicles.items():
            x, y, w, h = vehicle.box
            color = vehicle.color

            # Draw vehicle box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw trajectory path
            if len(vehicle.history) >= 2:
                pts = np.array([(int(pt[0]), int(pt[1])) for pt in vehicle.history], np.int32)
                cv2.polylines(annotated, [pts], False, color, 1)

            # Label badge
            label = f"ID:{vid} {vehicle.cls_name.capitalize()} {vehicle.speed_px_per_frame:.0f}px/f"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x, y - lh - 6), (x + lw + 6, y), color, -1)
            text_color = (255, 255, 255) if vehicle.cls_name in ["bus", "motorcycle"] else (0, 0, 0)
            cv2.putText(annotated, label, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        # Draw running count HUD overlay
        with self.lock:
            self.latest_counts = raw_counts
            self.active_tracked_count = len(tracked_vehicles)
            if frame_num % 5 == 0:
                self.counts_timeline.append({
                    "frame": frame_num,
                    "timestamp": time.time(),
                    "counts": dict(raw_counts)
                })
                if len(self.counts_timeline) > 100:
                    self.counts_timeline.pop(0)

        self._draw_stats_overlay(annotated, raw_counts)

        return annotated

    def _draw_stats_overlay(self, frame: np.ndarray, counts: Dict[str, int]):
        """Draw a dashboard legend & running count overlay on the frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Top-left HUD card
        cv2.rectangle(overlay, (10, 10), (220, 140), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "REAL-TIME VEHICLE COUNTS", (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        y_offset = 50
        for cls_name in ["car", "motorcycle", "bus", "truck"]:
            color = CLASS_COLORS_BGR[cls_name]
            cnt = counts[cls_name]
            # Color badge
            cv2.rectangle(frame, (18, y_offset - 10), (32, y_offset + 2), color, -1)
            cv2.rectangle(frame, (18, y_offset - 10), (32, y_offset + 2), (255, 255, 255), 1)
            # Text
            text = f"{cls_name.capitalize()}: {cnt}"
            cv2.putText(frame, text, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            y_offset += 22

    def get_jpeg_frame(self) -> Optional[bytes]:
        with self.lock:
            if self.current_frame_annotated is None:
                return None
            ret, buffer = cv2.imencode(".jpg", self.current_frame_annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                return None
            return buffer.tobytes()
