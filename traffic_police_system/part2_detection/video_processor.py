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

from part1_foundations.localize import VehiclePipeline, CLASS_COLORS_BGR
from part1_foundations.train_scratch import build_model
from part2_detection.tracker import CentroidTracker
from part2_detection.violations import ViolationEngine
from part3_advanced.accident_detector import AccidentDetector

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"


class VideoProcessor:
    def __init__(
        self,
        video_source: Any = 0,
        model_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)

        self.pipeline = VehiclePipeline(model=self.model, device=self.device, mode="video")
        self.tracker = CentroidTracker(max_disappeared=25, max_distance=120.0)
        self.violations_engine = ViolationEngine()
        self.accident_detector = AccidentDetector()

        self.video_source = video_source
        self.cap: Optional[cv2.VideoCapture] = None

        self.is_running = False
        self.current_frame_raw: Optional[np.ndarray] = None
        self.current_frame_annotated: Optional[np.ndarray] = None
        self.frame_number = 0

        # Real-time state
        self.latest_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        self.active_tracked_count = 0
        self.counts_timeline: List[Dict[str, Any]] = []

        self.lock = threading.Lock()

    def _load_model(self, model_path: Optional[Path]) -> torch.nn.Module:
        model = build_model(simclr_backbone_path=None).to(self.device)
        path = model_path or (CKPT_DIR / "scratch_best.pth")
        if not path.exists():
            path = CKPT_DIR / "pretrained_best.pth"

        if path.exists():
            print(f"[VideoProcessor] Loading model: {path}")
            ckpt = torch.load(path, map_location=self.device)
            model.load_state_dict(ckpt["model"])
        else:
            print("[VideoProcessor] Notice: No checkpoint found, using initial model weights for pipeline structure.")
        model.eval()
        return model

    def set_calibration(self, stop_line=None, allowed_direction=None):
        with self.lock:
            if stop_line is not None:
                self.violations_engine.set_stop_line(stop_line[0], stop_line[1])
            if allowed_direction is not None:
                self.violations_engine.set_allowed_direction(allowed_direction[0], allowed_direction[1])

    def start(self):
        self.is_running = True
        self.cap = cv2.VideoCapture(self.video_source)
        thread = threading.Thread(target=self._process_loop, daemon=True)
        thread.start()

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

    def _process_loop(self):
        fps_target = 30.0
        frame_interval = 1.0 / fps_target

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

            with self.lock:
                self.current_frame_raw = frame
                self.current_frame_annotated = annotated_frame

            elapsed = time.time() - t_start
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

    def process_single_frame(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        # 1. Candidate region extraction + classification
        candidate_boxes = self.pipeline.localizer.extract_candidate_regions(frame)
        detections = []
        raw_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

        for (x, y, w, h) in candidate_boxes:
            crop = frame[y:y+h, x:x+w]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            cls_name, conf, _ = self.pipeline.classify_crop(crop)
            if conf >= 0.25:
                raw_counts[cls_name] += 1
                color = CLASS_COLORS_BGR.get(cls_name, (0, 220, 255))
                detections.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "class": cls_name,
                    "conf": float(conf),
                    "color": color
                })

        # 2. Update tracking
        tracked_vehicles = self.tracker.update(detections, frame_num)

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

        self.pipeline._draw_stats_overlay(annotated, raw_counts)

        return annotated

    def get_jpeg_frame(self) -> Optional[bytes]:
        with self.lock:
            if self.current_frame_annotated is None:
                return None
            ret, buffer = cv2.imencode(".jpg", self.current_frame_annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                return None
            return buffer.tobytes()
