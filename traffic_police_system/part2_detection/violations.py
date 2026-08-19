"""
violations.py — Geometry-based Traffic Violation Detection Engine.

Detects:
  1. Wrong-Way Driving: Vehicle velocity vector dot product vs user direction line < -0.6
  2. Stop-Line Crossing: Vehicle centroid crossed user-defined calibration line between frames
  3. Excessive Speed: Real pixel-distance / time calculation (with mandatory camera calibration disclaimer)

All violation events are logged with timestamp, frame number, vehicle ID, and violation type.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import numpy as np
import cv2

sys_time = lambda: time.strftime("%Y-%m-%d %H:%M:%S")


def ccw(A, B, C):
    """Checks if points A, B, C are in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    """Return True if line segments AB and CD intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class ViolationEngine:
    def __init__(
        self,
        stop_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        allowed_direction: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        speed_threshold_px: float = 25.0,  # pixels per frame
    ):
        self.stop_line = stop_line  # ((x1, y1), (x2, y2))
        self.allowed_direction = allowed_direction  # ((start_x, start_y), (end_x, end_y))
        self.speed_threshold_px = speed_threshold_px

        self.violation_log: List[Dict[str, Any]] = []
        self.flagged_vehicles: Dict[int, set] = {}  # vehicle_id -> set of violation types

    def set_stop_line(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        self.stop_line = (p1, p2)

    def set_allowed_direction(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        self.allowed_direction = (p1, p2)

    def check_violations(
        self,
        tracked_vehicles: Dict[int, Any],
        frame_num: int
    ) -> List[Dict[str, Any]]:
        """
        Check all active tracked vehicles for geometry-based violations.
        Returns list of newly detected violation events in this frame.
        """
        new_events = []

        for vid, vehicle in tracked_vehicles.items():
            if vid not in self.flagged_vehicles:
                self.flagged_vehicles[vid] = set()

            # Need at least 2 history points
            if len(vehicle.history) < 2:
                continue

            prev_pt = (vehicle.history[-2][0], vehicle.history[-2][1])
            curr_pt = (vehicle.history[-1][0], vehicle.history[-1][1])

            # 1. Stop-Line Crossing
            if self.stop_line is not None and "STOP_LINE_CROSSING" not in self.flagged_vehicles[vid]:
                p1, p2 = self.stop_line
                if intersect(prev_pt, curr_pt, p1, p2):
                    self.flagged_vehicles[vid].add("STOP_LINE_CROSSING")
                    event = {
                        "timestamp": sys_time(),
                        "frame_num": frame_num,
                        "vehicle_id": vid,
                        "vehicle_type": vehicle.cls_name,
                        "violation_type": "Stop Line Crossing",
                        "severity": "High",
                        "details": f"Vehicle #{vid} ({vehicle.cls_name}) crossed stop line at frame {frame_num}."
                    }
                    self.violation_log.append(event)
                    new_events.append(event)

            # 2. Wrong-Way Driving
            if self.allowed_direction is not None and "WRONG_WAY" not in self.flagged_vehicles[vid]:
                if len(vehicle.history) >= 4:
                    p1, p2 = self.allowed_direction
                    allowed_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
                    norm_allowed = np.linalg.norm(allowed_vec)

                    veh_vec = vehicle.velocity_vector
                    norm_veh = np.linalg.norm(veh_vec)

                    if norm_allowed > 0 and norm_veh > 1.5:
                        cosine_sim = np.dot(allowed_vec, veh_vec) / (norm_allowed * norm_veh)
                        if cosine_sim < -0.55:  # Opposing direction
                            self.flagged_vehicles[vid].add("WRONG_WAY")
                            event = {
                                "timestamp": sys_time(),
                                "frame_num": frame_num,
                                "vehicle_id": vid,
                                "vehicle_type": vehicle.cls_name,
                                "violation_type": "Wrong-Way Driving",
                                "severity": "Critical",
                                "details": f"Vehicle #{vid} moving against defined direction (cos_sim={cosine_sim:.2f})."
                            }
                            self.violation_log.append(event)
                            new_events.append(event)

            # 3. Speeding (Pixel-distance / frame time)
            if "SPEEDING" not in self.flagged_vehicles[vid]:
                if vehicle.speed_px_per_frame > self.speed_threshold_px:
                    self.flagged_vehicles[vid].add("SPEEDING")
                    event = {
                        "timestamp": sys_time(),
                        "frame_num": frame_num,
                        "vehicle_id": vid,
                        "vehicle_type": vehicle.cls_name,
                        "violation_type": "Speeding (Uncalibrated)",
                        "severity": "Medium",
                        "details": f"Vehicle #{vid} speed: {vehicle.speed_px_per_frame:.1f} px/frame (Threshold: {self.speed_threshold_px} px/frame). *Disclaimer: Requires camera calibration for real km/h."
                    }
                    self.violation_log.append(event)
                    new_events.append(event)

        return new_events

    def draw_overlays(self, frame: np.ndarray):
        """Draw calibrated reference lines and direction arrows on the frame."""
        # Draw Stop Line
        if self.stop_line is not None:
            p1, p2 = self.stop_line
            cv2.line(frame, p1, p2, (0, 0, 255), 3)
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2) - 8
            cv2.putText(frame, "STOP LINE", (mid_x - 40, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw Allowed Direction Arrow
        if self.allowed_direction is not None:
            p1, p2 = self.allowed_direction
            cv2.arrowedLine(frame, p1, p2, (0, 255, 0), 3, tipLength=0.2)
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2) - 8
            cv2.putText(frame, "ALLOWED DIRECTION", (mid_x - 60, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
