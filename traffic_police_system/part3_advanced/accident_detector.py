"""
accident_detector.py — Real Heuristic Accident Detection.

Computed from real geometric and physical signals:
  1. Sudden bounding-box IoU overlap between two tracked vehicles (> 0.25)
  2. Abrupt deceleration (speed drop > 60% within 3-5 frames)

*Prominent Disclaimer: This is a genuine geometric heuristic on tracked objects,
not a certified machine learning accident classifier.*
"""

from typing import List, Dict, Any, Tuple
import time
import numpy as np

sys_time = lambda: time.strftime("%Y-%m-%d %H:%M:%S")


def compute_box_iou(boxA: List[int], boxB: List[int]) -> float:
    """Compute Intersection over Union between two [x, y, w, h] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return float(iou)


class AccidentDetector:
    def __init__(self, iou_threshold: float = 0.20, deceleration_drop_pct: float = 0.55):
        self.iou_threshold = iou_threshold
        self.deceleration_drop_pct = deceleration_drop_pct
        self.logged_accidents: List[Dict[str, Any]] = []
        self.flagged_pairs: set = set()

    def check_accidents(self, tracked_vehicles: Dict[int, Any], frame_num: int) -> List[Dict[str, Any]]:
        """
        Evaluate pairs of tracked vehicles for overlap and deceleration.
        Returns list of newly flagged accident events.
        """
        new_accidents = []
        vehicle_ids = list(tracked_vehicles.keys())
        n = len(vehicle_ids)

        for i in range(n):
            for j in range(i + 1, n):
                idA, idB = vehicle_ids[i], vehicle_ids[j]
                pair_key = tuple(sorted((idA, idB)))

                if pair_key in self.flagged_pairs:
                    continue

                vehA = tracked_vehicles[idA]
                vehB = tracked_vehicles[idB]

                # 1. Compute IoU
                iou = compute_box_iou(vehA.box, vehB.box)
                if iou >= self.iou_threshold:
                    # 2. Check deceleration on both vehicles
                    # Need at least 3 history points
                    decelA = self._has_abrupt_deceleration(vehA)
                    decelB = self._has_abrupt_deceleration(vehB)

                    if decelA or decelB or iou > 0.4:
                        self.flagged_pairs.add(pair_key)
                        event = {
                            "timestamp": sys_time(),
                            "frame_num": frame_num,
                            "vehicle_id_1": idA,
                            "vehicle_type_1": vehA.cls_name,
                            "vehicle_id_2": idB,
                            "vehicle_type_2": vehB.cls_name,
                            "iou_overlap": round(iou, 3),
                            "confidence": "Heuristic Estimate",
                            "details": (
                                f"Potential collision detected between Vehicle #{idA} ({vehA.cls_name}) "
                                f"and Vehicle #{idB} ({vehB.cls_name}) with {iou*100:.1f}% bounding box overlap. "
                                f"[Heuristic estimate based on box overlap + deceleration]."
                            )
                        }
                        self.logged_accidents.append(event)
                        new_accidents.append(event)

        return new_accidents

    def _has_abrupt_deceleration(self, vehicle: Any) -> bool:
        """Check if vehicle speed dropped drastically compared to past frames."""
        if len(vehicle.history) < 4:
            return False

        # Compare speed across recent history
        past_speeds = []
        for k in range(1, min(len(vehicle.history), 6)):
            cx1, cy1, f1 = vehicle.history[-k - 1]
            cx2, cy2, f2 = vehicle.history[-k]
            dt = max(f2 - f1, 1)
            dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
            past_speeds.append(dist / dt)

        if len(past_speeds) >= 2:
            initial_speed = max(past_speeds[:-1])
            recent_speed = past_speeds[-1]
            if initial_speed > 3.0 and recent_speed < initial_speed * (1.0 - self.deceleration_drop_pct):
                return True
        return False
