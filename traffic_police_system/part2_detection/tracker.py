"""
tracker.py — Object Tracking with Centroid + IoU Association.

Maintains persistent vehicle IDs across frames, trajectory history,
and velocity estimates required for real geometry-based traffic violations and accident heuristics.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial.distance import cdist


class TrackedVehicle:
    def __init__(self, vehicle_id: int, box: List[int], cls_name: str, conf: float, color: Tuple[int, int, int]):
        self.id = vehicle_id
        self.box = box  # [x, y, w, h]
        self.cls_name = cls_name
        self.conf = conf
        self.color = color
        
        # Trajectory history: list of (centroid_x, centroid_y, frame_number)
        cx = box[0] + box[2] / 2.0
        cy = box[1] + box[3] / 2.0
        self.history: List[Tuple[float, float, int]] = [(cx, cy, 0)]
        
        self.disappeared = 0
        self.speed_px_per_frame = 0.0
        self.velocity_vector = np.array([0.0, 0.0])

    def update(self, box: List[int], cls_name: str, conf: float, frame_num: int):
        self.box = box
        self.cls_name = cls_name
        self.conf = conf
        self.disappeared = 0

        cx = box[0] + box[2] / 2.0
        cy = box[1] + box[3] / 2.0
        self.history.append((cx, cy, frame_num))

        # Keep history to last 30 frames
        if len(self.history) > 30:
            self.history.pop(0)

        # Update velocity & speed
        if len(self.history) >= 2:
            prev_cx, prev_cy, prev_f = self.history[-2]
            dt = max(frame_num - prev_f, 1)
            dx = (cx - prev_cx) / dt
            dy = (cy - prev_cy) / dt
            self.velocity_vector = np.array([dx, dy])
            self.speed_px_per_frame = float(np.sqrt(dx**2 + dy**2))


class CentroidTracker:
    def __init__(self, max_disappeared: int = 20, max_distance: float = 100.0):
        self.next_id = 1
        self.objects: Dict[int, TrackedVehicle] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections: List[Dict[str, Any]], frame_num: int) -> Dict[int, TrackedVehicle]:
        """
        detections: List of dicts {box: [x,y,w,h], class: str, conf: float, color: tuple}
        """
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].disappeared += 1
                if self.objects[obj_id].disappeared > self.max_disappeared:
                    del self.objects[obj_id]
            return self.objects

        # Compute centroids of new detections
        input_centroids = np.zeros((len(detections), 2), dtype=float)
        for i, det in enumerate(detections):
            x, y, w, h = det["box"]
            input_centroids[i] = [x + w / 2.0, y + h / 2.0]

        # If no tracked objects currently, register all
        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                self._register(det, frame_num)
            return self.objects

        # Pair existing tracked objects with new detections via euclidean distance
        object_ids = list(self.objects.keys())
        object_centroids = np.zeros((len(object_ids), 2), dtype=float)
        for idx, obj_id in enumerate(object_ids):
            last_cx, last_cy, _ = self.objects[obj_id].history[-1]
            object_centroids[idx] = [last_cx, last_cy]

        D = cdist(object_centroids, input_centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            det = detections[col]
            self.objects[obj_id].update(det["box"], det["class"], det["conf"], frame_num)

            used_rows.add(row)
            used_cols.add(col)

        # Unmatched existing objects
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            obj_id = object_ids[row]
            self.objects[obj_id].disappeared += 1
            if self.objects[obj_id].disappeared > self.max_disappeared:
                del self.objects[obj_id]

        # Unmatched new detections → register as new objects
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            self._register(detections[col], frame_num)

        return self.objects

    def _register(self, det: Dict[str, Any], frame_num: int):
        self.objects[self.next_id] = TrackedVehicle(
            self.next_id, det["box"], det["class"], det["conf"], det["color"]
        )
        self.next_id += 1
