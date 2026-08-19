"""
state.py — Shared Thread-safe In-Memory State for Dashboard Backend.
"""

from typing import Optional, Dict, Any, List
import threading
from part2_detection.video_processor import VideoProcessor

class DashboardState:
    def __init__(self):
        self.processor: Optional[VideoProcessor] = None
        self.lock = threading.Lock()
        self.uploaded_video_path: Optional[str] = None
        self.is_demo_mode: bool = True

    def init_processor(self, video_path: Optional[str] = None):
        with self.lock:
            if self.processor is not None:
                self.processor.stop()
            
            # If no video provided, find a test image or sample
            source = video_path or 0
            self.processor = VideoProcessor(video_source=source)
            self.processor.start()

    def get_counts(self) -> Dict[str, Any]:
        if self.processor is None:
            return {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "active_tracked": 0}
        with self.processor.lock:
            counts = dict(self.processor.latest_counts)
            counts["active_tracked"] = self.processor.active_tracked_count
            return counts

    def get_timeline(self) -> List[Dict[str, Any]]:
        if self.processor is None:
            return []
        with self.processor.lock:
            return list(self.processor.counts_timeline)

    def get_violations(self) -> List[Dict[str, Any]]:
        if self.processor is None:
            return []
        return list(self.processor.violations_engine.violation_log)

    def get_accidents(self) -> List[Dict[str, Any]]:
        if self.processor is None:
            return []
        return list(self.accident_detector_logs())

    def accident_detector_logs(self) -> List[Dict[str, Any]]:
        if self.processor is None:
            return []
        return list(self.processor.accident_detector.logged_accidents)


# Global singleton instance
app_state = DashboardState()
