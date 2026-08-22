from fastapi import APIRouter
from dashboard.state import app_state

router = APIRouter(prefix="/api/system", tags=["System"])

@router.get("/status")
def get_system_status():
    status = app_state.get_source_status()
    if app_state.processor is None:
        return {
            "status": status,
            "model_version": "N/A",
            "fps": 0.0,
            "active_trackers": 0,
            "active_classes": []
        }
        
    proc = app_state.processor
    return {
        "status": status,
        "model_version": getattr(proc, "model_version", "YOLOv8"),
        "fps": round(getattr(proc, "current_fps", 0.0), 1),
        "active_trackers": proc.active_tracked_count,
        "active_classes": ["bus", "car", "motorcycle", "truck"]
    }
