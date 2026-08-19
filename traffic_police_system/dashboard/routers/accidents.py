"""
accidents.py — Accident Incident Logs & Feature Stubs Endpoints.
"""

from fastapi import APIRouter
from dashboard.state import app_state
from part3_advanced.stubs import detect_emergency_vehicle, get_vehicle_model, detect_helmet_violation

router = APIRouter()

@router.get("/api/accidents")
def get_accidents():
    return {
        "status": "success",
        "disclaimer": "Heuristic estimate based on bounding-box overlap + deceleration. Not certified accident detection.",
        "accidents": app_state.get_accidents()
    }


@router.get("/api/emergency-vehicle")
def emergency_vehicle_endpoint():
    """Explicit blocked stub response."""
    return detect_emergency_vehicle()


@router.get("/api/vehicle-model/{vehicle_id}")
def vehicle_model_endpoint(vehicle_id: int):
    """Explicit blocked stub response."""
    return get_vehicle_model()


@router.get("/api/helmet-violation/{vehicle_id}")
def helmet_violation_endpoint(vehicle_id: int):
    """Explicit blocked stub response."""
    return detect_helmet_violation()
