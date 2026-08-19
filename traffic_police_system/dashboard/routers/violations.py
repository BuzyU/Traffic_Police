"""
violations.py — Violations and Calibration Endpoints.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Tuple, Optional
from dashboard.state import app_state

router = APIRouter()

class CalibrationPayload(BaseModel):
    stop_line: Optional[List[List[int]]] = None  # [[x1, y1], [x2, y2]]
    allowed_direction: Optional[List[List[int]]] = None  # [[x1, y1], [x2, y2]]


@router.get("/api/violations")
def get_violations():
    return {
        "status": "success",
        "violations": app_state.get_violations()
    }


@router.post("/api/calibrate")
def set_calibration(payload: CalibrationPayload):
    sl = None
    ad = None
    if payload.stop_line and len(payload.stop_line) == 2:
        sl = ((payload.stop_line[0][0], payload.stop_line[0][1]),
              (payload.stop_line[1][0], payload.stop_line[1][1]))
    if payload.allowed_direction and len(payload.allowed_direction) == 2:
        ad = ((payload.allowed_direction[0][0], payload.allowed_direction[0][1]),
              (payload.allowed_direction[1][0], payload.allowed_direction[1][1]))

    if app_state.processor:
        app_state.processor.set_calibration(stop_line=sl, allowed_direction=ad)

    return {
        "status": "success",
        "message": "Calibration updated successfully",
        "stop_line": sl,
        "allowed_direction": ad
    }
