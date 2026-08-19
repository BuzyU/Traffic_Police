"""
counts.py — Vehicle Counts REST API.
"""

from fastapi import APIRouter
from dashboard.state import app_state

router = APIRouter()

@router.get("/api/counts")
def get_counts():
    return {
        "status": "success",
        "counts": app_state.get_counts(),
        "timeline": app_state.get_timeline()
    }
