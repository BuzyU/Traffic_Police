"""
stream.py — MJPEG Video Streaming Endpoint.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import time
from dashboard.state import app_state

router = APIRouter()


def generate_frames():
    while True:
        if app_state.processor is not None:
            frame_bytes = app_state.processor.get_jpeg_frame()
            if frame_bytes is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.033)  # ~30 FPS


@router.get("/api/stream")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
