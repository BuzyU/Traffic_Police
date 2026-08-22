"""
main.py — FastAPI Application Entrypoint.

Starts backend, mounts routers, static files, and initializes video processing.
"""

import sys
from pathlib import Path

SITE_PKGS = Path(__file__).parent.parent / "site_pkgs"
if SITE_PKGS.exists() and str(SITE_PKGS) not in sys.path:
    sys.path.insert(0, str(SITE_PKGS))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import re

TRAFFIC_DATASET_ROOT = os.environ.get("TRAFFIC_DATASET_ROOT", r"C:\Users\Umer Zingu\Desktop\Learning\Traffic_Police\Vehicles-coco.v2i.multiclass")
TRAFFIC_ALLOWED_ORIGINS = os.environ.get("TRAFFIC_ALLOWED_ORIGINS", "http://localhost:8000")

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.routers import stream, counts, violations, accidents, system
from dashboard.state import app_state

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Computer Vision Traffic Police System",
    description="Real-time vehicle counting, geometry-based traffic violations, and accident heuristics.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in TRAFFIC_ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(stream.router)
app.include_router(counts.router)
app.include_router(violations.router)
app.include_router(accidents.router)
app.include_router(system.router)


def secure_filename(filename: str) -> str:
    # strip path separators and null bytes
    name = re.sub(r'[\x00/\\:]', '', filename)
    # strip leading dots
    name = name.lstrip('.')
    return name or "upload.mp4"

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video to process in real-time."""
    sec_name = secure_filename(file.filename)
    ext = os.path.splitext(sec_name)[1].lower()
    
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(status_code=400, detail="Invalid video extension. Allowed: .mp4, .avi, .mov, .mkv, .webm")
        
    dest_path = UPLOAD_DIR / sec_name
    
    # Write and enforce size limit
    MAX_SIZE = 500 * 1024 * 1024
    size = 0
    with open(dest_path, "wb") as buffer:
        while chunk := file.file.read(1024 * 1024): # 1MB chunks
            size += len(chunk)
            if size > MAX_SIZE:
                buffer.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="File too large (max 500 MB)")
            buffer.write(chunk)

    app_state.uploaded_video_path = str(dest_path)
    app_state.init_processor(str(dest_path))

    return {
        "status": "success",
        "message": f"Uploaded and started processing {sec_name}",
        "filepath": str(dest_path)
    }


@app.on_event("startup")
def startup_event():
    # Find any sample video or fallback to webcam / synthetic stream
    test_dir = Path(TRAFFIC_DATASET_ROOT) / "test"
    # Initialize processor with test directory images or demo
    app_state.init_processor(0)


# Mount static frontend
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("Starting Traffic Police CV Dashboard on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
