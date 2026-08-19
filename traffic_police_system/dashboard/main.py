"""
main.py — FastAPI Application Entrypoint.

Starts backend, mounts routers, static files, and initializes video processing.
"""

import sys
from pathlib import Path

SITE_PKGS = Path(__file__).parent.parent / "site_pkgs"
if SITE_PKGS.exists() and str(SITE_PKGS) not in sys.path:
    sys.path.insert(0, str(SITE_PKGS))

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.routers import stream, counts, violations, accidents
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(stream.router)
app.include_router(counts.router)
app.include_router(violations.router)
app.include_router(accidents.router)


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video to process in real-time."""
    dest_path = UPLOAD_DIR / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    app_state.uploaded_video_path = str(dest_path)
    app_state.init_processor(str(dest_path))

    return {
        "status": "success",
        "message": f"Uploaded and started processing {file.filename}",
        "filepath": str(dest_path)
    }


@app.on_event("startup")
def startup_event():
    # Find any sample video or fallback to webcam / synthetic stream
    test_dir = Path(r"C:\Users\Umer Zingu\Desktop\Learning\Traffic_Police\Vehicles-coco.v2i.multiclass\test")
    # Initialize processor with test directory images or demo
    app_state.init_processor(0)


# Mount static frontend
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("Starting Traffic Police CV Dashboard on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
