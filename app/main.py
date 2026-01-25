"""StreamDiffusion Video-to-Video Demo - FastAPI Application."""

import asyncio
import io
import logging
import mimetypes
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException, Request, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from app.config import config, MODELS, DEFAULT_MODEL, DEFAULT_PROMPT, PROMPT_PRESETS, DEFAULT_PRESET
from app.pipeline import get_pipeline
from app.video_source import get_available_videos, get_video_path, VideoSource
from app.video_processor import get_processor, JobStatus

# Fix mime type on Windows
mimetypes.add_type("application/javascript", ".js")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response models
class ProcessRequest(BaseModel):
    """Request to process a video."""
    video_name: Optional[str] = Field(default=None, description="Server video filename")
    prompt: str = Field(default=DEFAULT_PROMPT, description="Generation prompt")
    model: str = Field(default=DEFAULT_MODEL, description="Model to use")


class SettingsResponse(BaseModel):
    """API settings response."""
    models: dict
    default_model: str
    default_prompt: str
    presets: dict
    default_preset: str
    width: int
    height: int


class App:
    """Main application class."""

    def __init__(self):
        self.app = FastAPI(title="StreamDiffusion Video-to-Video Demo")
        self._setup_middleware()
        self._setup_routes()
        self._setup_static()

    def _setup_middleware(self):
        """Configure CORS middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Set up API routes."""

        @self.app.get("/api/settings")
        async def get_settings() -> SettingsResponse:
            """Get application settings."""
            pipeline = get_pipeline()
            # Build presets dict with prompt and description
            presets = {
                key: {"prompt": preset.prompt, "description": preset.description}
                for key, preset in PROMPT_PRESETS.items()
            }
            return SettingsResponse(
                models=pipeline.get_available_models(),
                default_model=DEFAULT_MODEL,
                default_prompt=DEFAULT_PROMPT,
                presets=presets,
                default_preset=DEFAULT_PRESET,
                width=config.width,
                height=config.height,
            )

        @self.app.get("/api/videos")
        async def list_videos():
            """List available server-side videos."""
            return JSONResponse({"videos": get_available_videos()})

        @self.app.post("/api/upload")
        async def upload_video(file: UploadFile = File(...)):
            """Upload a video file for processing."""
            processor = get_processor()

            # Validate file type
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")

            allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            ext = Path(file.filename).suffix.lower()
            if ext not in allowed_extensions:
                raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_extensions}")

            # Save uploaded file
            upload_id = str(uuid.uuid4())[:8]
            safe_filename = f"{upload_id}_{file.filename}"
            upload_path = processor.uploads_dir / safe_filename

            try:
                with open(upload_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            finally:
                file.file.close()

            # Get video info
            cap = cv2.VideoCapture(str(upload_path))
            if not cap.isOpened():
                os.remove(upload_path)
                raise HTTPException(status_code=400, detail="Could not read video file")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            return JSONResponse({
                "filename": safe_filename,
                "original_name": file.filename,
                "path": str(upload_path),
                "duration": round(duration, 1),
                "fps": round(fps, 1),
                "width": width,
                "height": height,
                "frames": frames,
            })

        @self.app.post("/api/process")
        async def process_video(
            background_tasks: BackgroundTasks,
            video_name: str = Form(None),
            uploaded_file: str = Form(None),
            prompt: str = Form(DEFAULT_PROMPT),
            model: str = Form(DEFAULT_MODEL),
        ):
            """Start processing a video."""
            processor = get_processor()
            pipeline = get_pipeline()

            # Determine input path
            if uploaded_file:
                input_path = str(processor.uploads_dir / uploaded_file)
                if not os.path.exists(input_path):
                    raise HTTPException(status_code=404, detail="Uploaded file not found")
            elif video_name:
                input_path = get_video_path(video_name)
                if not input_path:
                    raise HTTPException(status_code=404, detail="Video not found")
            else:
                raise HTTPException(status_code=400, detail="No video specified")

            # Create job
            try:
                job = processor.create_job(
                    input_path=input_path,
                    prompt=prompt,
                    model=model,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            # Start processing in background
            background_tasks.add_task(processor.process_job, job.job_id, pipeline)

            return JSONResponse(job.to_dict())

        @self.app.get("/api/job/{job_id}")
        async def get_job_status(job_id: str):
            """Get the status of a processing job."""
            processor = get_processor()
            job = processor.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            return JSONResponse(job.to_dict())

        @self.app.get("/api/jobs")
        async def list_jobs():
            """List all processing jobs."""
            processor = get_processor()
            return JSONResponse({"jobs": processor.get_all_jobs()})

        @self.app.get("/api/output/{filename}")
        async def get_output_video(filename: str):
            """Serve a processed output video."""
            processor = get_processor()

            # Security: validate filename
            if ".." in filename or "/" in filename or "\\" in filename:
                raise HTTPException(status_code=400, detail="Invalid filename")

            output_path = processor.outputs_dir / filename
            if not output_path.exists():
                raise HTTPException(status_code=404, detail="Output not found")

            return FileResponse(
                path=str(output_path),
                media_type="video/mp4",
                filename=filename,
            )

        @self.app.get("/api/input/{filename}")
        async def get_input_video(filename: str):
            """Serve an input video (from uploads or videos dir)."""
            processor = get_processor()

            # Security: validate filename
            if ".." in filename or "/" in filename or "\\" in filename:
                raise HTTPException(status_code=400, detail="Invalid filename")

            # Check uploads dir first
            upload_path = processor.uploads_dir / filename
            if upload_path.exists():
                return FileResponse(
                    path=str(upload_path),
                    media_type="video/mp4",
                    filename=filename,
                )

            # Check videos dir
            video_path = get_video_path(filename)
            if video_path:
                return FileResponse(
                    path=video_path,
                    media_type="video/mp4",
                    filename=filename,
                )

            raise HTTPException(status_code=404, detail="Video not found")

        @self.app.delete("/api/job/{job_id}")
        async def delete_job(job_id: str):
            """Delete a job and its output file."""
            processor = get_processor()
            job = processor.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            # Remove output file if exists
            if os.path.exists(job.output_path):
                os.remove(job.output_path)

            # Remove preview files
            input_preview = processor.preview_dir / f"{job_id}_input.jpg"
            output_preview = processor.preview_dir / f"{job_id}_output.jpg"
            if input_preview.exists():
                os.remove(input_preview)
            if output_preview.exists():
                os.remove(output_preview)

            del processor.jobs[job_id]
            return JSONResponse({"status": "deleted"})

        @self.app.get("/api/preview/{filename}")
        async def get_preview_frame(filename: str):
            """Serve a preview frame image for real-time visualization."""
            processor = get_processor()

            # Security: validate filename
            if ".." in filename or "/" in filename or "\\" in filename:
                raise HTTPException(status_code=400, detail="Invalid filename")

            preview_path = processor.preview_dir / filename
            if not preview_path.exists():
                raise HTTPException(status_code=404, detail="Preview not found")

            # Add cache-busting header
            return FileResponse(
                path=str(preview_path),
                media_type="image/jpeg",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

    def _setup_static(self):
        """Set up static file serving."""
        frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")

        if os.path.exists(frontend_dir):
            self.app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")
        else:
            # Fallback: serve a simple HTML page
            @self.app.get("/")
            async def index():
                return JSONResponse({
                    "message": "StreamDiffusion Video-to-Video Demo API",
                    "docs": "/docs",
                    "note": "Frontend not built. Run 'npm run build' in frontend directory."
                })


# Create application instance
app_instance = App()
app = app_instance.app


if __name__ == "__main__":
    import uvicorn

    # Initialize pipeline on startup
    logger.info("Initializing pipeline...")
    get_pipeline()
    logger.info("Pipeline ready!")

    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )
