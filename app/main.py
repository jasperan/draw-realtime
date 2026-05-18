"""StreamDiffusion Video-to-Video Demo - FastAPI Application."""

import json
import logging
import mimetypes
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import config, MODELS, DEFAULT_MODEL, DEFAULT_PROMPT, PROMPT_PRESETS, DEFAULT_PRESET
from app.pipeline import get_pipeline
from app.video_source import get_available_videos, get_video_path
from app.video_processor import get_processor
from app.multistyle import get_multistyle_processor, STYLES as MULTISTYLE_STYLES
from app.monarchrt_pipeline import is_monarchrt_available

# Fix mime type on Windows
mimetypes.add_type("application/javascript", ".js")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Upload stream chunk size: 1 MB. Small enough to bail fast on oversize uploads,
# large enough not to dominate per-request latency.
_UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB


def _safe_path(base_dir: Path, filename: str) -> Path:
    """Validate `filename` and return its absolute path under `base_dir`.

    Rejects path separators, parent references, and NUL bytes up front
    (cheap check that matches existing test expectations), then verifies
    the resolved path is actually contained in base_dir (defense against
    symlink tricks and edge-case OS quirks).
    """
    if (
        not filename
        or ".." in filename
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
    ):
        raise HTTPException(status_code=400, detail="Invalid filename")

    try:
        base_resolved = Path(base_dir).resolve()
        candidate = (base_resolved / filename).resolve()
        candidate.relative_to(base_resolved)
    except (ValueError, OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return candidate


def _safe_child_path(base_dir: Path, filename: str) -> Path:
    """Return a direct child path under `base_dir` for internal file handles."""
    if not filename or "/" in filename or "\\" in filename or "\x00" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    try:
        base_resolved = Path(base_dir).resolve()
        candidate = (base_resolved / filename).resolve()
        candidate.relative_to(base_resolved)
    except (ValueError, OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return candidate


def _remove_if_exists(path: Path | str) -> None:
    """Best-effort removal for files that may already be gone."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _resolve_input_video_path(
    processor,
    *,
    uploaded_file: Optional[str],
    video_name: Optional[str],
) -> str:
    """Resolve the selected input video from uploads or the server library."""
    if uploaded_file:
        upload_path = _safe_child_path(processor.uploads_dir, uploaded_file)
        if not upload_path.exists():
            raise HTTPException(status_code=404, detail="Uploaded file not found")
        return str(upload_path)

    if video_name:
        input_path = get_video_path(video_name)
        if not input_path:
            raise HTTPException(status_code=404, detail="Video not found")
        return input_path

    raise HTTPException(status_code=400, detail="No video specified")


def _probe_video_metadata(video_path: Path) -> Optional[dict]:
    """Return basic metadata for a valid video, or None for corrupt files."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-show_entries", "format=duration",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    streams = payload.get("streams") or []
    if not streams:
        return None

    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration_raw = (payload.get("format") or {}).get("duration")

    if width <= 0 or height <= 0:
        return None

    try:
        duration = float(duration_raw) if duration_raw is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0

    return {"width": width, "height": height, "duration": duration}


def _generate_video_thumbnail(video_path: Path, thumbnail_path: Path) -> bool:
    """Generate a thumbnail from the first usable frame of a video."""
    commands = (
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "thumbnail,scale=320:-1",
            "-frames:v", "1",
            str(thumbnail_path),
        ],
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "scale=320:-1",
            "-frames:v", "1",
            str(thumbnail_path),
        ],
    )

    for command in commands:
        if thumbnail_path.exists():
            _remove_if_exists(thumbnail_path)
        try:
            result = subprocess.run(command, capture_output=True, timeout=10)
        except Exception:
            continue

        if result.returncode == 0 and thumbnail_path.exists() and thumbnail_path.stat().st_size > 0:
            return True

    if thumbnail_path.exists() and thumbnail_path.stat().st_size == 0:
        _remove_if_exists(thumbnail_path)

    return False


# Request/Response models
class ProcessRequest(BaseModel):
    """Request to process a video."""
    video_name: Optional[str] = Field(default=None, description="Server video filename")
    prompt: str = Field(default=DEFAULT_PROMPT, description="Generation prompt")
    model: str = Field(default=DEFAULT_MODEL, description="Model to use")


class GenerateRequest(BaseModel):
    """Request to generate a video from text (MonarchRT)."""
    prompt: str = Field(description="Text description of the video to generate")
    model: str = Field(default="monarchrt-sf", description="MonarchRT model to use")
    num_frames: int = Field(default=21, description="Number of frames to generate")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")


class SettingsResponse(BaseModel):
    """API settings response."""
    models: dict
    default_model: str
    default_prompt: str
    presets: dict
    default_preset: str
    width: int
    height: int
    monarchrt_available: bool = False


class App:
    """Main application class."""

    def __init__(self):
        self.app = FastAPI(title="StreamDiffusion Video-to-Video Demo")
        self._setup_middleware()
        self._setup_routes()
        self._setup_static()

    def _setup_middleware(self):
        """Configure CORS middleware."""
        # Per CORS spec, wildcard origins + credentials is invalid and browsers
        # will drop the credentials header anyway. Don't pretend we support
        # credentialed cross-origin requests when we don't.
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
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
                monarchrt_available=is_monarchrt_available(),
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

            # Build target path with a UUID prefix so user-supplied filenames
            # can never collide or escape the uploads dir.
            upload_id = str(uuid.uuid4())[:8]
            sanitized_name = Path(file.filename).name  # strip any path components
            safe_filename = f"{upload_id}_{sanitized_name}"
            upload_path = processor.uploads_dir / safe_filename

            # Stream to disk in chunks and bail as soon as we exceed the size
            # limit. Previous implementation wrote the full upload first and
            # then checked the size, which let attackers fill the disk.
            bytes_written = 0
            try:
                try:
                    with open(upload_path, "wb") as buffer:
                        while True:
                            chunk = await file.read(_UPLOAD_CHUNK_SIZE)
                            if not chunk:
                                break
                            bytes_written += len(chunk)
                            if bytes_written > MAX_UPLOAD_SIZE:
                                raise HTTPException(
                                    status_code=413,
                                    detail=(
                                        f"File too large. Maximum: "
                                        f"{MAX_UPLOAD_SIZE // (1024 * 1024)} MB"
                                    ),
                                )
                            buffer.write(chunk)
                finally:
                    await file.close()
            except HTTPException:
                _remove_if_exists(upload_path)
                raise

            # Get video info
            cap = cv2.VideoCapture(str(upload_path))
            if not cap.isOpened():
                _remove_if_exists(upload_path)
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

            input_path = _resolve_input_video_path(
                processor,
                uploaded_file=uploaded_file,
                video_name=video_name,
            )

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

        @self.app.post("/api/generate")
        async def generate_video(
            background_tasks: BackgroundTasks,
            prompt: str = Form(...),
            model: str = Form("monarchrt-sf"),
            num_frames: int = Form(21),
            seed: int = Form(-1),
        ):
            """Generate a video from text using MonarchRT.

            This endpoint creates videos from text prompts without requiring
            an input video. Only works with MonarchRT models.
            """
            processor = get_processor()
            pipeline = get_pipeline()

            # Validate model is MonarchRT
            if model not in MODELS:
                raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

            model_config = MODELS[model]
            if model_config.pipeline_type != "monarchrt":
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model}' is not a MonarchRT model. Use /api/process for video-to-video.",
                )

            if not is_monarchrt_available():
                raise HTTPException(
                    status_code=503,
                    detail="MonarchRT is not installed. See MonarchRT/README.md for setup.",
                )

            # Create generation job (no input video)
            fps = 16.0 if model_config.monarchrt_mode == "causal" else 24.0
            job = processor.create_generate_job(
                prompt=prompt,
                model=model,
                num_frames=num_frames,
                fps=fps,
            )

            # Start generation in background
            background_tasks.add_task(processor.process_generate_job, job.job_id, pipeline)

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
            output_path = _safe_path(processor.outputs_dir, filename)
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
            upload_path = _safe_path(processor.uploads_dir, filename)
            if upload_path.exists():
                return FileResponse(
                    path=str(upload_path),
                    media_type="video/mp4",
                    filename=filename,
                )

            # Check videos dir (get_video_path already validates containment).
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

            # Best-effort cleanup of files on disk. Missing files are fine.
            for path in (
                job.output_path,
                str(processor.preview_dir / f"{job_id}_input.jpg"),
                str(processor.preview_dir / f"{job_id}_output.jpg"),
            ):
                _remove_if_exists(path)

            processor.delete_job(job_id)
            return JSONResponse({"status": "deleted"})

        @self.app.get("/api/preview/{filename}")
        async def get_preview_frame(filename: str):
            """Serve a preview frame image for real-time visualization."""
            processor = get_processor()
            preview_path = _safe_path(processor.preview_dir, filename)
            if not preview_path.exists():
                raise HTTPException(status_code=404, detail="Preview not found")

            # Read file into memory to avoid Content-Length mismatch when
            # the file is being concurrently rewritten by the worker.
            try:
                with open(preview_path, "rb") as f:
                    content = f.read()
            except OSError:
                raise HTTPException(status_code=404, detail="Preview not found")

            return Response(
                content=content,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        @self.app.get("/api/outputs")
        async def list_outputs():
            """List all output videos with metadata."""
            processor = get_processor()
            outputs_dir = processor.outputs_dir
            thumbnails_dir = outputs_dir.parent / "thumbnails"
            thumbnails_dir.mkdir(exist_ok=True)

            outputs = []
            for filepath in outputs_dir.glob("*.mp4"):
                try:
                    stat = filepath.stat()
                    size_bytes = stat.st_size

                    # Skip tiny/corrupted files
                    if size_bytes < 1000:
                        continue

                    metadata = _probe_video_metadata(filepath)
                    if metadata is None:
                        continue

                    # Format size
                    if size_bytes >= 1024 * 1024:
                        size_human = f"{size_bytes / (1024 * 1024):.1f} MB"
                    else:
                        size_human = f"{size_bytes / 1024:.1f} KB"

                    outputs.append({
                        "filename": filepath.name,
                        "size_bytes": size_bytes,
                        "size_human": size_human,
                        "duration": round(metadata["duration"], 2),
                        "width": metadata["width"],
                        "height": metadata["height"],
                        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "thumbnail": f"/api/output-thumbnail/{filepath.name}"
                    })
                except Exception as e:
                    logger.warning(f"Error reading output file {filepath}: {e}")
                    continue

            # Sort by creation date (newest first)
            outputs.sort(key=lambda x: x["created_at"], reverse=True)
            return JSONResponse({"outputs": outputs})

        @self.app.get("/api/output-thumbnail/{filename}")
        async def get_output_thumbnail(filename: str):
            """Get or generate a thumbnail for an output video."""
            processor = get_processor()
            video_path = _safe_path(processor.outputs_dir, filename)
            if not video_path.exists():
                raise HTTPException(status_code=404, detail="Video not found")

            thumbnails_dir = processor.outputs_dir.parent / "thumbnails"
            thumbnails_dir.mkdir(exist_ok=True)
            thumbnail_name = filename.rsplit(".", 1)[0] + ".jpg"
            thumbnail_path = thumbnails_dir / thumbnail_name

            # Regenerate if missing or the video is newer than the thumbnail.
            if not thumbnail_path.exists() or video_path.stat().st_mtime > thumbnail_path.stat().st_mtime:
                if not _generate_video_thumbnail(video_path, thumbnail_path):
                    raise HTTPException(status_code=404, detail="Thumbnail not found")

            if not thumbnail_path.exists() or thumbnail_path.stat().st_size == 0:
                raise HTTPException(status_code=404, detail="Thumbnail not found")

            return FileResponse(
                path=str(thumbnail_path),
                media_type="image/jpeg",
                filename=thumbnail_name,
            )

        @self.app.delete("/api/output/{filename}")
        async def delete_output(filename: str):
            """Delete an output video file."""
            processor = get_processor()
            video_path = _safe_path(processor.outputs_dir, filename)
            if not video_path.exists():
                raise HTTPException(status_code=404, detail="Output not found")

            os.remove(video_path)

            thumbnails_dir = processor.outputs_dir.parent / "thumbnails"
            thumbnail_name = filename.rsplit(".", 1)[0] + ".jpg"
            thumbnail_path = thumbnails_dir / thumbnail_name
            _remove_if_exists(thumbnail_path)

            return JSONResponse({"status": "deleted", "filename": filename})

        # ============ Multi-Style API Endpoints ============

        @self.app.get("/api/multistyle/styles")
        async def get_multistyle_styles():
            """Get available multi-style painting styles."""
            return JSONResponse({
                "styles": [
                    {"slug": slug, "description": desc}
                    for slug, desc in MULTISTYLE_STYLES
                ]
            })

        @self.app.post("/api/multistyle/process")
        async def process_multistyle(
            background_tasks: BackgroundTasks,
            video_name: str = Form(None),
            uploaded_file: str = Form(None),
            custom_description: str = Form(None),
        ):
            """Start multi-style video processing with LLaVA + FLUX."""
            processor = get_processor()
            multistyle = get_multistyle_processor()
            pipeline = get_pipeline()

            input_path = _resolve_input_video_path(
                processor,
                uploaded_file=uploaded_file,
                video_name=video_name,
            )

            # Create job
            try:
                job = multistyle.create_job(input_path)
                if custom_description:
                    job.description = custom_description
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            # Start processing in background
            background_tasks.add_task(multistyle.process_job, job.job_id, pipeline)

            return JSONResponse(job.to_dict())

        @self.app.get("/api/multistyle/job/{job_id}")
        async def get_multistyle_job_status(job_id: str):
            """Get the status of a multi-style processing job."""
            multistyle = get_multistyle_processor()
            job = multistyle.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            return JSONResponse(job.to_dict())

        @self.app.get("/api/multistyle/jobs")
        async def list_multistyle_jobs():
            """List all multi-style processing jobs."""
            multistyle = get_multistyle_processor()
            return JSONResponse({"jobs": multistyle.get_all_jobs()})

        @self.app.get("/api/multistyle/output/{job_id}/{filename}")
        async def get_multistyle_output(job_id: str, filename: str):
            """Serve a multi-style output video."""
            multistyle = get_multistyle_processor()
            job_dir = _safe_path(multistyle.multistyle_dir, job_id)
            output_path = _safe_path(job_dir, filename)
            if not output_path.exists():
                raise HTTPException(status_code=404, detail="Output not found")

            return FileResponse(
                path=str(output_path),
                media_type="video/mp4",
                filename=filename,
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
