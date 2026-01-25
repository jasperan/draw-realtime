"""StreamDiffusion Real-Time Demo - FastAPI Application."""

import asyncio
import io
import logging
import mimetypes
import os
import time
import uuid
from types import SimpleNamespace
from typing import Optional

import cv2
import markdown2
import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from app.config import config, MODELS, DEFAULT_MODEL, DEFAULT_PROMPT
from app.connection_manager import ConnectionManager, ServerFullException
from app.pipeline import get_pipeline
from app.video_source import get_available_videos, get_video_path, VideoSource

# Fix mime type on Windows
mimetypes.add_type("application/javascript", ".js")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Throttle for frame processing
THROTTLE = 1.0 / 120


# Request/Response models
class InputParams(BaseModel):
    """Parameters for image generation."""
    prompt: str = Field(default=DEFAULT_PROMPT, description="Generation prompt")
    model: str = Field(default=DEFAULT_MODEL, description="Model to use")
    source_type: str = Field(default="webcam", description="Input source type")
    video_name: Optional[str] = Field(default=None, description="Server video filename")


class SettingsResponse(BaseModel):
    """API settings response."""
    models: dict
    default_model: str
    default_prompt: str
    width: int
    height: int
    max_queue_size: int


def pil_to_frame(image: Image.Image) -> bytes:
    """Convert PIL image to MJPEG frame bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    frame_bytes = buffer.getvalue()
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
        + frame_bytes
        + b"\r\n"
    )


def bytes_to_pil(data: bytes) -> Image.Image:
    """Convert bytes to PIL image."""
    return Image.open(io.BytesIO(data)).convert("RGB")


class App:
    """Main application class."""

    def __init__(self):
        self.app = FastAPI(title="StreamDiffusion Real-Time Demo")
        self.conn_manager = ConnectionManager()
        self.video_sources: dict = {}  # user_id -> VideoSource

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
            return SettingsResponse(
                models=pipeline.get_available_models(),
                default_model=DEFAULT_MODEL,
                default_prompt=DEFAULT_PROMPT,
                width=config.width,
                height=config.height,
                max_queue_size=config.max_queue_size,
            )

        @self.app.get("/api/videos")
        async def list_videos():
            """List available server-side videos."""
            return JSONResponse({"videos": get_available_videos()})

        @self.app.get("/api/queue")
        async def get_queue_size():
            """Get current connection count."""
            return JSONResponse({"queue_size": self.conn_manager.get_user_count()})

        @self.app.post("/api/model/{model_key}")
        async def switch_model(model_key: str):
            """Switch to a different model."""
            pipeline = get_pipeline()
            if pipeline.load_model(model_key):
                return JSONResponse({"status": "ok", "model": model_key})
            else:
                raise HTTPException(status_code=400, detail=f"Failed to load model: {model_key}")

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            """WebSocket endpoint for frame streaming."""
            try:
                await self.conn_manager.connect(user_id, websocket, config.max_queue_size)
                await self._handle_websocket(user_id)
            except ServerFullException as e:
                logger.error(f"Server full: {e}")
            finally:
                await self.conn_manager.disconnect(user_id)
                # Clean up video source if any
                if user_id in self.video_sources:
                    self.video_sources[user_id].close()
                    del self.video_sources[user_id]
                logger.info(f"User disconnected: {user_id}")

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            """MJPEG stream endpoint for processed frames."""
            if not self.conn_manager.check_user(user_id):
                raise HTTPException(status_code=404, detail="User not found")

            async def generate():
                pipeline = get_pipeline()
                last_params = SimpleNamespace()

                while self.conn_manager.check_user(user_id):
                    start_time = time.time()

                    params = await self.conn_manager.get_latest_data(user_id)

                    if not vars(params):
                        await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                        await asyncio.sleep(THROTTLE)
                        continue

                    # Check for model change
                    if hasattr(params, 'model') and params.model != pipeline.get_current_model():
                        pipeline.load_model(params.model)

                    # Check for prompt change
                    if hasattr(params, 'prompt') and params.prompt:
                        pipeline.update_prompt(params.prompt)

                    # Get image from params
                    if hasattr(params, 'image') and params.image is not None:
                        output = pipeline.predict(params.image)
                        if output:
                            yield pil_to_frame(output)
                            # Chrome bug workaround
                            user_agent = request.headers.get("user-agent", "")
                            if "Firefox" not in user_agent:
                                yield pil_to_frame(output)

                    await self.conn_manager.send_json(user_id, {"status": "send_frame"})

                    if config.debug:
                        logger.info(f"Frame time: {time.time() - start_time:.3f}s")

            return StreamingResponse(
                generate(),
                media_type="multipart/x-mixed-replace;boundary=frame",
                headers={"Cache-Control": "no-cache"},
            )

    async def _handle_websocket(self, user_id: uuid.UUID):
        """Handle WebSocket messages from a user."""
        pipeline = get_pipeline()
        last_time = time.time()

        while True:
            if config.timeout > 0 and time.time() - last_time > config.timeout:
                await self.conn_manager.send_json(
                    user_id,
                    {"status": "timeout", "message": "Session ended"},
                )
                break

            try:
                data = await self.conn_manager.receive_json(user_id)
            except Exception:
                break

            if data.get("status") != "next_frame":
                await asyncio.sleep(THROTTLE)
                continue

            # Receive parameters
            params_data = await self.conn_manager.receive_json(user_id)
            params = SimpleNamespace(**params_data)

            # Handle different source types
            source_type = getattr(params, 'source_type', 'webcam')

            if source_type == "webcam" or source_type == "upload":
                # Receive image from client
                image_data = await self.conn_manager.receive_bytes(user_id)
                if len(image_data) == 0:
                    await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                    await asyncio.sleep(THROTTLE)
                    continue
                params.image = bytes_to_pil(image_data)

            elif source_type == "server_video":
                # Read frame from server-side video
                video_name = getattr(params, 'video_name', None)
                if video_name:
                    # Initialize video source if needed
                    if user_id not in self.video_sources:
                        video_path = get_video_path(video_name)
                        if video_path:
                            self.video_sources[user_id] = VideoSource(video_path)

                    if user_id in self.video_sources:
                        frame = self.video_sources[user_id].read_frame()
                        if frame is None:
                            self.video_sources[user_id].reset()
                            frame = self.video_sources[user_id].read_frame()
                        if frame is not None:
                            params.image = Image.fromarray(frame)

            await self.conn_manager.update_data(user_id, params)
            await self.conn_manager.send_json(user_id, {"status": "wait"})
            last_time = time.time()

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
                    "message": "StreamDiffusion Real-Time Demo API",
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
