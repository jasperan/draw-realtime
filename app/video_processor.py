"""Video-to-video batch processor using StreamDiffusion."""

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Callable
import cv2
import numpy as np
from PIL import Image

from app.config import config


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingJob:
    """Represents a video processing job."""
    job_id: str
    input_path: str
    output_path: str
    prompt: str
    model: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    fps: float = 30.0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "input_path": os.path.basename(self.input_path),
            "output_filename": os.path.basename(self.output_path),
            "prompt": self.prompt,
            "model": self.model,
            "status": self.status.value,
            "progress": round(self.progress, 2),
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class VideoProcessor:
    """Processes videos through StreamDiffusion pipeline."""

    def __init__(self):
        self.jobs: Dict[str, ProcessingJob] = {}
        self.outputs_dir = Path(config.outputs_dir) if hasattr(config, 'outputs_dir') else Path("outputs")
        self.uploads_dir = Path("uploads")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def create_job(
        self,
        input_path: str,
        prompt: str,
        model: str,
    ) -> ProcessingJob:
        """Create a new processing job."""
        job_id = str(uuid.uuid4())[:8]

        # Create output filename
        input_name = Path(input_path).stem
        output_filename = f"{input_name}_{job_id}_output.mp4"
        output_path = str(self.outputs_dir / output_filename)

        # Get video info
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        job = ProcessingJob(
            job_id=job_id,
            input_path=input_path,
            output_path=output_path,
            prompt=prompt,
            model=model,
            total_frames=total_frames,
            fps=fps,
        )

        self.jobs[job_id] = job
        return job

    async def process_job(self, job_id: str, pipeline) -> bool:
        """Process a video job asynchronously."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        job.status = JobStatus.PROCESSING
        job.started_at = time.time()

        try:
            # Check/update model
            if job.model != pipeline.get_current_model():
                pipeline.load_model(job.model)

            # Update prompt
            pipeline.update_prompt(job.prompt)

            # Open input video
            cap = cv2.VideoCapture(job.input_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {job.input_path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                job.output_path,
                fourcc,
                job.fps,
                (config.width, config.height)  # Output at model resolution
            )

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Process through pipeline
                output_image = pipeline.predict(pil_image)

                if output_image is not None:
                    # Convert back to BGR for OpenCV
                    output_array = np.array(output_image)
                    output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
                    out.write(output_bgr)

                frame_idx += 1
                job.current_frame = frame_idx
                job.progress = (frame_idx / job.total_frames) * 100

                # Yield control to allow other async operations
                if frame_idx % 10 == 0:
                    await asyncio.sleep(0)

            cap.release()
            out.release()

            # Re-encode with ffmpeg for browser compatibility (H.264)
            temp_path = job.output_path
            final_path = job.output_path.replace('.mp4', '_h264.mp4')

            # Use ffmpeg to re-encode
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y', '-i', temp_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-pix_fmt', 'yuv420p',
                final_path
            ], capture_output=True, text=True)

            if result.returncode == 0:
                # Replace original with H.264 version
                os.remove(temp_path)
                os.rename(final_path, temp_path)

            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.progress = 100.0
            return True

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> list:
        """Get all jobs."""
        return [job.to_dict() for job in self.jobs.values()]

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove old completed/failed jobs."""
        now = time.time()
        max_age_seconds = max_age_hours * 3600

        to_remove = []
        for job_id, job in self.jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                if job.completed_at and (now - job.completed_at) > max_age_seconds:
                    to_remove.append(job_id)

        for job_id in to_remove:
            del self.jobs[job_id]


# Global processor instance
_processor: Optional[VideoProcessor] = None


def get_processor() -> VideoProcessor:
    """Get or create the global processor instance."""
    global _processor
    if _processor is None:
        _processor = VideoProcessor()
    return _processor
