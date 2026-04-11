"""Video-to-video batch processor using StreamDiffusion."""

import asyncio
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict
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
    processing_fps: float = 0.0  # Actual processing speed
    eta_seconds: float = 0.0  # Estimated time remaining
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    # For real-time preview
    preview_frame_path: Optional[str] = None  # Path to latest output frame image
    input_frame_path: Optional[str] = None  # Path to corresponding input frame

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
            "processing_fps": round(self.processing_fps, 1),
            "eta_seconds": round(self.eta_seconds, 0),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "preview_frame": self.preview_frame_path,
            "input_frame": self.input_frame_path,
        }


class VideoProcessor:
    """Processes videos through StreamDiffusion pipeline."""

    def __init__(self):
        self.jobs: Dict[str, ProcessingJob] = {}
        self.outputs_dir = Path(config.outputs_dir) if hasattr(config, 'outputs_dir') else Path("outputs")
        self.uploads_dir = Path("uploads")
        self.preview_dir = Path("previews")  # For real-time frame previews
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

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

    def create_generate_job(
        self,
        prompt: str,
        model: str,
        num_frames: int = 21,
        fps: float = 16.0,
    ) -> ProcessingJob:
        """Create a text-to-video generation job (MonarchRT).

        Unlike create_job(), this does not require an input video.
        """
        job_id = str(uuid.uuid4())[:8]

        output_filename = f"generated_{job_id}_output.mp4"
        output_path = str(self.outputs_dir / output_filename)

        job = ProcessingJob(
            job_id=job_id,
            input_path="",  # No input for text-to-video
            output_path=output_path,
            prompt=prompt,
            model=model,
            total_frames=num_frames,
            fps=fps,
        )

        self.jobs[job_id] = job
        return job

    async def process_generate_job(self, job_id: str, pipeline) -> bool:
        """Process a MonarchRT text-to-video generation job."""
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

            from app.config import MODELS
            model_config = MODELS.get(job.model)
            num_frames = model_config.num_output_frames if model_config else job.total_frames
            out_w = model_config.width if model_config and model_config.width else 832
            out_h = model_config.height if model_config and model_config.height else 480

            job.total_frames = num_frames
            job.progress = 5.0  # Show initial progress

            # Yield control before heavy computation
            await asyncio.sleep(0)

            # Generate video frames (this is a single batch operation)
            frames = pipeline.generate_video(
                prompt=job.prompt,
                num_frames=num_frames,
                width=out_w,
                height=out_h,
            )

            if not frames:
                raise ValueError("MonarchRT generation returned no frames")

            job.progress = 70.0
            await asyncio.sleep(0)

            # Write frames to video file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                job.output_path,
                fourcc,
                job.fps,
                (out_w, out_h),
            )

            for i, frame in enumerate(frames):
                output_array = np.array(frame)
                output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)

                # Resize if needed
                h, w = output_bgr.shape[:2]
                if w != out_w or h != out_h:
                    output_bgr = cv2.resize(output_bgr, (out_w, out_h))

                out.write(output_bgr)

                job.current_frame = i + 1
                job.progress = 70.0 + (i / len(frames)) * 20.0

                # Save preview frame periodically
                if i == 0 or i == len(frames) - 1 or i % max(1, len(frames) // 5) == 0:
                    output_preview_path = str(self.preview_dir / f"{job_id}_output.jpg")
                    output_temp = str(self.preview_dir / f"{job_id}_output_tmp.jpg")
                    cv2.imwrite(output_temp, output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    os.replace(output_temp, output_preview_path)
                    job.preview_frame_path = f"{job_id}_output.jpg"

            out.release()

            job.progress = 90.0
            await asyncio.sleep(0)

            # Re-encode with ffmpeg for browser compatibility (H.264)
            temp_path = job.output_path
            final_path = job.output_path.replace('.mp4', '_h264.mp4')

            result = subprocess.run([
                'ffmpeg', '-y', '-i', temp_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-pix_fmt', 'yuv420p',
                final_path
            ], capture_output=True, text=True)

            if result.returncode == 0:
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
            print(f"MonarchRT generation error for job {job_id}: {e}")
            import traceback
            traceback.print_exc()

            try:
                if os.path.exists(job.output_path):
                    os.remove(job.output_path)
            except OSError:
                pass

            return False

    async def process_job(self, job_id: str, pipeline) -> bool:
        """Process a video job asynchronously."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        job.status = JobStatus.PROCESSING
        job.started_at = time.time()

        # Preview frame paths for this job
        input_preview_path = str(self.preview_dir / f"{job_id}_input.jpg")
        output_preview_path = str(self.preview_dir / f"{job_id}_output.jpg")

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

            # Create output video writer at model-specific resolution
            out_w, out_h = pipeline.get_output_resolution()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                job.output_path,
                fourcc,
                job.fps,
                (out_w, out_h)
            )

            frame_idx = 0
            last_speed_update = time.time()
            frames_since_update = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Process through pipeline
                output_image = pipeline.predict(pil_image)

                # Resize input frame for preview (match output resolution)
                input_resized = cv2.resize(frame, (out_w, out_h))

                if output_image is not None:
                    # Convert back to BGR for OpenCV
                    output_array = np.array(output_image)
                    output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
                    out.write(output_bgr)
                else:
                    # If prediction failed, use input frame as output placeholder
                    output_bgr = input_resized

                # Save preview frames every 100 frames for real-time visualization
                # Use atomic writes (write to temp file, then rename) to avoid race conditions
                if frame_idx % 100 == 0:
                    input_temp = str(self.preview_dir / f"{job_id}_input_tmp.jpg")
                    output_temp = str(self.preview_dir / f"{job_id}_output_tmp.jpg")
                    cv2.imwrite(input_temp, input_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    cv2.imwrite(output_temp, output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    # Atomic rename to avoid serving partial files
                    os.replace(input_temp, input_preview_path)
                    os.replace(output_temp, output_preview_path)
                    job.input_frame_path = f"{job_id}_input.jpg"
                    job.preview_frame_path = f"{job_id}_output.jpg"

                frame_idx += 1
                frames_since_update += 1
                job.current_frame = frame_idx
                job.progress = (frame_idx / job.total_frames) * 100

                # Update processing speed every second
                now = time.time()
                if now - last_speed_update >= 1.0:
                    job.processing_fps = frames_since_update / (now - last_speed_update)
                    frames_remaining = job.total_frames - frame_idx
                    if job.processing_fps > 0:
                        job.eta_seconds = frames_remaining / job.processing_fps
                    last_speed_update = now
                    frames_since_update = 0

                # Yield control to allow other async operations
                if frame_idx % 10 == 0:
                    await asyncio.sleep(0)

            cap.release()
            out.release()

            # Re-encode with ffmpeg for browser compatibility (H.264)
            temp_path = job.output_path
            final_path = job.output_path.replace('.mp4', '_h264.mp4')

            # Use ffmpeg to re-encode
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
            print(f"Processing error for job {job_id}: {e}")
            import traceback
            traceback.print_exc()

            # Clean up partial output file
            try:
                if os.path.exists(job.output_path):
                    os.remove(job.output_path)
            except OSError:
                pass

            # Clean up any temp H.264 re-encode file
            try:
                h264_path = job.output_path.replace('.mp4', '_h264.mp4')
                if os.path.exists(h264_path):
                    os.remove(h264_path)
            except OSError:
                pass

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
