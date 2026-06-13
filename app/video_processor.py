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
from app.pipeline import pipeline_lock


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def _remove_paths(*paths: str) -> None:
    """Best-effort cleanup for generated files and temp outputs."""
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            pass


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
        self.outputs_dir = Path(config.outputs_dir)
        self.uploads_dir = Path("uploads")
        self.preview_dir = Path("previews")  # For real-time frame previews
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

    def _reencode_to_h264(self, source_path: str) -> bool:
        """Re-encode an mp4 in place to H.264 for browser compatibility.

        Writes to a sibling file, then atomically swaps it in. Returns
        False and leaves the original intact if ffmpeg fails.
        """
        final_path = source_path.replace('.mp4', '_h264.mp4')
        result = subprocess.run([
            'ffmpeg', '-y', '-i', source_path,
            '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', '-pix_fmt', 'yuv420p',
            final_path,
        ], capture_output=True, text=True)

        if result.returncode != 0:
            _remove_paths(final_path)
            return False

        try:
            os.replace(final_path, source_path)  # atomic on POSIX + Windows
        except OSError:
            _remove_paths(final_path)
            return False
        return True

    def delete_job(self, job_id: str) -> bool:
        """Remove a job from the registry. Returns True if removed."""
        return self.jobs.pop(job_id, None) is not None

    def create_job(
        self,
        input_path: str,
        prompt: str,
        model: str,
    ) -> ProcessingJob:
        """Create a new processing job."""
        self.cleanup_old_jobs()  # Opportunistic trim to bound memory growth.
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
        self.cleanup_old_jobs()
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
        """Async entry point for a MonarchRT text-to-video job.

        Runs the blocking body (CUDA inference, cv2, ffmpeg) inside
        asyncio.to_thread so the FastAPI event loop stays responsive for
        status polling and new uploads while a job is in flight.
        """
        job = self.jobs.get(job_id)
        if not job:
            return False
        return await asyncio.to_thread(self._process_generate_job_sync, job, pipeline)

    def _process_generate_job_sync(self, job: "ProcessingJob", pipeline) -> bool:
        """Blocking body of process_generate_job. Runs in a worker thread.

        Holds pipeline_lock for the entire run so concurrent jobs can't
        swap the loaded model out from under us.
        """
        job.status = JobStatus.PROCESSING
        job.started_at = time.time()

        with pipeline_lock():
            try:
                if job.model != pipeline.get_current_model():
                    if not pipeline.load_model(job.model):
                        raise ValueError(f"Failed to load model: {job.model}")
                pipeline.update_prompt(job.prompt)

                from app.config import MODELS
                model_config = MODELS.get(job.model)
                num_frames = model_config.num_output_frames if model_config else job.total_frames
                out_w = model_config.width if model_config and model_config.width else 832
                out_h = model_config.height if model_config and model_config.height else 480

                job.total_frames = num_frames
                job.progress = 5.0

                frames = pipeline.generate_video(
                    prompt=job.prompt,
                    num_frames=num_frames,
                    width=out_w,
                    height=out_h,
                )

                if not frames:
                    raise ValueError("MonarchRT generation returned no frames")

                job.progress = 70.0

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(job.output_path, fourcc, job.fps, (out_w, out_h))

                for i, frame in enumerate(frames):
                    output_array = np.array(frame)
                    output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)

                    h, w = output_bgr.shape[:2]
                    if w != out_w or h != out_h:
                        output_bgr = cv2.resize(output_bgr, (out_w, out_h))

                    out.write(output_bgr)

                    job.current_frame = i + 1
                    job.progress = 70.0 + (i / max(1, len(frames))) * 20.0

                    # Save a preview frame at start, end, and a few mid-points.
                    if i == 0 or i == len(frames) - 1 or i % max(1, len(frames) // 5) == 0:
                        output_preview_path = str(self.preview_dir / f"{job.job_id}_output.jpg")
                        output_temp = str(self.preview_dir / f"{job.job_id}_output_tmp.jpg")
                        cv2.imwrite(output_temp, output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        os.replace(output_temp, output_preview_path)
                        job.preview_frame_path = f"{job.job_id}_output.jpg"

                out.release()

                job.progress = 90.0
                self._reencode_to_h264(job.output_path)

                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                job.progress = 100.0
                return True

            except Exception as e:
                return self._fail_job(
                    job,
                    e,
                    "MonarchRT generation error",
                    cleanup_paths=(job.output_path,),
                )

    async def process_job(self, job_id: str, pipeline) -> bool:
        """Async entry point for a video-to-video job.

        The blocking per-frame work (cv2 reads, model inference, preview
        writes, ffmpeg re-encode) runs in asyncio.to_thread so the event
        loop stays free to serve /api/job/* polling while a job is running.
        """
        job = self.jobs.get(job_id)
        if not job:
            return False
        return await asyncio.to_thread(self._process_job_sync, job, pipeline)

    def _process_job_sync(self, job: "ProcessingJob", pipeline) -> bool:
        """Blocking body of process_job. Runs in a worker thread.

        Holds pipeline_lock for the whole run so a concurrent job can't
        swap the loaded model between frames.
        """
        job.status = JobStatus.PROCESSING
        job.started_at = time.time()

        input_preview_path = str(self.preview_dir / f"{job.job_id}_input.jpg")
        output_preview_path = str(self.preview_dir / f"{job.job_id}_output.jpg")

        with pipeline_lock():
            try:
                if job.model != pipeline.get_current_model():
                    if not pipeline.load_model(job.model):
                        raise ValueError(f"Failed to load model: {job.model}")
                pipeline.update_prompt(job.prompt)

                cap = cv2.VideoCapture(job.input_path)
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video: {job.input_path}")

                out_w, out_h = pipeline.get_output_resolution()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(job.output_path, fourcc, job.fps, (out_w, out_h))

                frame_idx = 0
                last_speed_update = time.time()
                frames_since_update = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    output_image = pipeline.predict(pil_image)

                    # Resize the input frame to preview resolution so the
                    # split-view previews line up side by side.
                    input_resized = cv2.resize(frame, (out_w, out_h))

                    if output_image is not None:
                        output_array = np.array(output_image)
                        output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
                        out.write(output_bgr)
                    else:
                        # Prediction failed: reuse the input as a placeholder
                        # so the preview still shows something.
                        output_bgr = input_resized

                    # Every 100 frames, refresh the preview images atomically
                    # so the HTTP handler never serves half-written JPEGs.
                    if frame_idx % 100 == 0:
                        input_temp = str(self.preview_dir / f"{job.job_id}_input_tmp.jpg")
                        output_temp = str(self.preview_dir / f"{job.job_id}_output_tmp.jpg")
                        cv2.imwrite(input_temp, input_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        cv2.imwrite(output_temp, output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        os.replace(input_temp, input_preview_path)
                        os.replace(output_temp, output_preview_path)
                        job.input_frame_path = f"{job.job_id}_input.jpg"
                        job.preview_frame_path = f"{job.job_id}_output.jpg"

                    frame_idx += 1
                    frames_since_update += 1
                    job.current_frame = frame_idx
                    # Guard against frame-count probing failures (VFR codecs,
                    # stream errors) that would otherwise divide by zero.
                    job.progress = (frame_idx / max(1, job.total_frames)) * 100

                    now = time.time()
                    if now - last_speed_update >= 1.0:
                        job.processing_fps = frames_since_update / (now - last_speed_update)
                        frames_remaining = job.total_frames - frame_idx
                        if job.processing_fps > 0:
                            job.eta_seconds = frames_remaining / job.processing_fps
                        last_speed_update = now
                        frames_since_update = 0

                cap.release()
                out.release()

                self._reencode_to_h264(job.output_path)

                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                job.progress = 100.0
                return True

            except Exception as e:
                return self._fail_job(
                    job,
                    e,
                    "Processing error",
                    cleanup_paths=(
                        job.output_path,
                        job.output_path.replace('.mp4', '_h264.mp4'),
                    ),
                )

    def _fail_job(
        self,
        job: "ProcessingJob",
        error: Exception,
        message: str,
        cleanup_paths: tuple[str, ...] = (),
    ) -> bool:
        """Mark a job failed, preserve the error, and remove partial files."""
        job.status = JobStatus.FAILED
        job.error = str(error)
        job.completed_at = time.time()
        print(f"{message} for job {job.job_id}: {error}")
        import traceback
        traceback.print_exc()
        _remove_paths(*cleanup_paths)
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
