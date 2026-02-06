"""Multi-style FLUX video generator with LLaVA preprocessing."""

import asyncio
import base64
import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Callable
import cv2
import numpy as np
from PIL import Image

from app.config import config


class MultiStyleStatus(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"  # LLaVA phase
    GENERATING = "generating"  # FLUX phase
    CREATING_GRID = "creating_grid"
    COMPLETED = "completed"
    FAILED = "failed"


# The 5 painting styles
STYLES = [
    ("oil-painting", "oil painting with rich textures and classic brushstrokes"),
    ("watercolor", "soft flowing watercolor painting"),
    ("impressionist", "impressionist style with visible brushstrokes like Monet"),
    ("pop-art", "bold pop art style with vibrant colors like Warhol"),
    ("ukiyo-e", "ukiyo-e Japanese woodblock print style"),
]


@dataclass
class MultiStyleJob:
    """Represents a multi-style video processing job."""
    job_id: str
    input_path: str
    output_dir: str
    status: MultiStyleStatus = MultiStyleStatus.PENDING
    description: str = ""  # LLaVA description
    current_style_index: int = 0
    current_style_name: str = ""
    style_progress: float = 0.0  # Progress within current style (0-100)
    overall_progress: float = 0.0  # Overall progress (0-100)
    completed_outputs: List[str] = field(default_factory=list)
    grid_output: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    fps: float = 30.0
    total_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "input_path": os.path.basename(self.input_path),
            "status": self.status.value,
            "description": self.description,
            "current_style_index": self.current_style_index,
            "current_style_name": self.current_style_name,
            "style_progress": round(self.style_progress, 2),
            "overall_progress": round(self.overall_progress, 2),
            "completed_outputs": self.completed_outputs,
            "grid_output": self.grid_output,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_styles": len(STYLES),
        }


class MultiStyleProcessor:
    """Processes videos through LLaVA + FLUX multi-style pipeline."""

    def __init__(self):
        self.jobs: Dict[str, MultiStyleJob] = {}
        self.outputs_dir = Path(config.outputs_dir) if hasattr(config, 'outputs_dir') else Path("outputs")
        self.multistyle_dir = self.outputs_dir / "multistyle"
        self.multistyle_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, input_path: str) -> MultiStyleJob:
        """Create a new multi-style processing job."""
        job_id = str(uuid.uuid4())[:8]

        # Create output directory for this job
        job_output_dir = self.multistyle_dir / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        # Get video info
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        job = MultiStyleJob(
            job_id=job_id,
            input_path=input_path,
            output_dir=str(job_output_dir),
            fps=fps,
            total_frames=total_frames,
        )

        self.jobs[job_id] = job
        return job

    def _sample_frames(self, video_path: str, num_frames: int = 5) -> List[Image.Image]:
        """Sample evenly-spaced frames from a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            num_frames = max(1, total_frames)

        # Calculate frame indices to sample
        indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)] if num_frames > 1 else [total_frames // 2]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def analyze_video_with_llava(self, video_path: str) -> str:
        """Analyze video frames with LLaVA via Ollama."""
        frames = self._sample_frames(video_path, num_frames=5)

        descriptions = []
        for i, frame in enumerate(frames):
            # Resize frame for LLaVA (smaller = faster)
            frame_resized = frame.copy()
            frame_resized.thumbnail((512, 512))

            # Convert to base64
            img_base64 = self._image_to_base64(frame_resized)

            # Call Ollama LLaVA
            prompt = "Describe what you see in this image in one sentence. Focus on the main subject, action, and setting."

            try:
                result = subprocess.run(
                    ["ollama", "run", "llava", prompt],
                    input=f"[img]{img_base64}[/img]",
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0 and result.stdout.strip():
                    descriptions.append(result.stdout.strip())
            except subprocess.TimeoutExpired:
                print(f"LLaVA timeout on frame {i}")
            except Exception as e:
                print(f"LLaVA error on frame {i}: {e}")

        if not descriptions:
            return "A video scene"

        # Combine descriptions into a coherent summary
        if len(descriptions) == 1:
            return descriptions[0]

        # Use Ollama to combine multiple frame descriptions
        combined_prompt = f"""Combine these frame descriptions into one coherent sentence describing the video:
{chr(10).join(f'- {d}' for d in descriptions)}

Provide only the combined description, no other text."""

        try:
            result = subprocess.run(
                ["ollama", "run", "llama3.2"],
                input=combined_prompt,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            print(f"Failed to combine descriptions: {e}")

        # Fallback: just use the middle description
        return descriptions[len(descriptions) // 2]

    def _build_prompt(self, description: str, style_desc: str) -> str:
        """Build the FLUX prompt from description and style."""
        return f"{description}, detailed, vibrant colors, {style_desc}"

    async def generate_style_video(
        self,
        job: MultiStyleJob,
        pipeline,
        style_slug: str,
        style_desc: str,
        style_index: int,
    ) -> Optional[str]:
        """Generate a single style variant of the video."""
        job.current_style_index = style_index
        job.current_style_name = style_slug
        job.style_progress = 0.0

        prompt = self._build_prompt(job.description, style_desc)
        print(f"[{style_index + 1}/{len(STYLES)}] Generating {style_slug}: {prompt[:60]}...")

        # Ensure FLUX model is loaded
        if pipeline.current_pipeline_type != "flux":
            pipeline.load_model("flux-klein")

        pipeline.update_prompt(prompt)

        # Output path for this style
        input_name = Path(job.input_path).stem
        output_filename = f"{input_name}_{style_slug}.mp4"
        output_path = str(Path(job.output_dir) / output_filename)

        # Open input video
        cap = cv2.VideoCapture(job.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {job.input_path}")

        # Get output resolution from pipeline
        out_w, out_h = pipeline.get_output_resolution()

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, job.fps, (out_w, out_h))

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
                output_array = np.array(output_image)
                output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
                out.write(output_bgr)

            frame_idx += 1
            job.style_progress = (frame_idx / job.total_frames) * 100

            # Calculate overall progress
            base_progress = (style_index / len(STYLES)) * 100
            style_contribution = (job.style_progress / 100) * (100 / len(STYLES))
            job.overall_progress = base_progress + style_contribution

            # Yield control
            if frame_idx % 5 == 0:
                await asyncio.sleep(0)

        cap.release()
        out.release()

        # Re-encode with H.264 for browser compatibility
        final_path = output_path.replace('.mp4', '_h264.mp4')
        result = subprocess.run([
            'ffmpeg', '-y', '-i', output_path,
            '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', '-pix_fmt', 'yuv420p',
            final_path
        ], capture_output=True, text=True)

        if result.returncode == 0:
            os.remove(output_path)
            os.rename(final_path, output_path)

        return output_path

    def create_comparison_grid(self, job: MultiStyleJob) -> Optional[str]:
        """Create a 2x3 comparison grid video using FFmpeg."""
        if len(job.completed_outputs) < len(STYLES):
            return None

        job.status = MultiStyleStatus.CREATING_GRID

        input_name = Path(job.input_path).stem
        grid_output = str(Path(job.output_dir) / f"{input_name}_grid.mp4")

        # Build FFmpeg filter for 2x3 grid
        # Layout: [0][1][2] / [3][4][original]
        inputs = []
        for i, path in enumerate(job.completed_outputs):
            inputs.extend(['-i', path])
        inputs.extend(['-i', job.input_path])  # Original as 6th

        # Scale all inputs to same size and create grid
        filter_complex = []
        for i in range(6):
            filter_complex.append(f"[{i}:v]scale=512:512,setsar=1[v{i}]")

        # Stack into 2x3 grid
        filter_complex.append(
            "[v0][v1][v2]hstack=inputs=3[top];"
            "[v3][v4][v5]hstack=inputs=3[bottom];"
            "[top][bottom]vstack=inputs=2[out]"
        )

        filter_str = ";".join(filter_complex)

        cmd = [
            'ffmpeg', '-y',
            *inputs,
            '-filter_complex', filter_str,
            '-map', '[out]',
            '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', '-pix_fmt', 'yuv420p',
            grid_output
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return grid_output
        else:
            print(f"FFmpeg grid error: {result.stderr}")
            return None

    async def process_job(self, job_id: str, pipeline) -> bool:
        """Process a complete multi-style job."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        job.started_at = time.time()

        try:
            # Phase 1: Analyze with LLaVA
            job.status = MultiStyleStatus.ANALYZING
            print(f"Analyzing video with LLaVA...")
            job.description = self.analyze_video_with_llava(job.input_path)
            print(f"Description: {job.description}")

            # Phase 2: Generate each style
            job.status = MultiStyleStatus.GENERATING
            for i, (style_slug, style_desc) in enumerate(STYLES):
                output_path = await self.generate_style_video(
                    job, pipeline, style_slug, style_desc, i
                )
                if output_path:
                    job.completed_outputs.append(output_path)
                    print(f"  Completed: {os.path.basename(output_path)}")

            # Phase 3: Create comparison grid
            grid_path = self.create_comparison_grid(job)
            if grid_path:
                job.grid_output = grid_path
                print(f"Grid created: {os.path.basename(grid_path)}")

            job.status = MultiStyleStatus.COMPLETED
            job.completed_at = time.time()
            job.overall_progress = 100.0
            return True

        except Exception as e:
            job.status = MultiStyleStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            print(f"Multi-style processing error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_job(self, job_id: str) -> Optional[MultiStyleJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> list:
        """Get all jobs."""
        return [job.to_dict() for job in self.jobs.values()]


# Global processor instance
_multistyle_processor: Optional[MultiStyleProcessor] = None


def get_multistyle_processor() -> MultiStyleProcessor:
    """Get or create the global multi-style processor instance."""
    global _multistyle_processor
    if _multistyle_processor is None:
        _multistyle_processor = MultiStyleProcessor()
    return _multistyle_processor
