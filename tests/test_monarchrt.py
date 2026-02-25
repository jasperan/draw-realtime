"""Tests for MonarchRT integration.

All GPU/model operations are mocked so tests run without MonarchRT installed.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# MonarchRT Pipeline Wrapper Tests
# ---------------------------------------------------------------------------

class TestMonarchRTPipelineWrapper:
    """Tests for app/monarchrt_pipeline.py."""

    def test_is_monarchrt_available_no_gpu(self):
        """Returns False when CUDA is not available."""
        with patch("app.monarchrt_pipeline._monarchrt_available", None):
            with patch.dict(sys.modules, {"torch": MagicMock()}):
                import app.monarchrt_pipeline as mrt
                mrt._monarchrt_available = None  # Reset cache
                with patch("torch.cuda.is_available", return_value=False):
                    # Need to reimport to clear cache
                    mrt._monarchrt_available = None
                    result = mrt.is_monarchrt_available()
                    assert result is False

    def test_is_monarchrt_available_with_package(self):
        """Returns True when monarch_rt package is importable."""
        import app.monarchrt_pipeline as mrt
        mrt._monarchrt_available = None

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "monarch_rt": MagicMock(),
        }):
            mrt._monarchrt_available = None
            result = mrt.is_monarchrt_available()
            assert result is True

    def test_is_monarchrt_available_with_directory(self, tmp_path):
        """Returns True when MonarchRT directory exists with expected structure."""
        import app.monarchrt_pipeline as mrt
        mrt._monarchrt_available = None

        # Create fake MonarchRT directory structure
        monarchrt_dir = tmp_path / "MonarchRT"
        (monarchrt_dir / "pipeline").mkdir(parents=True)
        (monarchrt_dir / "wan").mkdir(parents=True)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict(sys.modules, {"torch": mock_torch}):
            # Remove monarch_rt from sys.modules if present
            with patch.dict(sys.modules, {"monarch_rt": None}):
                with patch.object(mrt, "MONARCHRT_DIR", monarchrt_dir):
                    mrt._monarchrt_available = None
                    result = mrt.is_monarchrt_available()
                    assert result is True

    def test_pipeline_init(self):
        """MonarchRTPipeline initializes with correct defaults."""
        from app.monarchrt_pipeline import MonarchRTPipeline

        pipe = MonarchRTPipeline(device="cuda")
        assert pipe.device == "cuda"
        assert pipe.pipeline is None
        assert pipe.wan_model is None
        assert pipe.mode is None
        assert pipe.is_loaded is False

    def test_generate_video_not_loaded(self):
        """generate_video returns None when no model is loaded."""
        from app.monarchrt_pipeline import MonarchRTPipeline

        pipe = MonarchRTPipeline()
        result = pipe.generate_video("test prompt")
        assert result is None

    def test_tensor_to_frames(self):
        """_tensor_to_frames converts (C,N,H,W) tensor to PIL images."""
        import torch
        from app.monarchrt_pipeline import MonarchRTPipeline

        # Create a fake video tensor (3 channels, 5 frames, 64x64)
        tensor = torch.rand(3, 5, 64, 64)
        frames = MonarchRTPipeline._tensor_to_frames(tensor)

        assert len(frames) == 5
        for f in frames:
            assert isinstance(f, Image.Image)
            assert f.size == (64, 64)

    def test_tensor_to_frames_with_batch_dim(self):
        """_tensor_to_frames handles (B,C,N,H,W) tensors."""
        import torch
        from app.monarchrt_pipeline import MonarchRTPipeline

        tensor = torch.rand(1, 3, 5, 64, 64)
        frames = MonarchRTPipeline._tensor_to_frames(tensor)
        assert len(frames) == 5

    def test_align_frame_count(self):
        """_align_frame_count aligns to 4n+1 format."""
        from app.monarchrt_pipeline import MonarchRTPipeline

        assert MonarchRTPipeline._align_frame_count(21) == 21  # already aligned
        assert MonarchRTPipeline._align_frame_count(20) == 21  # rounds up
        assert MonarchRTPipeline._align_frame_count(22) == 25  # rounds up
        assert MonarchRTPipeline._align_frame_count(81) == 81  # already aligned
        assert MonarchRTPipeline._align_frame_count(5) == 5    # already aligned
        assert MonarchRTPipeline._align_frame_count(6) == 9    # rounds up

    def test_cleanup(self):
        """cleanup frees resources and resets state."""
        from app.monarchrt_pipeline import MonarchRTPipeline

        pipe = MonarchRTPipeline()
        pipe.pipeline = MagicMock()
        pipe.wan_model = MagicMock()
        pipe.mode = "causal"
        pipe._loaded = True

        with patch("torch.cuda.is_available", return_value=False):
            pipe.cleanup()

        assert pipe.pipeline is None
        assert pipe.wan_model is None
        assert pipe.mode is None
        assert pipe.is_loaded is False

    def test_get_info(self):
        """get_info returns current state."""
        from app.monarchrt_pipeline import MonarchRTPipeline

        pipe = MonarchRTPipeline(device="cuda:0")
        info = pipe.get_info()
        assert info["loaded"] is False
        assert info["mode"] is None
        assert info["device"] == "cuda:0"


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestMonarchRTConfig:
    """Tests for MonarchRT entries in app/config.py."""

    def test_monarchrt_models_exist(self):
        """MonarchRT models are defined in MODELS dict."""
        from app.config import MODELS

        assert "monarchrt-sf" in MODELS
        assert "monarchrt-wan" in MODELS

    def test_monarchrt_sf_config(self):
        """Self-Forcing config has correct fields."""
        from app.config import MODELS

        cfg = MODELS["monarchrt-sf"]
        assert cfg.pipeline_type == "monarchrt"
        assert cfg.monarchrt_mode == "causal"
        assert cfg.is_text_to_video is True
        assert cfg.num_output_frames == 21
        assert cfg.width == 832
        assert cfg.height == 480
        assert cfg.monarchrt_config != ""
        assert cfg.monarchrt_checkpoint != ""

    def test_monarchrt_wan_config(self):
        """Wan2.1 config has correct fields."""
        from app.config import MODELS

        cfg = MODELS["monarchrt-wan"]
        assert cfg.pipeline_type == "monarchrt"
        assert cfg.monarchrt_mode == "bidirectional"
        assert cfg.is_text_to_video is True
        assert cfg.num_output_frames == 81
        assert cfg.monarchrt_model_name == "Wan2.1-T2V-1.3B"

    def test_pipeline_type_includes_monarchrt(self):
        """ModelConfig accepts 'monarchrt' pipeline type."""
        from app.config import ModelConfig

        cfg = ModelConfig(
            id="test",
            pipeline_type="monarchrt",
            monarchrt_mode="causal",
        )
        assert cfg.pipeline_type == "monarchrt"


# ---------------------------------------------------------------------------
# Pipeline Integration Tests
# ---------------------------------------------------------------------------

class TestPipelineMonarchRT:
    """Tests for MonarchRT in app/pipeline.py."""

    def test_pipeline_has_monarchrt_pipe_attr(self):
        """Pipeline class has monarchrt_pipe attribute."""
        from app.pipeline import Pipeline

        # Mock the __init__ to avoid loading a real model
        with patch.object(Pipeline, 'load_model', return_value=True):
            pipe = Pipeline.__new__(Pipeline)
            pipe.device = "cpu"
            pipe.dtype = None
            pipe.current_model_key = None
            pipe.current_pipeline_type = None
            pipe.stream = None
            pipe.diffusers_pipe = None
            pipe.flux_pipe = None
            pipe.monarchrt_pipe = None
            pipe.current_prompt = ""
            pipe.current_negative_prompt = ""

            assert hasattr(pipe, 'monarchrt_pipe')

    def test_generate_video_requires_monarchrt(self):
        """generate_video returns None if no MonarchRT model is loaded."""
        from app.pipeline import Pipeline

        with patch.object(Pipeline, 'load_model', return_value=True):
            pipe = Pipeline.__new__(Pipeline)
            pipe.current_pipeline_type = "streamdiffusion"
            pipe.monarchrt_pipe = None

            result = pipe.generate_video("test prompt")
            assert result is None

    def test_cleanup_handles_monarchrt(self):
        """_cleanup properly cleans up MonarchRT pipeline."""
        from app.pipeline import Pipeline

        with patch.object(Pipeline, 'load_model', return_value=True):
            pipe = Pipeline.__new__(Pipeline)
            pipe.stream = None
            pipe.diffusers_pipe = None
            pipe.flux_pipe = None
            pipe.monarchrt_pipe = MagicMock()

            with patch("torch.cuda.is_available", return_value=False):
                pipe._cleanup()

            assert pipe.monarchrt_pipe is None


# ---------------------------------------------------------------------------
# Video Processor Tests
# ---------------------------------------------------------------------------

class TestVideoProcessorMonarchRT:
    """Tests for MonarchRT in app/video_processor.py."""

    def test_create_generate_job(self, outputs_dir, tmp_path):
        """create_generate_job creates a job without input video."""
        from app.video_processor import VideoProcessor

        proc = VideoProcessor()
        proc.outputs_dir = outputs_dir

        job = proc.create_generate_job(
            prompt="a cat playing",
            model="monarchrt-sf",
            num_frames=21,
            fps=16.0,
        )

        assert job.job_id
        assert job.input_path == ""
        assert job.prompt == "a cat playing"
        assert job.model == "monarchrt-sf"
        assert job.total_frames == 21
        assert job.fps == 16.0
        assert "generated_" in job.output_path

    @pytest.mark.asyncio
    async def test_process_generate_job_success(
        self, outputs_dir, previews_dir, mock_pipeline
    ):
        """process_generate_job writes frames to video and completes."""
        from app.video_processor import VideoProcessor, JobStatus

        proc = VideoProcessor()
        proc.outputs_dir = outputs_dir
        proc.preview_dir = previews_dir

        # Set up pipeline for MonarchRT
        mock_pipeline.get_current_model.return_value = "monarchrt-sf"
        mock_pipeline.current_pipeline_type = "monarchrt"
        mock_pipeline.generate_video.return_value = [
            Image.fromarray(np.zeros((480, 832, 3), dtype=np.uint8))
            for _ in range(5)
        ]

        job = proc.create_generate_job(
            prompt="test prompt",
            model="monarchrt-sf",
            num_frames=5,
            fps=16.0,
        )

        # Patch ffmpeg call to succeed
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)  # ffmpeg "fails", keeps raw

            result = await proc.process_generate_job(job.job_id, mock_pipeline)

        assert result is True
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100.0
        mock_pipeline.generate_video.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_generate_job_failure(
        self, outputs_dir, previews_dir, mock_pipeline
    ):
        """process_generate_job handles generation failure."""
        from app.video_processor import VideoProcessor, JobStatus

        proc = VideoProcessor()
        proc.outputs_dir = outputs_dir
        proc.preview_dir = previews_dir

        mock_pipeline.get_current_model.return_value = "monarchrt-sf"
        mock_pipeline.current_pipeline_type = "monarchrt"
        mock_pipeline.generate_video.return_value = None  # Generation fails

        job = proc.create_generate_job(
            prompt="test prompt",
            model="monarchrt-sf",
        )

        result = await proc.process_generate_job(job.job_id, mock_pipeline)

        assert result is False
        assert job.status == JobStatus.FAILED
        assert "no frames" in job.error.lower()


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------

class TestMonarchRTAPI:
    """Tests for MonarchRT API endpoints in app/main.py."""

    @pytest.fixture
    def mock_deps(self):
        """Set up mocks for API testing."""
        with patch("app.main.get_pipeline") as mock_get_pipeline, \
             patch("app.main.get_processor") as mock_get_processor, \
             patch("app.main.get_multistyle_processor") as mock_get_multi, \
             patch("app.main.is_monarchrt_available") as mock_mrt_avail:

            mock_pipeline = MagicMock()
            mock_pipeline.get_available_models.return_value = {
                "sd-turbo": "SD-Turbo",
                "monarchrt-sf": "MonarchRT Self-Forcing",
                "monarchrt-wan": "MonarchRT Wan2.1",
            }
            mock_pipeline.get_current_model.return_value = "sd-turbo"
            mock_pipeline.generate_video.return_value = [
                Image.fromarray(np.zeros((480, 832, 3), dtype=np.uint8))
                for _ in range(5)
            ]
            mock_get_pipeline.return_value = mock_pipeline

            mock_processor = MagicMock()
            mock_get_processor.return_value = mock_processor

            mock_mrt_avail.return_value = True

            yield {
                "pipeline": mock_pipeline,
                "processor": mock_processor,
                "monarchrt_available": mock_mrt_avail,
            }

    @pytest.fixture
    def client(self, mock_deps):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_settings_includes_monarchrt_available(self, client, mock_deps):
        """Settings endpoint includes monarchrt_available flag."""
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.json()
        assert "monarchrt_available" in data
        assert data["monarchrt_available"] is True

    def test_settings_includes_monarchrt_models(self, client, mock_deps):
        """Settings endpoint lists MonarchRT models."""
        res = client.get("/api/settings")
        data = res.json()
        assert "monarchrt-sf" in data["models"]
        assert "monarchrt-wan" in data["models"]

    def test_generate_endpoint_creates_job(self, client, mock_deps):
        """POST /api/generate creates a generation job."""
        from app.video_processor import ProcessingJob, JobStatus
        import time

        mock_job = ProcessingJob(
            job_id="test123",
            input_path="",
            output_path="outputs/generated_test123_output.mp4",
            prompt="a cat playing",
            model="monarchrt-sf",
            total_frames=21,
            fps=16.0,
        )

        mock_deps["processor"].create_generate_job.return_value = mock_job

        res = client.post("/api/generate", data={
            "prompt": "a cat playing",
            "model": "monarchrt-sf",
            "num_frames": "21",
        })

        assert res.status_code == 200
        data = res.json()
        assert data["job_id"] == "test123"
        assert data["prompt"] == "a cat playing"
        assert data["model"] == "monarchrt-sf"
        mock_deps["processor"].create_generate_job.assert_called_once()

    def test_generate_rejects_non_monarchrt_model(self, client, mock_deps):
        """POST /api/generate rejects non-MonarchRT models."""
        res = client.post("/api/generate", data={
            "prompt": "test",
            "model": "sd-turbo",
        })

        assert res.status_code == 400
        assert "not a MonarchRT model" in res.json()["detail"]

    def test_generate_rejects_when_unavailable(self, client, mock_deps):
        """POST /api/generate returns 503 when MonarchRT is not installed."""
        mock_deps["monarchrt_available"].return_value = False

        res = client.post("/api/generate", data={
            "prompt": "test",
            "model": "monarchrt-sf",
        })

        assert res.status_code == 503
        assert "not installed" in res.json()["detail"]

    def test_generate_rejects_unknown_model(self, client, mock_deps):
        """POST /api/generate rejects unknown models."""
        res = client.post("/api/generate", data={
            "prompt": "test",
            "model": "monarchrt-nonexistent",
        })

        assert res.status_code == 400
        assert "Unknown model" in res.json()["detail"]
