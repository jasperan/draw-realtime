"""Shared test fixtures for draw-realtime tests.

All GPU/model operations are mocked so tests run without a GPU.
"""

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level mocking: prevent heavy imports (StreamDiffusion, etc.)
# These MUST be set before any app.pipeline import happens.
# ---------------------------------------------------------------------------
sys.modules.setdefault("utils", MagicMock())
sys.modules.setdefault("utils.wrapper", MagicMock())
sys.modules.setdefault("streamdiffusion", MagicMock())


# ---------------------------------------------------------------------------
# Temp directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory that is cleaned up after the test."""
    return tmp_path


@pytest.fixture
def videos_dir(tmp_path):
    """Temporary videos directory."""
    d = tmp_path / "videos"
    d.mkdir()
    return d


@pytest.fixture
def outputs_dir(tmp_path):
    """Temporary outputs directory."""
    d = tmp_path / "outputs"
    d.mkdir()
    return d


@pytest.fixture
def uploads_dir(tmp_path):
    """Temporary uploads directory."""
    d = tmp_path / "uploads"
    d.mkdir()
    return d


@pytest.fixture
def previews_dir(tmp_path):
    """Temporary previews directory."""
    d = tmp_path / "previews"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Dummy video creation helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def create_dummy_video(tmp_path):
    """Factory fixture that creates a minimal valid video file using raw bytes.

    Returns a callable(filename, dir=None) -> Path.
    """
    def _create(filename="test.mp4", directory=None):
        directory = directory or tmp_path
        video_path = Path(directory) / filename
        # Create a minimal MP4 file using opencv if available, otherwise raw bytes
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (64, 64))
            for _ in range(10):
                frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()
        except ImportError:
            # Fallback: write minimal bytes so the file exists
            video_path.write_bytes(b'\x00' * 1024)
        return video_path

    return _create


@pytest.fixture
def dummy_pil_image():
    """Return a simple 64x64 PIL Image."""
    return Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Mock pipeline / model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pipeline():
    """A mock Pipeline object that does not require GPU or real models."""
    pipeline = MagicMock()
    pipeline.get_current_model.return_value = "sd-turbo"
    pipeline.get_available_models.return_value = {
        "sd-turbo": "SD-Turbo (fast, lower quality)",
        "sd15-lcm": "SD 1.5 + LCM-LoRA (slower, higher quality)",
    }
    pipeline.get_output_resolution.return_value = (512, 512)
    pipeline.predict.return_value = Image.fromarray(
        np.zeros((512, 512, 3), dtype=np.uint8)
    )
    pipeline.load_model.return_value = True
    pipeline.update_prompt.return_value = None
    return pipeline


@pytest.fixture
def mock_torch_cuda():
    """Patch torch.cuda so GPU calls don't fail."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.empty_cache"):
        yield


