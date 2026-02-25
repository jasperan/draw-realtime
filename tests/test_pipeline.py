"""Tests for app/pipeline.py - model switching logic (all GPU ops mocked)."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

# conftest.py already mocks sys.modules for utils.wrapper and streamdiffusion


# ---------------------------------------------------------------------------
# Helper to create Pipeline instance without loading a real model
# ---------------------------------------------------------------------------

def _make_bare_pipeline():
    """Create a Pipeline instance with __init__ skipped."""
    from app.pipeline import Pipeline
    p = Pipeline.__new__(Pipeline)
    p.device = "cpu"
    p.dtype = torch.float16
    p.current_model_key = None
    p.current_pipeline_type = None
    p.stream = None
    p.diffusers_pipe = None
    p.flux_pipe = None
    p.monarchrt_pipe = None
    p.current_prompt = "test prompt"
    p.current_negative_prompt = "bad"
    return p


# ---------------------------------------------------------------------------
# Pipeline class tests (mocked model loading)
# ---------------------------------------------------------------------------

class TestPipelineModelSwitching:
    """Test model loading/switching logic without real models."""

    def test_unknown_model_returns_false(self):
        """load_model returns False for unknown model key."""
        p = _make_bare_pipeline()
        result = p.load_model("nonexistent-model")
        assert result is False

    def test_get_available_models(self):
        """get_available_models returns dict of model keys to descriptions."""
        p = _make_bare_pipeline()
        result = p.get_available_models()
        assert isinstance(result, dict)
        assert "sd-turbo" in result

    def test_get_current_model_initially_none(self):
        """get_current_model returns None when nothing is loaded."""
        p = _make_bare_pipeline()
        assert p.get_current_model() is None

    def test_get_output_resolution_default(self):
        """get_output_resolution returns config default when no model loaded."""
        p = _make_bare_pipeline()
        w, h = p.get_output_resolution()
        assert w > 0
        assert h > 0

    def test_get_output_resolution_flux_model(self):
        """get_output_resolution uses model-specific resolution for FLUX."""
        p = _make_bare_pipeline()
        p.current_model_key = "flux2-klein"
        w, h = p.get_output_resolution()
        assert w == 512
        assert h == 512


class TestPipelineUpdatePrompt:
    """Test prompt update logic."""

    def test_update_prompt_changes_current(self):
        p = _make_bare_pipeline()
        p.update_prompt("new prompt")
        assert p.current_prompt == "new prompt"

    def test_update_prompt_noop_same_prompt(self):
        p = _make_bare_pipeline()
        p.update_prompt("test prompt")  # same as initial
        assert p.current_prompt == "test prompt"

    def test_update_negative_prompt(self):
        p = _make_bare_pipeline()
        p.update_prompt("new", negative_prompt="new negative")
        assert p.current_negative_prompt == "new negative"

    def test_update_prompt_calls_stream(self):
        p = _make_bare_pipeline()
        mock_stream = MagicMock()
        p.stream = mock_stream
        p.update_prompt("updated prompt")
        mock_stream.stream.update_prompt.assert_called_once_with("updated prompt")


class TestPipelinePredict:
    """Test predict logic with mocked pipelines."""

    def test_predict_returns_none_no_pipeline(self):
        p = _make_bare_pipeline()
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        result = p.predict(img)
        assert result is None

    def test_predict_accepts_numpy_array(self):
        p = _make_bare_pipeline()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = p.predict(arr)
        assert result is None  # no pipeline loaded, but shouldn't error

    def test_predict_with_diffusers_pipe(self):
        p = _make_bare_pipeline()
        p.current_pipeline_type = "diffusers"
        p.current_model_key = "hyper-sdxl"

        mock_pipe = MagicMock()
        output_img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        mock_pipe.return_value.images = [output_img]
        p.diffusers_pipe = mock_pipe

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        result = p.predict(img)
        assert result is output_img

    def test_predict_error_returns_none(self):
        p = _make_bare_pipeline()
        p.current_pipeline_type = "diffusers"
        p.current_model_key = "hyper-sdxl"

        mock_pipe = MagicMock()
        mock_pipe.side_effect = RuntimeError("GPU OOM")
        p.diffusers_pipe = mock_pipe

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        result = p.predict(img)
        assert result is None


class TestPipelineCleanup:
    """Test _cleanup method."""

    def test_cleanup_sets_references_to_none(self):
        with patch("torch.cuda.empty_cache"):
            p = _make_bare_pipeline()
            p.stream = MagicMock()
            p.diffusers_pipe = MagicMock()
            p.flux_pipe = MagicMock()

            p._cleanup()

            assert p.stream is None
            assert p.diffusers_pipe is None
            assert p.flux_pipe is None
