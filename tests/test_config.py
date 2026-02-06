"""Tests for app/config.py - model configs, style presets, AppConfig."""

import os
from unittest.mock import patch

import pytest


def test_model_config_defaults():
    """ModelConfig has sensible defaults."""
    from app.config import ModelConfig

    cfg = ModelConfig(id="test/model")
    assert cfg.id == "test/model"
    assert cfg.use_lcm_lora is False
    assert cfg.pipeline_type == "streamdiffusion"
    assert cfg.num_inference_steps == 1
    assert cfg.guidance_scale is None
    assert cfg.width is None
    assert cfg.height is None


def test_model_config_flux():
    """FLUX model config overrides resolution."""
    from app.config import ModelConfig

    cfg = ModelConfig(
        id="flux-model",
        pipeline_type="flux",
        width=512,
        height=512,
        guidance_scale=1.0,
        num_inference_steps=4,
    )
    assert cfg.pipeline_type == "flux"
    assert cfg.width == 512
    assert cfg.guidance_scale == 1.0


def test_models_dict_is_populated():
    """MODELS dict has known keys."""
    from app.config import MODELS

    assert "sd-turbo" in MODELS
    assert "sd15-lcm" in MODELS
    assert "flux2-klein" in MODELS
    assert "sd-turbo-1.58bit" in MODELS


def test_model_config_descriptions():
    """Every model has a non-empty description."""
    from app.config import MODELS

    for key, model in MODELS.items():
        assert model.description, f"Model {key} has no description"


def test_quantized_model_has_base():
    """Quantized models reference a valid base model."""
    from app.config import MODELS

    for key, model in MODELS.items():
        if "quantized" in model.pipeline_type:
            assert model.base_model, f"Quantized model {key} has no base_model"
            assert model.base_model in MODELS, (
                f"Quantized model {key} references unknown base {model.base_model}"
            )


def test_default_model_exists():
    """DEFAULT_MODEL is a valid key in MODELS."""
    from app.config import MODELS, DEFAULT_MODEL

    assert DEFAULT_MODEL in MODELS


def test_app_config_defaults():
    """AppConfig uses sensible defaults."""
    from app.config import AppConfig

    cfg = AppConfig()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 7860
    assert cfg.width == 512
    assert cfg.height == 512
    assert cfg.max_queue_size == 4
    assert cfg.timeout == 0


def test_app_config_from_env():
    """AppConfig can be loaded from environment variables."""
    from app.config import AppConfig

    with patch.dict(os.environ, {"HOST": "127.0.0.1", "PORT": "9000", "DEBUG": "true"}):
        cfg = AppConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "7860")),
            debug=os.getenv("DEBUG", "").lower() == "true",
        )
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9000
        assert cfg.debug is True


def test_prompt_presets_populated():
    """PROMPT_PRESETS dict is non-empty with valid entries."""
    from app.config import PROMPT_PRESETS

    assert len(PROMPT_PRESETS) > 0
    for key, preset in PROMPT_PRESETS.items():
        assert preset.prompt, f"Preset {key} has no prompt"
        assert preset.description, f"Preset {key} has no description"


def test_default_preset_exists():
    """DEFAULT_PRESET is a valid key in PROMPT_PRESETS."""
    from app.config import PROMPT_PRESETS, DEFAULT_PRESET

    assert DEFAULT_PRESET in PROMPT_PRESETS


def test_prompt_preset_model():
    """PromptPreset model validates fields."""
    from app.config import PromptPreset

    p = PromptPreset(prompt="test prompt", description="Test")
    assert p.prompt == "test prompt"
    assert p.description == "Test"


def test_pipeline_type_values():
    """pipeline_type only accepts valid literals."""
    from app.config import ModelConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModelConfig(id="bad", pipeline_type="invalid-type")

    # Valid types should all work
    valid_types = ["streamdiffusion", "streamdiffusion-quantized", "diffusers", "flux", "flux-quantized", "flux-4bit"]
    for pt in valid_types:
        cfg = ModelConfig(id="test", pipeline_type=pt)
        assert cfg.pipeline_type == pt
