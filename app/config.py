"""Configuration for StreamDiffusion real-time demo."""

from typing import Dict, Any, Literal
from pydantic import BaseModel
import os


class ModelConfig(BaseModel):
    """Configuration for a diffusion model."""
    id: str
    use_lcm_lora: bool = False
    description: str = ""


# Available model presets
MODELS: Dict[str, ModelConfig] = {
    "sd-turbo": ModelConfig(
        id="stabilityai/sd-turbo",
        use_lcm_lora=False,
        description="SD-Turbo (fast, default)"
    ),
    "sd15-lcm": ModelConfig(
        id="runwayml/stable-diffusion-v1-5",
        use_lcm_lora=True,
        description="SD 1.5 + LCM-LoRA"
    ),
}

DEFAULT_MODEL = "sd-turbo"


class AppConfig(BaseModel):
    """Application configuration."""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 7860
    reload: bool = False

    # Paths
    videos_dir: str = "videos"
    engines_dir: str = "engines"

    # Pipeline settings
    width: int = 512
    height: int = 512
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt"
    use_tiny_vae: bool = True

    # Connection settings
    max_queue_size: int = 4
    timeout: int = 0  # 0 = no timeout

    # Debug
    debug: bool = False


# Load config from environment or use defaults
config = AppConfig(
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "7860")),
    videos_dir=os.getenv("VIDEOS_DIR", "videos"),
    engines_dir=os.getenv("ENGINES_DIR", "engines"),
    debug=os.getenv("DEBUG", "").lower() == "true",
)


# Default prompts
DEFAULT_PROMPT = "masterpiece, best quality, highly detailed"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted"
