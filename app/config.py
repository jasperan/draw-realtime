"""Configuration for StreamDiffusion real-time demo."""

from typing import Dict, Any, Literal, Optional
from pydantic import BaseModel
import os


class ModelConfig(BaseModel):
    """Configuration for a diffusion model."""
    id: str
    use_lcm_lora: bool = False
    description: str = ""
    pipeline_type: Literal["streamdiffusion", "streamdiffusion-quantized", "diffusers", "flux", "flux-quantized", "flux-4bit"] = "streamdiffusion"
    lora_id: str = ""  # Optional LoRA to load
    num_inference_steps: int = 1  # For diffusers/flux pipeline
    guidance_scale: Optional[float] = None  # Per-model guidance scale
    width: Optional[int] = None  # Per-model output width (overrides config.width)
    height: Optional[int] = None  # Per-model output height (overrides config.height)
    # Quantization-specific fields
    base_model: str = ""  # For quantized models: the base model key
    quantized_path: str = ""  # Path to quantized weights
    use_ternary_linear: bool = True  # Convert BitLinear → TernaryLinear for speed (uses more VRAM)
    use_torch_compile: bool = False  # Use torch.compile for 1.2x speedup (slow first inference)


# Available model presets
MODELS: Dict[str, ModelConfig] = {
    "sd-turbo": ModelConfig(
        id="stabilityai/sd-turbo",
        use_lcm_lora=False,
        description="SD-Turbo (fast, lower quality)",
        pipeline_type="streamdiffusion",
    ),
    "sd15-lcm": ModelConfig(
        id="runwayml/stable-diffusion-v1-5",
        use_lcm_lora=True,
        description="SD 1.5 + LCM-LoRA (slower, higher quality)",
        pipeline_type="streamdiffusion",
    ),
    "hyper-sdxl": ModelConfig(
        id="stabilityai/stable-diffusion-xl-base-1.0",
        use_lcm_lora=False,
        description="Hyper-SDXL 1-step (fastest, SDXL quality)",
        pipeline_type="diffusers",
        lora_id="ByteDance/Hyper-SD",
        num_inference_steps=1,
    ),
    "flux2-klein": ModelConfig(
        id="black-forest-labs/FLUX.2-klein-4B",
        description="FLUX.2 Klein 4B (best quality, recommended)",
        pipeline_type="flux",
        num_inference_steps=4,
        guidance_scale=1.0,
        width=512,
        height=512,
    ),
    # 1.58-bit Quantized Models (BitNet-style PTQ)
    "sd-turbo-1.58bit": ModelConfig(
        id="stabilityai/sd-turbo",
        use_lcm_lora=False,
        description="SD-Turbo 1.58-bit (faster, lower memory)",
        pipeline_type="streamdiffusion-quantized",
        base_model="sd-turbo",
        quantized_path="models/quantized/sd-turbo-1.58bit",
    ),
    "sd15-lcm-1.58bit": ModelConfig(
        id="runwayml/stable-diffusion-v1-5",
        use_lcm_lora=True,
        description="SD 1.5 + LCM 1.58-bit (faster, lower memory)",
        pipeline_type="streamdiffusion-quantized",
        base_model="sd15-lcm",
        quantized_path="models/quantized/sd15-lcm-1.58bit",
    ),
    # FLUX Quantized Models
    "flux-klein-1.58bit": ModelConfig(
        id="black-forest-labs/FLUX.2-klein-4B",
        description="FLUX.2 Klein 1.58-bit (faster, lower memory) [QUALITY ISSUES]",
        pipeline_type="flux-quantized",
        base_model="flux2-klein",
        quantized_path="models/quantized/flux-klein-1.58bit",
        num_inference_steps=4,
        guidance_scale=1.0,
        width=512,
        height=512,
    ),
    # FLUX 4-bit Quantization (for lower VRAM GPUs)
    "flux-klein-4bit": ModelConfig(
        id="black-forest-labs/FLUX.2-klein-4B",
        description="FLUX.2 Klein 4-bit NF4 (36% less VRAM, same quality)",
        pipeline_type="flux-4bit",
        num_inference_steps=4,
        guidance_scale=1.0,
        width=512,
        height=512,
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
    outputs_dir: str = "outputs"
    uploads_dir: str = "uploads"
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


class PromptPreset(BaseModel):
    """A preset prompt with name and description."""
    prompt: str
    description: str


# Preset prompts for different styles
PROMPT_PRESETS: Dict[str, PromptPreset] = {
    "anime-ghibli": PromptPreset(
        prompt="anime style, studio ghibli inspired, soft colors, whimsical, magical atmosphere, hand-drawn, beautiful scenery, masterpiece",
        description="Anime - Studio Ghibli"
    ),
    "anime-cyberpunk": PromptPreset(
        prompt="anime cyberpunk style, neon lights, futuristic city, makoto shinkai inspired, vibrant colors, detailed, cinematic lighting",
        description="Anime - Cyberpunk"
    ),
    "cyberpunk-neon": PromptPreset(
        prompt="cyberpunk neon city, futuristic, rain, neon lights reflecting on wet streets, high tech, vibrant colors, highly detailed",
        description="Cyberpunk Neon City"
    ),
    "oil-painting": PromptPreset(
        prompt="oil painting style, classical art, rich colors, visible brushstrokes, dramatic lighting, renaissance inspired, masterpiece",
        description="Oil Painting - Classical"
    ),
    "watercolor": PromptPreset(
        prompt="watercolor painting, soft edges, flowing colors, artistic, delicate, paper texture, impressionist style, beautiful",
        description="Watercolor Painting"
    ),
    "fantasy": PromptPreset(
        prompt="fantasy art style, magical, ethereal lighting, intricate details, epic, mystical atmosphere, trending on artstation",
        description="Fantasy Art"
    ),
    "dark-gothic": PromptPreset(
        prompt="dark gothic style, dramatic shadows, moody atmosphere, dark fantasy, cinematic, muted colors, haunting, detailed",
        description="Dark Gothic"
    ),
    "comic-pop": PromptPreset(
        prompt="comic book style, pop art, bold colors, thick outlines, halftone dots, dynamic, vibrant, graphic novel aesthetic",
        description="Comic / Pop Art"
    ),
    "photorealistic": PromptPreset(
        prompt="photorealistic, ultra detailed, 8k, professional photography, sharp focus, natural lighting, hyperrealistic",
        description="Photorealistic"
    ),
    "impressionist": PromptPreset(
        prompt="impressionist painting style, visible brushstrokes, light and color, monet inspired, dreamy, soft focus, artistic",
        description="Impressionist"
    ),
    "pixel-art": PromptPreset(
        prompt="pixel art style, 16-bit, retro gaming aesthetic, vibrant colors, clean pixels, nostalgic, detailed sprites",
        description="Pixel Art / Retro"
    ),
    "sketch": PromptPreset(
        prompt="pencil sketch, hand-drawn, detailed linework, crosshatching, artistic, monochrome, professional illustration",
        description="Pencil Sketch"
    ),
}

DEFAULT_PRESET = "cyberpunk-neon"
