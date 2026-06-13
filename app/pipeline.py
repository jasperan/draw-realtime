"""StreamDiffusion pipeline wrapper with model switching support.

Supports:
- StreamDiffusion models (SD-Turbo, SD 1.5 + LCM)
- Diffusers models (Hyper-SDXL)
- FLUX models (FLUX.2 Klein)
- 1.58-bit quantized models (BitNet-style PTQ)
- MonarchRT models (real-time video generation via Monarch attention DiT)
"""

import os
import sys
import gc
import threading
from pathlib import Path
from typing import Optional, Dict, Union, Any
from PIL import Image
import torch
import numpy as np

from app.startup_hygiene import install_warning_filters, normalize_hf_telemetry_env

normalize_hf_telemetry_env()
install_warning_filters()

# Add StreamDiffusion to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "StreamDiffusion"))

from utils.wrapper import StreamDiffusionWrapper
from app.config import MODELS, DEFAULT_MODEL, DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT, config

# Quantization support
from app.quantization import BitLinear, TernaryLinear
from app.quantization.utils import (
    load_quantized_model,
    get_model_size_mb,
    get_module,
    replace_module,
)


# Serializes GPU access so concurrent jobs can't clobber each other mid-load.
# It's a threading.Lock (not asyncio.Lock) because worker bodies run inside
# asyncio.to_thread(...) — a thread lock is loop-agnostic and survives the
# fresh-event-loop-per-test pattern used by pytest-asyncio.
_pipeline_lock = threading.Lock()


def pipeline_lock() -> threading.Lock:
    """Return the shared lock that serializes access to the Pipeline singleton."""
    return _pipeline_lock


class Pipeline:
    """Manages StreamDiffusion and Diffusers pipelines with model switching."""

    # Which instance attribute holds the active pipe for each pipeline_type.
    # Single source of truth for cleanup, already-loaded check, and dispatch.
    _PIPE_ATTR: Dict[str, str] = {
        "streamdiffusion": "stream",
        "streamdiffusion-quantized": "stream",
        "diffusers": "diffusers_pipe",
        "flux": "flux_pipe",
        "flux-quantized": "flux_pipe",
        "flux-4bit": "flux_pipe",
        "monarchrt": "monarchrt_pipe",
    }

    def __init__(
        self,
        model_key: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        self.current_model_key: Optional[str] = None
        self.current_pipeline_type: Optional[str] = None
        self.stream: Optional[StreamDiffusionWrapper] = None  # For StreamDiffusion
        self.diffusers_pipe: Optional[Any] = None  # For Diffusers AutoPipeline
        self.flux_pipe: Optional[Any] = None  # For FLUX.2 Klein pipeline
        self.monarchrt_pipe: Optional[Any] = None  # For MonarchRT pipeline
        self.current_prompt = DEFAULT_PROMPT
        self.current_negative_prompt = DEFAULT_NEGATIVE_PROMPT

        # Load initial model
        self.load_model(model_key)

    def load_model(self, model_key: str) -> bool:
        """Load a model by its key. Returns True if successful."""
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}")
            return False

        model_config = MODELS[model_key]

        # Already loaded? Check via the attr table.
        if model_key == self.current_model_key:
            attr = self._PIPE_ATTR.get(model_config.pipeline_type)
            if attr and getattr(self, attr, None) is not None:
                print(f"Model {model_key} already loaded")
                return True

        self._cleanup()
        print(f"Loading model: {model_config.description}")

        loaders = {
            "monarchrt": self._load_monarchrt_model,
            "flux": self._load_flux_model,
            "flux-quantized": self._load_flux_quantized_model,
            "flux-4bit": self._load_flux_4bit_model,
            "diffusers": self._load_diffusers_model,
            "streamdiffusion-quantized": self._load_quantized_model,
        }
        loader = loaders.get(model_config.pipeline_type, self._load_streamdiffusion_model)
        return loader(model_key, model_config)

    def _cleanup(self):
        """Clean up any loaded models and free GPU memory."""
        disposers = {
            "stream": self._dispose_stream,
            "diffusers_pipe": self._dispose_diffusers_like,
            "flux_pipe": self._dispose_diffusers_like,
            "monarchrt_pipe": self._dispose_monarchrt,
        }
        for attr, dispose in disposers.items():
            pipe = getattr(self, attr, None)
            if pipe is None:
                continue
            try:
                dispose(pipe)
            except Exception:
                pass
            setattr(self, attr, None)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _dispose_stream(stream):
        # StreamDiffusion may hold internal references.
        if hasattr(stream, "stream") and stream.stream is not None:
            del stream.stream

    @staticmethod
    def _dispose_diffusers_like(pipe):
        # Move components to CPU so GPU VRAM is released on del.
        if hasattr(pipe, "components"):
            for component in pipe.components.values():
                if hasattr(component, "to"):
                    component.to("cpu")

    @staticmethod
    def _dispose_monarchrt(pipe):
        pipe.cleanup()

    def _load_monarchrt_model(self, model_key: str, model_config) -> bool:
        """Load a MonarchRT model for real-time video generation."""
        try:
            from app.monarchrt_pipeline import MonarchRTPipeline, is_monarchrt_available

            if not is_monarchrt_available():
                print("MonarchRT is not installed. See MonarchRT/README.md for installation.")
                return False

            self.monarchrt_pipe = MonarchRTPipeline(device=self.device)

            # Resolve relative paths
            project_root = Path(__file__).parent.parent

            if model_config.monarchrt_mode == "causal":
                config_path = str(project_root / model_config.monarchrt_config)
                checkpoint_path = str(project_root / model_config.monarchrt_checkpoint)
                success = self.monarchrt_pipe.load_causal_model(
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                )
            elif model_config.monarchrt_mode == "bidirectional":
                checkpoint_dir = str(project_root / model_config.monarchrt_checkpoint) if model_config.monarchrt_checkpoint else ""
                success = self.monarchrt_pipe.load_bidirectional_model(
                    model_name=model_config.monarchrt_model_name,
                    checkpoint_dir=checkpoint_dir,
                )
            else:
                print(f"Unknown MonarchRT mode: {model_config.monarchrt_mode}")
                return False

            if success:
                self.current_model_key = model_key
                self.current_pipeline_type = "monarchrt"
                print(f"MonarchRT model {model_key} loaded successfully")
            else:
                self.monarchrt_pipe = None

            return success

        except Exception as e:
            print(f"Failed to load MonarchRT model {model_key}: {e}")
            import traceback
            traceback.print_exc()
            self.monarchrt_pipe = None
            return False

    def generate_video(
        self,
        prompt: str,
        num_frames: int = 21,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: int = -1,
    ) -> Optional[list]:
        """Generate a complete video from a text prompt (MonarchRT only).

        Returns:
            List of PIL Images (one per frame), or None on failure.
        """
        if self.current_pipeline_type != "monarchrt" or self.monarchrt_pipe is None:
            print("generate_video() requires a loaded MonarchRT model")
            return None

        model_config = MODELS.get(self.current_model_key)
        out_w = width or (model_config.width if model_config else 832)
        out_h = height or (model_config.height if model_config else 480)
        steps = model_config.num_inference_steps if model_config else 4
        guidance = model_config.guidance_scale if model_config and model_config.guidance_scale else 3.0

        return self.monarchrt_pipe.generate_video(
            prompt=prompt,
            num_frames=num_frames,
            width=out_w,
            height=out_h,
            seed=seed,
            guidance_scale=guidance,
            num_inference_steps=steps,
            negative_prompt=self.current_negative_prompt,
        )

    def _load_diffusers_model(self, model_key: str, model_config) -> bool:
        """Load a model using Diffusers AutoPipeline."""
        try:
            from diffusers import AutoPipelineForImage2Image

            print(f"Loading Diffusers pipeline: {model_config.id}")

            self.diffusers_pipe = AutoPipelineForImage2Image.from_pretrained(
                model_config.id,
                torch_dtype=self.dtype,
                variant="fp16",
            ).to(self.device)

            # Load LoRA if specified (e.g., Hyper-SD)
            if model_config.lora_id:
                print(f"Loading LoRA: {model_config.lora_id}")
                # For Hyper-SDXL, load the 1-step LoRA
                self.diffusers_pipe.load_lora_weights(
                    model_config.lora_id,
                    weight_name="Hyper-SDXL-1step-lora.safetensors"
                )
                self.diffusers_pipe.fuse_lora()

            # Optimize for speed
            self.diffusers_pipe.set_progress_bar_config(disable=True)

            self.current_model_key = model_key
            self.current_pipeline_type = "diffusers"
            print(f"Model {model_key} loaded successfully")
            return True

        except Exception as e:
            print(f"Failed to load Diffusers model {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_flux_model(self, model_key: str, model_config) -> bool:
        """Load a model using the FLUX.2 Klein pipeline."""
        try:
            from diffusers import Flux2KleinPipeline

            print(f"Loading FLUX.2 Klein pipeline: {model_config.id}")

            self.flux_pipe = Flux2KleinPipeline.from_pretrained(
                model_config.id,
                torch_dtype=torch.bfloat16,  # FLUX requires bfloat16
            ).to(self.device)

            self.flux_pipe.set_progress_bar_config(disable=True)

            self.current_model_key = model_key
            self.current_pipeline_type = "flux"
            print(f"Model {model_key} loaded successfully")
            return True

        except Exception as e:
            print(f"Failed to load FLUX model {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_flux_quantized_model(self, model_key: str, model_config) -> bool:
        """Load a 1.58-bit quantized FLUX model.

        This loads the base FLUX model, then replaces the transformer
        weights with the quantized version.
        """
        try:
            from diffusers import Flux2KleinPipeline
            import json
            from app.quantization import BitLinear

            # Determine quantized model path
            quantized_path = Path(model_config.quantized_path)
            if not quantized_path.is_absolute():
                quantized_path = Path(__file__).parent.parent / quantized_path

            if not quantized_path.exists():
                print(f"Quantized model not found at: {quantized_path}")
                print(f"Run 'python scripts/quantize_flux.py' first")
                return False

            print(f"Loading quantized FLUX model from: {quantized_path}")

            # Load base FLUX pipeline
            print(f"Loading base FLUX.2 Klein pipeline...")
            self.flux_pipe = Flux2KleinPipeline.from_pretrained(
                model_config.id,
                torch_dtype=torch.bfloat16,
            ).to(self.device)

            # Load quantization config
            config_path = quantized_path / "config.json"
            with open(config_path) as f:
                quant_config = json.load(f)

            # Load quantized weights
            weights_path = quantized_path / "transformer_quantized.safetensors"
            if weights_path.exists():
                from safetensors.torch import load_file
                state_dict = load_file(str(weights_path))
            else:
                weights_path = quantized_path / "transformer_quantized.pt"
                state_dict = torch.load(weights_path, map_location='cpu')

            # Replace Linear layers with BitLinear for quantized layers
            transformer = self.flux_pipe.transformer
            quantized_layers = quant_config.get('layer_names', [])

            print(f"Replacing {len(quantized_layers)} layers with BitLinear...")
            for layer_name in quantized_layers:
                # Get original module to extract dimensions
                original = get_module(transformer, layer_name)
                has_bias = original.bias is not None

                # Create BitLinear replacement
                bit_linear = BitLinear(
                    in_features=original.in_features,
                    out_features=original.out_features,
                    bias=has_bias,
                    device='cpu',  # Load to CPU first
                    dtype=torch.bfloat16,
                )

                # Load quantized weights
                prefix = layer_name
                bit_linear.weight_ternary = state_dict[f"{prefix}.weight_ternary"]
                bit_linear.weight_scale = state_dict[f"{prefix}.weight_scale"]
                if has_bias and f"{prefix}.bias" in state_dict:
                    bit_linear.bias = torch.nn.Parameter(state_dict[f"{prefix}.bias"])

                # Move to device and replace
                bit_linear = bit_linear.to(self.device)
                replace_module(transformer, layer_name, bit_linear)

            # Clear original state dict to free memory
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()

            # Optionally convert BitLinear to TernaryLinear for faster inference
            if model_config.use_ternary_linear:
                print("Converting BitLinear → TernaryLinear (cached mode)...")
                converted = self._convert_bitlinear_to_ternary(transformer)
                print(f"Converted {converted} layers to TernaryLinear")

            # Optionally apply torch.compile for additional speedup
            if model_config.use_torch_compile:
                print("Applying torch.compile (first inference will be slow)...")
                self.flux_pipe.transformer = torch.compile(
                    self.flux_pipe.transformer,
                    mode="reduce-overhead"
                )

            self.flux_pipe.set_progress_bar_config(disable=True)

            self.current_model_key = model_key
            self.current_pipeline_type = "flux-quantized"
            print(f"Quantized FLUX model {model_key} loaded successfully")
            return True

        except Exception as e:
            print(f"Failed to load quantized FLUX model {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_flux_4bit_model(self, model_key: str, model_config) -> bool:
        """Load FLUX model with 4-bit NF4 quantization using bitsandbytes.

        This provides a good balance between speed, memory, and quality:
        - 3.5x faster than original
        - 45% less VRAM
        - Quality nearly identical to original
        """
        try:
            from transformers import BitsAndBytesConfig
            from diffusers import Flux2KleinPipeline
            from diffusers.models import Flux2Transformer2DModel

            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # Normal Float 4
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,  # Nested quantization
            )

            print("Loading transformer with 4-bit NF4 quantization...")

            # Load transformer with 4-bit quantization
            transformer = Flux2Transformer2DModel.from_pretrained(
                model_config.id,
                subfolder="transformer",
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )

            print("Loading rest of pipeline...")

            # Load the rest of the pipeline
            self.flux_pipe = Flux2KleinPipeline.from_pretrained(
                model_config.id,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )

            # Move non-transformer components to GPU
            self.flux_pipe.vae.to(self.device)
            self.flux_pipe.text_encoder.to(self.device)

            self.flux_pipe.set_progress_bar_config(disable=True)

            self.current_model_key = model_key
            self.current_pipeline_type = "flux-4bit"
            print(f"FLUX 4-bit model {model_key} loaded successfully")
            return True

        except Exception as e:
            print(f"Failed to load FLUX 4-bit model {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _convert_bitlinear_to_ternary(self, model: torch.nn.Module) -> int:
        """Convert all BitLinear layers to TernaryLinear for faster inference.

        TernaryLinear uses cached weights (pre-unpacked and scaled) which
        enables fast cuBLAS matmul instead of on-the-fly dequantization.

        Returns:
            Number of layers converted
        """
        converted = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, BitLinear):
                # Convert to TernaryLinear with cached weights
                ternary = TernaryLinear.from_bitlinear(
                    module,
                    use_triton=False,  # Use cached cuBLAS mode
                    cache_weights=True
                )
                replace_module(model, name, ternary)
                converted += 1

        return converted

    def _build_streamdiffusion_wrapper(
        self,
        model_id: str,
        use_lcm_lora: bool,
        acceleration: str,
        engine_dir: Optional[str],
    ) -> StreamDiffusionWrapper:
        """Build a StreamDiffusionWrapper with a given acceleration backend.

        engine_dir is only passed when using TensorRT (the xformers path
        doesn't need a compiled engine cache).
        """
        kwargs: Dict[str, Any] = dict(
            model_id_or_path=model_id,
            t_index_list=[35, 45],
            frame_buffer_size=1,
            width=config.width,
            height=config.height,
            use_tiny_vae=config.use_tiny_vae,
            warmup=10,
            acceleration=acceleration,
            do_add_noise=False,
            mode="img2img",
            output_type="pil",
            use_denoising_batch=True,
            cfg_type="none",
            use_lcm_lora=use_lcm_lora,
            use_safety_checker=False,
            device=self.device,
            dtype=self.dtype,
        )
        if engine_dir is not None:
            kwargs["engine_dir"] = engine_dir
        return StreamDiffusionWrapper(**kwargs)

    def _prepare_stream(self):
        """Run the default prepare() call on the active stream."""
        self.stream.prepare(
            prompt=self.current_prompt,
            negative_prompt=self.current_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2,
        )

    def _load_streamdiffusion_model(self, model_key: str, model_config) -> bool:
        """Load a model using StreamDiffusion wrapper, with xformers fallback."""
        def _build(acceleration: str, engine_dir: Optional[str]):
            self.stream = self._build_streamdiffusion_wrapper(
                model_config.id, model_config.use_lcm_lora, acceleration, engine_dir,
            )
            self._prepare_stream()

        try:
            _build(config.acceleration, config.engines_dir)
            self.current_model_key = model_key
            self.current_pipeline_type = "streamdiffusion"
            print(f"Model {model_key} loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load model {model_key}: {e}")
            if config.acceleration != "tensorrt":
                return False

        print("Falling back to xformers acceleration...")
        try:
            _build("xformers", None)
            self.current_model_key = model_key
            self.current_pipeline_type = "streamdiffusion"
            print(f"Model {model_key} loaded with xformers fallback")
            return True
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return False

    def _load_quantized_model(self, model_key: str, model_config) -> bool:
        """Load a 1.58-bit quantized StreamDiffusion model.

        This loads the base model via StreamDiffusion, then replaces the U-Net
        weights with the quantized version.
        """
        try:
            # Get the base model key (e.g., "sd-turbo" from "sd-turbo-1.58bit")
            base_model_key = model_config.base_model
            base_config = MODELS.get(base_model_key)

            if not base_config:
                print(f"Base model {base_model_key} not found in MODELS config")
                return False

            # Determine quantized model path
            quantized_path = Path(model_config.quantized_path)
            if not quantized_path.is_absolute():
                quantized_path = Path(__file__).parent.parent / quantized_path

            if not quantized_path.exists():
                print(f"Quantized model not found at: {quantized_path}")
                print(f"Run 'python scripts/quantize_model.py --model {base_model_key}' first")
                return False

            print(f"Loading quantized model from: {quantized_path}")

            # TensorRT can't compile custom quantized layers, so quantized
            # models always run through xformers.
            self.stream = self._build_streamdiffusion_wrapper(
                base_config.id, base_config.use_lcm_lora, "xformers", engine_dir=None,
            )

            print("Loading quantized U-Net weights...")
            unet = self.stream.stream.unet
            load_quantized_model(unet, str(quantized_path), device=self.device)
            print(f"Quantized U-Net size: {get_model_size_mb(unet):.2f} MB")

            self._prepare_stream()

            self.current_model_key = model_key
            self.current_pipeline_type = "streamdiffusion-quantized"
            print(f"Quantized model {model_key} loaded successfully")
            return True

        except Exception as e:
            print(f"Failed to load quantized model {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_prompt(self, prompt: str, negative_prompt: Optional[str] = None):
        """Update the generation prompt.

        Diffusers-style pipes don't need a separate prompt update — the
        prompt is passed at inference time. Only StreamDiffusion needs
        to push the new prompt into its internal scheduler state.
        """
        if prompt != self.current_prompt:
            self.current_prompt = prompt
            if self.stream is not None:
                self._streamdiffusion_set_prompt(prompt)

        if negative_prompt and negative_prompt != self.current_negative_prompt:
            self.current_negative_prompt = negative_prompt

    def _streamdiffusion_set_prompt(self, prompt: str):
        """Push a prompt into StreamDiffusion's internal scheduler.

        The upstream StreamDiffusionWrapper doesn't expose update_prompt,
        so we have to reach through `.stream` to the underlying
        StreamDiffusion object. This helper is the single place that
        coupling lives — if the wrapper ever adds a surface method, only
        this function needs to change.
        """
        self.stream.stream.update_prompt(prompt)

    def get_output_resolution(self) -> tuple[int, int]:
        """Get the output resolution for the currently loaded model."""
        if self.current_model_key and self.current_model_key in MODELS:
            model_config = MODELS[self.current_model_key]
            w = model_config.width or config.width
            h = model_config.height or config.height
            return (w, h)
        return (config.width, config.height)

    def predict(self, image: Union[Image.Image, np.ndarray]) -> Optional[Image.Image]:
        """Process an image through the active pipeline."""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            out_w, out_h = self.get_output_resolution()
            image = image.resize((out_w, out_h))

            if self.current_pipeline_type in ("flux", "flux-quantized", "flux-4bit") and self.flux_pipe is not None:
                model_config = MODELS[self.current_model_key]
                return self.flux_pipe(
                    prompt=self.current_prompt,
                    image=image,
                    height=out_h,
                    width=out_w,
                    guidance_scale=model_config.guidance_scale,
                    num_inference_steps=model_config.num_inference_steps,
                ).images[0]

            if self.current_pipeline_type == "diffusers" and self.diffusers_pipe is not None:
                model_config = MODELS[self.current_model_key]
                return self.diffusers_pipe(
                    prompt=self.current_prompt,
                    negative_prompt=self.current_negative_prompt,
                    image=image,
                    num_inference_steps=model_config.num_inference_steps,
                    strength=0.5,
                    guidance_scale=0.0,  # 1-step models (Hyper-SDXL) need CFG=0
                ).images[0]

            if self.stream is not None:
                image_tensor = self.stream.preprocess_image(image)
                return self.stream(image=image_tensor)

            return None
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models."""
        return {key: cfg.description for key, cfg in MODELS.items()}

    def get_current_model(self) -> Optional[str]:
        """Get the currently loaded model key."""
        return self.current_model_key


# Global pipeline instance
_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _pipeline = Pipeline(device=device)
    return _pipeline
