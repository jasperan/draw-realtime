"""StreamDiffusion pipeline wrapper with model switching support."""

import os
import sys
import gc
from typing import Optional, Dict, Union, Any
from PIL import Image
import torch
import numpy as np

# Add StreamDiffusion to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "StreamDiffusion"))

from utils.wrapper import StreamDiffusionWrapper
from app.config import MODELS, DEFAULT_MODEL, DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT, config


class Pipeline:
    """Manages StreamDiffusion and Diffusers pipelines with model switching."""

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

        # Check if already loaded
        if model_key == self.current_model_key:
            if model_config.pipeline_type == "streamdiffusion" and self.stream is not None:
                print(f"Model {model_key} already loaded")
                return True
            if model_config.pipeline_type == "diffusers" and self.diffusers_pipe is not None:
                print(f"Model {model_key} already loaded")
                return True

        # Clean up existing models
        self._cleanup()

        print(f"Loading model: {model_config.description}")

        if model_config.pipeline_type == "diffusers":
            return self._load_diffusers_model(model_key, model_config)
        else:
            return self._load_streamdiffusion_model(model_key, model_config)

    def _cleanup(self):
        """Clean up any loaded models."""
        if self.stream is not None:
            del self.stream
            self.stream = None
        if self.diffusers_pipe is not None:
            del self.diffusers_pipe
            self.diffusers_pipe = None
        gc.collect()
        torch.cuda.empty_cache()

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

    def _load_streamdiffusion_model(self, model_key: str, model_config) -> bool:
        """Load a model using StreamDiffusion wrapper."""
        try:
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=model_config.id,
                t_index_list=[35, 45],
                frame_buffer_size=1,
                width=config.width,
                height=config.height,
                use_tiny_vae=config.use_tiny_vae,
                warmup=10,
                acceleration=config.acceleration,
                do_add_noise=False,
                mode="img2img",
                output_type="pil",
                use_denoising_batch=True,
                cfg_type="none",
                use_lcm_lora=model_config.use_lcm_lora,
                use_safety_checker=False,
                engine_dir=config.engines_dir,
                device=self.device,
                dtype=self.dtype,
            )

            # Prepare with default prompt
            self.stream.prepare(
                prompt=self.current_prompt,
                negative_prompt=self.current_negative_prompt,
                num_inference_steps=50,
                guidance_scale=1.2,
            )

            self.current_model_key = model_key
            self.current_pipeline_type = "streamdiffusion"
            print(f"Model {model_key} loaded successfully")
            return True

        except Exception as e:
            print(f"Failed to load model {model_key}: {e}")
            # Try fallback to xformers if TensorRT fails
            if config.acceleration == "tensorrt":
                print("Falling back to xformers acceleration...")
                try:
                    self.stream = StreamDiffusionWrapper(
                        model_id_or_path=model_config.id,
                        t_index_list=[35, 45],
                        frame_buffer_size=1,
                        width=config.width,
                        height=config.height,
                        use_tiny_vae=config.use_tiny_vae,
                        warmup=10,
                        acceleration="xformers",
                        do_add_noise=False,
                        mode="img2img",
                        output_type="pil",
                        use_denoising_batch=True,
                        cfg_type="none",
                        use_lcm_lora=model_config.use_lcm_lora,
                        use_safety_checker=False,
                        device=self.device,
                        dtype=self.dtype,
                    )

                    self.stream.prepare(
                        prompt=self.current_prompt,
                        negative_prompt=self.current_negative_prompt,
                        num_inference_steps=50,
                        guidance_scale=1.2,
                    )

                    self.current_model_key = model_key
                    self.current_pipeline_type = "streamdiffusion"
                    print(f"Model {model_key} loaded with xformers fallback")
                    return True
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
            return False

    def update_prompt(self, prompt: str, negative_prompt: Optional[str] = None):
        """Update the generation prompt."""
        if prompt != self.current_prompt:
            self.current_prompt = prompt
            if self.stream:
                self.stream.stream.update_prompt(prompt)
            # Diffusers pipe doesn't need prompt update - it's passed at inference time

        if negative_prompt and negative_prompt != self.current_negative_prompt:
            self.current_negative_prompt = negative_prompt

    def predict(self, image: Union[Image.Image, np.ndarray]) -> Optional[Image.Image]:
        """Process an image through the pipeline."""
        try:
            # Convert numpy array to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Resize to model dimensions
            image = image.resize((config.width, config.height))

            if self.current_pipeline_type == "diffusers" and self.diffusers_pipe is not None:
                # Use Diffusers pipeline
                model_config = MODELS[self.current_model_key]
                output = self.diffusers_pipe(
                    prompt=self.current_prompt,
                    negative_prompt=self.current_negative_prompt,
                    image=image,
                    num_inference_steps=model_config.num_inference_steps,
                    strength=0.5,  # How much to transform (0=no change, 1=complete regeneration)
                    guidance_scale=0.0,  # CFG scale (0 for 1-step models)
                ).images[0]
                return output

            elif self.stream is not None:
                # Use StreamDiffusion
                image_tensor = self.stream.preprocess_image(image)
                output = self.stream(image=image_tensor)
                return output

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
