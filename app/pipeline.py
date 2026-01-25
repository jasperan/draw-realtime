"""StreamDiffusion pipeline wrapper with model switching support."""

import os
import sys
import gc
from typing import Optional, Dict, Union
from PIL import Image
import torch
import numpy as np

# Add StreamDiffusion to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "StreamDiffusion"))

from utils.wrapper import StreamDiffusionWrapper
from app.config import MODELS, DEFAULT_MODEL, DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT, config


class Pipeline:
    """Manages StreamDiffusion pipeline with model switching."""

    def __init__(
        self,
        model_key: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        self.current_model_key: Optional[str] = None
        self.stream: Optional[StreamDiffusionWrapper] = None
        self.current_prompt = DEFAULT_PROMPT
        self.current_negative_prompt = DEFAULT_NEGATIVE_PROMPT

        # Load initial model
        self.load_model(model_key)

    def load_model(self, model_key: str) -> bool:
        """Load a model by its key. Returns True if successful."""
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}")
            return False

        if model_key == self.current_model_key and self.stream is not None:
            print(f"Model {model_key} already loaded")
            return True

        # Clean up existing model
        if self.stream is not None:
            del self.stream
            gc.collect()
            torch.cuda.empty_cache()

        model_config = MODELS[model_key]
        print(f"Loading model: {model_config.description}")

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

        if negative_prompt and negative_prompt != self.current_negative_prompt:
            self.current_negative_prompt = negative_prompt

    def predict(self, image: Union[Image.Image, np.ndarray]) -> Optional[Image.Image]:
        """Process an image through the pipeline."""
        if self.stream is None:
            return None

        try:
            # Convert numpy array to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Resize to model dimensions
            image = image.resize((config.width, config.height))

            # Process
            image_tensor = self.stream.preprocess_image(image)
            output = self.stream(image=image_tensor)

            return output
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
