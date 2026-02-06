"""
FLUX ControlNet integration for sketch-to-image conditioning.

Supports:
- Canny edge detection
- HED edge detection (softer edges, better for sketches)
- Custom control images

Note: FLUX ControlNets are designed for FLUX.1-dev. Compatibility with
FLUX.2-klein may vary.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union
import cv2


def apply_canny(
    image: Union[Image.Image, np.ndarray],
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
    """Apply Canny edge detection to an image.

    Args:
        image: Input image (PIL or numpy array)
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection

    Returns:
        Grayscale edge image as PIL Image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Convert to 3-channel image for ControlNet
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(edges_rgb)


def apply_hed(
    image: Union[Image.Image, np.ndarray],
) -> Image.Image:
    """Apply HED-like edge detection (approximated with bilateral filter + Canny).

    HED (Holistically-Nested Edge Detection) produces softer edges that are
    better for sketch-like inputs. This is an approximation using OpenCV.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Soft edge image as PIL Image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply bilateral filter for smoothing while preserving edges
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    # Detect edges with lower thresholds for softer output
    edges = cv2.Canny(smooth, 50, 150)

    # Dilate slightly for softer appearance
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Convert to 3-channel
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(edges_rgb)


def preprocess_sketch(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (512, 512),
    invert: bool = True,
) -> Image.Image:
    """Preprocess a sketch image for ControlNet.

    Args:
        image: Input sketch (white lines on dark or dark lines on white)
        target_size: Target resolution (width, height)
        invert: If True, invert colors (assumes dark lines on white background)

    Returns:
        Preprocessed control image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Resize
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')

    # Invert if needed (ControlNet expects white edges on black)
    if invert:
        image = Image.fromarray(255 - np.array(image))

    # Convert back to RGB
    image = image.convert('RGB')

    return image


class FluxControlNetWrapper:
    """Wrapper for FLUX ControlNet inference.

    This class handles loading and running FLUX with ControlNet conditioning.
    """

    def __init__(
        self,
        base_model_id: str = "black-forest-labs/FLUX.1-dev",
        controlnet_id: str = "InstantX/FLUX.1-dev-controlnet-canny",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize FLUX ControlNet pipeline.

        Args:
            base_model_id: HuggingFace model ID for base FLUX model
            controlnet_id: HuggingFace model ID for ControlNet
            device: Device to run on
            dtype: Data type for inference
        """
        from diffusers import FluxControlNetPipeline, FluxControlNetModel

        self.device = device
        self.dtype = dtype

        print(f"Loading ControlNet: {controlnet_id}")
        self.controlnet = FluxControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=dtype,
        )

        print(f"Loading base model: {base_model_id}")
        self.pipe = FluxControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=self.controlnet,
            torch_dtype=dtype,
        ).to(device)

        self.pipe.set_progress_bar_config(disable=True)

    def generate(
        self,
        prompt: str,
        control_image: Image.Image,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        controlnet_conditioning_scale: float = 0.7,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """Generate image with ControlNet conditioning.

        Args:
            prompt: Text prompt for generation
            control_image: Control image (edges, sketch, etc.)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            controlnet_conditioning_scale: How strongly to apply ControlNet
            control_guidance_start: When to start applying ControlNet (0-1)
            control_guidance_end: When to stop applying ControlNet (0-1)
            width: Output width
            height: Output height

        Returns:
            Generated image
        """
        # Resize control image to match output size
        control_image = control_image.resize((width, height), Image.Resampling.LANCZOS)

        output = self.pipe(
            prompt=prompt,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images[0]

        return output
