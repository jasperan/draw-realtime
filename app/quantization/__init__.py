"""
BitNet 1.58-bit Post-Training Quantization module for draw-realtime.

This module provides PTQ (Post-Training Quantization) to convert U-Net weights
to 1.58-bit ternary format ({-1, 0, +1}) using absmean scaling.

Key components:
- BitLinear: Drop-in replacement for nn.Linear with ternary weights
- quantize_unet: Function to convert a U-Net to use BitLinear layers
- load_quantized_unet: Function to load a pre-quantized U-Net
"""

from .bitlinear import BitLinear
from .quantize import quantize_unet, replace_linear_with_bitlinear
from .utils import save_quantized_model, load_quantized_model

__all__ = [
    "BitLinear",
    "quantize_unet",
    "replace_linear_with_bitlinear",
    "save_quantized_model",
    "load_quantized_model",
]
