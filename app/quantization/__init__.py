"""
BitNet 1.58-bit Post-Training Quantization module for draw-realtime.

This module provides PTQ (Post-Training Quantization) to convert U-Net weights
to 1.58-bit ternary format ({-1, 0, +1}) using absmean scaling.

Key components:
- BitLinear: Drop-in replacement for nn.Linear with int8 ternary weights
- TernaryLinear: Optimized layer with 2-bit packed weights (4x memory reduction)
- quantize_unet: Function to convert a U-Net to use BitLinear layers
- load_quantized_unet: Function to load a pre-quantized U-Net
- pack_for_triton/unpack_for_triton: 2-bit weight packing utilities
"""

from .bitlinear import BitLinear
from .quantize import quantize_unet, replace_linear_with_bitlinear
from .utils import save_quantized_model, load_quantized_model
from .packing import (
    pack_ternary_weights,
    unpack_ternary_weights,
    pack_for_triton,
    unpack_for_triton,
    verify_packing_correctness,
    verify_triton_packing_correctness,
    get_packed_size,
)
from .kernels import TernaryLinear, ternary_matmul, is_triton_available

__all__ = [
    # Core quantization
    "BitLinear",
    "TernaryLinear",
    "quantize_unet",
    "replace_linear_with_bitlinear",
    "save_quantized_model",
    "load_quantized_model",
    # Packing utilities
    "pack_ternary_weights",
    "unpack_ternary_weights",
    "pack_for_triton",
    "unpack_for_triton",
    "verify_packing_correctness",
    "verify_triton_packing_correctness",
    "get_packed_size",
    # Triton/CUDA kernels
    "ternary_matmul",
    "is_triton_available",
]
