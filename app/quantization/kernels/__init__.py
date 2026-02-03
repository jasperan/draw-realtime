"""
Custom CUDA/Triton kernels for 1.58-bit ternary operations.

This module provides optimized kernels for:
- Ternary matrix multiplication with packed 2-bit weights
- Fused unpack + matmul + scale operations
"""

from .ternary_matmul import (
    ternary_matmul,
    ternary_matmul_triton,
    TernaryLinear,
    is_triton_available,
)

__all__ = [
    "ternary_matmul",
    "ternary_matmul_triton",
    "TernaryLinear",
    "is_triton_available",
]
