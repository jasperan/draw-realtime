"""
Triton kernel for ternary matrix multiplication with packed 2-bit weights.

This implements a fused operation:
    output = (input @ W_ternary.T) * scale + bias

Where W_ternary contains values in {-1, 0, +1} stored as packed 2-bit values.

The kernel:
1. Loads packed weights from global memory
2. Unpacks 4 ternary values per byte
3. Performs matrix multiplication using conditional add/subtract
4. Applies scale factor
5. Optionally adds bias

Performance optimizations:
- Fused operations reduce memory bandwidth
- Tiled computation for cache efficiency
- Vectorized unpacking within warps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


def is_triton_available() -> bool:
    """Check if Triton is available."""
    return TRITON_AVAILABLE


if TRITON_AVAILABLE:
    @triton.jit
    def _ternary_matmul_kernel_tiled(
        # Pointers
        x_ptr,          # Input: [M, K]
        w_packed_ptr,   # Packed weights: [N, K//4]
        scale_ptr,      # Scale: scalar
        bias_ptr,       # Bias: [N] or None
        out_ptr,        # Output: [M, N]
        # Shapes
        M, N, K, K_packed,
        # Strides
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        # Meta
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,  # Process this many K values per iteration (must be multiple of 4)
    ):
        """
        Optimized tiled Triton kernel for ternary matmul.
        Uses vectorized loading and outer products.
        """
        pid = tl.program_id(0)

        # Compute tile indices
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        # Compute block offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Initialize accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Precompute base pointers
        x_base = x_ptr + offs_m * stride_xm
        w_base = w_packed_ptr + offs_n * stride_wn
        m_mask = offs_m < M
        n_mask = offs_n < N

        # Process K dimension - iterate over packed bytes
        for k_packed in range(K_packed):
            k_base = k_packed * 4

            # Load packed weights: [BLOCK_N]
            w_packed_val = tl.load(w_base + k_packed * stride_wk, mask=n_mask, other=0)

            # Unpack all 4 codes at once
            code_0 = w_packed_val & 0x03
            code_1 = (w_packed_val >> 2) & 0x03
            code_2 = (w_packed_val >> 4) & 0x03
            code_3 = (w_packed_val >> 6) & 0x03

            # Convert to ternary (vectorized for BLOCK_N elements)
            w0 = (code_0 & 1).to(tl.float32) - (code_0 >> 1).to(tl.float32)
            w1 = (code_1 & 1).to(tl.float32) - (code_1 >> 1).to(tl.float32)
            w2 = (code_2 & 1).to(tl.float32) - (code_2 >> 1).to(tl.float32)
            w3 = (code_3 & 1).to(tl.float32) - (code_3 >> 1).to(tl.float32)

            # Load x values and compute outer products
            # Using precomputed base to reduce address calculation
            x0 = tl.load(x_base + (k_base + 0) * stride_xk, mask=m_mask & ((k_base + 0) < K), other=0.0).to(tl.float32)
            x1 = tl.load(x_base + (k_base + 1) * stride_xk, mask=m_mask & ((k_base + 1) < K), other=0.0).to(tl.float32)
            x2 = tl.load(x_base + (k_base + 2) * stride_xk, mask=m_mask & ((k_base + 2) < K), other=0.0).to(tl.float32)
            x3 = tl.load(x_base + (k_base + 3) * stride_xk, mask=m_mask & ((k_base + 3) < K), other=0.0).to(tl.float32)

            # Accumulate outer products
            acc += x0[:, None] * w0[None, :]
            acc += x1[:, None] * w1[None, :]
            acc += x2[:, None] * w2[None, :]
            acc += x3[:, None] * w3[None, :]

        # Apply scale
        scale = tl.load(scale_ptr)
        acc = acc * scale

        # Add bias
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
            acc = acc + bias[None, :]

        # Store output
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)

    @triton.jit
    def _ternary_matmul_kernel(
        # Pointers
        x_ptr,          # Input: [M, K]
        w_packed_ptr,   # Packed weights: [N, K//4]
        scale_ptr,      # Scale: [1] or [N]
        bias_ptr,       # Bias: [N] or None
        out_ptr,        # Output: [M, N]
        # Shapes
        M, N, K,
        # Strides
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        # Meta
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Triton kernel for ternary matrix multiplication.

        Computes: out[m, n] = sum_k(x[m, k] * w_ternary[n, k]) * scale + bias[n]

        Where w_ternary values are unpacked from w_packed on-the-fly.
        """
        # Program IDs
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Block starting positions
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Initialize accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K must be processed in chunks of 4 (since 4 values per packed byte)
        # BLOCK_K should be multiple of 4
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)

            # Load input tile: [BLOCK_M, BLOCK_K]
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

            # Load packed weights: [BLOCK_N, BLOCK_K//4]
            # Each byte contains 4 ternary values
            offs_k_packed = (k_start // 4) + tl.arange(0, BLOCK_K // 4)
            w_ptrs = w_packed_ptr + offs_n[:, None] * stride_wn + offs_k_packed[None, :] * stride_wk
            w_mask = (offs_n[:, None] < N) & (offs_k_packed[None, :] < (K + 3) // 4)
            w_packed = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.uint8)

            # Unpack weights: each byte → 4 ternary values
            # Bit layout: [v3|v2|v1|v0] where each vi is 2 bits
            # Extract each 2-bit value and map: 0→0, 1→+1, 2→-1

            # For each packed byte, extract 4 values
            for sub_k in range(4):
                k_idx = k_start + tl.arange(0, BLOCK_K // 4) * 4 + sub_k

                # Extract 2-bit code for this sub-position
                shift = sub_k * 2
                codes = (w_packed >> shift) & 0x03  # [BLOCK_N, BLOCK_K//4]

                # Map codes to ternary: 0→0, 1→+1, 2→-1
                # Using: ternary = (code & 1) - (code >> 1)
                # code=0: (0&1) - (0>>1) = 0 - 0 = 0
                # code=1: (1&1) - (1>>1) = 1 - 0 = 1
                # code=2: (2&1) - (2>>1) = 0 - 1 = -1
                w_ternary = (codes & 1).to(tl.int8) - (codes >> 1).to(tl.int8)
                w_ternary = w_ternary.to(tl.float32)  # [BLOCK_N, BLOCK_K//4]

                # Get corresponding x values
                x_sub_idx = sub_k + tl.arange(0, BLOCK_K // 4) * 4
                x_sub = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + (k_start + x_sub_idx[None, :]) * stride_xk,
                    mask=(offs_m[:, None] < M) & ((k_start + x_sub_idx[None, :]) < K),
                    other=0.0
                ).to(tl.float32)  # [BLOCK_M, BLOCK_K//4]

                # Accumulate: acc[m, n] += x[m, k] * w[n, k]
                acc += tl.dot(x_sub, tl.trans(w_ternary))

        # Load scale and apply
        scale = tl.load(scale_ptr)
        acc = acc * scale

        # Add bias if present
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc = acc + bias[None, :]

        # Store output
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


    @triton.jit
    def _ternary_matmul_kernel_simple(
        # Pointers
        x_ptr,          # Input: [M, K]
        w_packed_ptr,   # Packed weights: [N, K//4]
        scale_ptr,      # Scale: scalar
        bias_ptr,       # Bias: [N] or None
        out_ptr,        # Output: [M, N]
        # Shapes
        M, N, K, K_packed,
        # Strides
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        # Meta
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Simpler Triton kernel - processes one output element per thread block.
        Less optimized but more straightforward.
        """
        # Program IDs - each block computes one BLOCK_M x BLOCK_N tile of output
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Offsets within block
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Initialize accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Process K dimension - unroll the inner loop for 4 values per packed byte
        for k_packed in range(K_packed):
            # Load packed weights for this k position: [BLOCK_N]
            w_ptrs = w_packed_ptr + offs_n * stride_wn + k_packed * stride_wk
            w_mask = offs_n < N
            w_packed_val = tl.load(w_ptrs, mask=w_mask, other=0)

            # Process 4 values from this packed byte
            # Value 0 (bits 0-1)
            k_idx_0 = k_packed * 4 + 0
            code_0 = w_packed_val & 0x03
            w_ternary_0 = (code_0 & 1).to(tl.float32) - (code_0 >> 1).to(tl.float32)
            x_ptrs_0 = x_ptr + offs_m * stride_xm + k_idx_0 * stride_xk
            x_mask_0 = (offs_m < M) & (k_idx_0 < K)
            x_vals_0 = tl.load(x_ptrs_0, mask=x_mask_0, other=0.0).to(tl.float32)
            acc += x_vals_0[:, None] * w_ternary_0[None, :]

            # Value 1 (bits 2-3)
            k_idx_1 = k_packed * 4 + 1
            code_1 = (w_packed_val >> 2) & 0x03
            w_ternary_1 = (code_1 & 1).to(tl.float32) - (code_1 >> 1).to(tl.float32)
            x_ptrs_1 = x_ptr + offs_m * stride_xm + k_idx_1 * stride_xk
            x_mask_1 = (offs_m < M) & (k_idx_1 < K)
            x_vals_1 = tl.load(x_ptrs_1, mask=x_mask_1, other=0.0).to(tl.float32)
            acc += x_vals_1[:, None] * w_ternary_1[None, :]

            # Value 2 (bits 4-5)
            k_idx_2 = k_packed * 4 + 2
            code_2 = (w_packed_val >> 4) & 0x03
            w_ternary_2 = (code_2 & 1).to(tl.float32) - (code_2 >> 1).to(tl.float32)
            x_ptrs_2 = x_ptr + offs_m * stride_xm + k_idx_2 * stride_xk
            x_mask_2 = (offs_m < M) & (k_idx_2 < K)
            x_vals_2 = tl.load(x_ptrs_2, mask=x_mask_2, other=0.0).to(tl.float32)
            acc += x_vals_2[:, None] * w_ternary_2[None, :]

            # Value 3 (bits 6-7)
            k_idx_3 = k_packed * 4 + 3
            code_3 = (w_packed_val >> 6) & 0x03
            w_ternary_3 = (code_3 & 1).to(tl.float32) - (code_3 >> 1).to(tl.float32)
            x_ptrs_3 = x_ptr + offs_m * stride_xm + k_idx_3 * stride_xk
            x_mask_3 = (offs_m < M) & (k_idx_3 < K)
            x_vals_3 = tl.load(x_ptrs_3, mask=x_mask_3, other=0.0).to(tl.float32)
            acc += x_vals_3[:, None] * w_ternary_3[None, :]

        # Apply scale
        scale = tl.load(scale_ptr)
        acc = acc * scale

        # Add bias
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc = acc + bias[None, :]

        # Store
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


def ternary_matmul_triton(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    original_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Ternary matrix multiplication using Triton kernel.

    Computes: output = (x @ W_ternary.T) * scale + bias

    Args:
        x: Input tensor [*, K] or [M, K]
        w_packed: Packed weights [N, K//4] as uint8
        scale: Scale factor (scalar tensor)
        bias: Optional bias [N]
        original_k: Original K dimension (before padding)

    Returns:
        Output tensor [*, N] or [M, N]
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")

    # Handle batched input
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])

    M, K = x.shape
    N, K_packed = w_packed.shape

    # Use original K if provided, otherwise infer from packed
    if original_k is None:
        original_k = K_packed * 4
    K = min(K, original_k)

    # Ensure contiguous
    x = x.contiguous()
    w_packed = w_packed.contiguous()

    # Allocate output
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Kernel config - use larger blocks for better performance
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64  # Must be multiple of 4

    # 1D grid for tiled kernel
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)

    # Launch optimized tiled kernel
    _ternary_matmul_kernel_tiled[grid](
        x, w_packed, scale, bias, out,
        M, N, K, K_packed,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=(bias is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Restore batch dimensions
    if len(orig_shape) > 2:
        out = out.reshape(*orig_shape[:-1], N)

    return out


def ternary_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    original_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Ternary matrix multiplication with automatic backend selection.

    Uses Triton kernel if available, otherwise falls back to PyTorch.

    Args:
        x: Input tensor [*, K]
        w_packed: Packed weights [N, K//4] as uint8
        scale: Scale factor
        bias: Optional bias [N]
        original_k: Original K dimension

    Returns:
        Output tensor [*, N]
    """
    if TRITON_AVAILABLE and x.is_cuda:
        return ternary_matmul_triton(x, w_packed, scale, bias, original_k)
    else:
        return ternary_matmul_pytorch(x, w_packed, scale, bias, original_k)


def ternary_matmul_pytorch(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    original_k: Optional[int] = None,
) -> torch.Tensor:
    """
    PyTorch fallback for ternary matmul.

    Unpacks weights to int8, then uses standard matmul.
    """
    from ..packing import unpack_for_triton

    N, K_packed = w_packed.shape
    K = original_k if original_k is not None else K_packed * 4

    # Unpack weights
    w_ternary = unpack_for_triton(w_packed, (N, K))

    # Convert to float and matmul
    w_float = w_ternary.to(x.dtype)
    out = F.linear(x, w_float, None)

    # Apply scale
    out = out * scale

    # Add bias
    if bias is not None:
        out = out + bias

    return out


class TernaryLinear(nn.Module):
    """
    Linear layer with packed 2-bit ternary weights.

    This is an optimized replacement for BitLinear that:
    1. Stores weights in packed 2-bit format (4x memory reduction for storage)
    2. Optionally caches unpacked weights for fast inference (use_triton=False)
    3. Can use Triton kernel for memory-efficient inference (use_triton=True)

    The default mode (use_triton=False) unpacks weights once and caches them,
    trading memory for speed. This is recommended for inference.

    Attributes:
        in_features: Input feature dimension
        out_features: Output feature dimension
        weight_packed: Packed weights [out_features, in_features//4] as uint8
        weight_scale: Scale factor for dequantization
        bias: Optional bias term
        use_triton: Whether to use Triton kernel (slower but memory-efficient)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_triton: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.original_in_features = in_features  # Before padding
        self.use_triton = use_triton
        self._cached_weight = None

        # Pad in_features to multiple of 4
        padded_in = (in_features + 3) // 4 * 4

        # Packed weights: [out_features, padded_in // 4]
        self.register_buffer(
            "weight_packed",
            torch.zeros(out_features, padded_in // 4, dtype=torch.uint8, device=device)
        )

        # Scale factor
        self.register_buffer(
            "weight_scale",
            torch.ones(1, dtype=dtype or torch.float16, device=device)
        )

        # Bias
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype or torch.float16, device=device)
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_bitlinear(
        cls,
        bitlinear,
        device: Optional[torch.device] = None,
        use_triton: bool = False,
        cache_weights: bool = True,
    ) -> "TernaryLinear":
        """
        Convert a BitLinear layer to TernaryLinear (packed).

        Args:
            bitlinear: BitLinear layer with int8 ternary weights
            device: Target device
            use_triton: Whether to use Triton kernel (slower but memory-efficient)
            cache_weights: Whether to cache unpacked weights (faster inference)

        Returns:
            TernaryLinear with packed weights
        """
        from ..packing import pack_for_triton

        device = device or bitlinear.weight_ternary.device
        dtype = bitlinear.weight_scale.dtype

        # Create TernaryLinear
        layer = cls(
            in_features=bitlinear.in_features,
            out_features=bitlinear.out_features,
            bias=bitlinear.bias is not None,
            device=device,
            dtype=dtype,
            use_triton=use_triton,
        )

        # Pack weights
        packed, _ = pack_for_triton(bitlinear.weight_ternary)
        layer.weight_packed = packed.to(device)
        layer.weight_scale = bitlinear.weight_scale.to(device)
        layer.original_in_features = bitlinear.in_features

        # Copy bias
        if bitlinear.bias is not None:
            layer.bias = nn.Parameter(bitlinear.bias.to(device))

        # Cache weights for fast inference
        if cache_weights and not use_triton:
            layer.cache_weights()

        return layer

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        device: Optional[torch.device] = None,
        use_triton: bool = False,
        cache_weights: bool = True,
    ) -> "TernaryLinear":
        """
        Create TernaryLinear from nn.Linear via quantization.

        Args:
            linear: Source nn.Linear layer
            device: Target device
            use_triton: Whether to use Triton kernel (slower but memory-efficient)
            cache_weights: Whether to cache unpacked weights (faster inference)

        Returns:
            Quantized TernaryLinear layer
        """
        from ..packing import pack_for_triton

        device = device or linear.weight.device
        dtype = linear.weight.dtype

        # Create layer
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=device,
            dtype=dtype,
            use_triton=use_triton,
        )

        # Quantize weights
        with torch.no_grad():
            weight = linear.weight.float()

            # Absmean scaling
            scale = weight.abs().mean()
            if scale < 1e-8:
                scale = torch.tensor(1e-8, device=device)

            # Ternarize
            weight_norm = weight / scale
            weight_ternary = weight_norm.round().clamp(-1, 1).to(torch.int8)

            # Pack
            packed, _ = pack_for_triton(weight_ternary)

            layer.weight_packed = packed.to(device)
            layer.weight_scale = scale.to(dtype).unsqueeze(0).to(device)
            layer.original_in_features = linear.in_features

            # Copy bias
            if linear.bias is not None:
                layer.bias = nn.Parameter(linear.bias.clone().to(device))

        # Cache weights for fast inference
        if cache_weights and not use_triton:
            layer.cache_weights()

        return layer

    def _unpack_weights(self) -> torch.Tensor:
        """Unpack weights from 2-bit to float16."""
        from ..packing import unpack_for_triton

        w_ternary = unpack_for_triton(
            self.weight_packed,
            (self.out_features, self.original_in_features)
        )
        return w_ternary.to(self.weight_scale.dtype)

    def _get_weight(self) -> torch.Tensor:
        """Get unpacked and scaled weight, using cache if available."""
        if self._cached_weight is not None:
            return self._cached_weight

        w_float = self._unpack_weights()
        return w_float * self.weight_scale

    def cache_weights(self):
        """
        Cache unpacked weights for fast inference.

        Call this after loading the model to trade memory for speed.
        The cached weights use ~4x more memory than packed, but inference
        is much faster (uses optimized cuBLAS instead of custom kernel).
        """
        if self._cached_weight is None:
            self._cached_weight = self._get_weight()
        return self

    def clear_cache(self):
        """Clear cached weights to free memory."""
        self._cached_weight = None
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        If use_triton=True, uses Triton kernel for memory-efficient inference.
        Otherwise, uses cached/unpacked weights with optimized cuBLAS matmul.

        Args:
            x: Input tensor [*, in_features]

        Returns:
            Output tensor [*, out_features]
        """
        if self.use_triton and TRITON_AVAILABLE and x.is_cuda:
            # Memory-efficient path using Triton kernel
            return ternary_matmul(
                x,
                self.weight_packed,
                self.weight_scale,
                self.bias,
                self.original_in_features,
            )
        else:
            # Fast path using cached weights and cuBLAS
            weight = self._get_weight()
            out = F.linear(x, weight, self.bias)
            return out

    def extra_repr(self) -> str:
        mode = "triton" if self.use_triton else "cached" if self._cached_weight is not None else "lazy"
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"packed=2-bit, mode={mode}"
        )

    def get_memory_savings(self) -> dict:
        """
        Calculate memory usage compared to FP16 Linear.

        Returns dict with:
        - original_bytes: FP16 Linear memory
        - packed_bytes: Packed storage size (what's saved to disk)
        - runtime_bytes: Actual memory used at runtime
        - storage_ratio: Storage savings (packed vs original)
        - runtime_ratio: Runtime savings (runtime vs original)
        """
        # Original FP16
        original_weight_bytes = self.in_features * self.out_features * 2
        original_bias_bytes = self.out_features * 2 if self.bias is not None else 0
        original_bytes = original_weight_bytes + original_bias_bytes

        # Packed storage
        packed_weight_bytes = self.weight_packed.numel() * 1  # uint8
        packed_scale_bytes = 2
        packed_bias_bytes = self.out_features * 2 if self.bias is not None else 0
        packed_bytes = packed_weight_bytes + packed_scale_bytes + packed_bias_bytes

        # Runtime memory (includes cache if present)
        cache_bytes = 0
        if self._cached_weight is not None:
            cache_bytes = self._cached_weight.numel() * self._cached_weight.element_size()
        runtime_bytes = packed_bytes + cache_bytes

        storage_ratio = original_bytes / packed_bytes if packed_bytes > 0 else 0
        return {
            "original_bytes": original_bytes,
            "packed_bytes": packed_bytes,
            "runtime_bytes": runtime_bytes,
            "storage_ratio": storage_ratio,
            "runtime_ratio": original_bytes / runtime_bytes if runtime_bytes > 0 else 0,
            "savings_ratio": storage_ratio,  # Backward compatibility alias
        }
