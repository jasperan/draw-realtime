#!/usr/bin/env python3
"""
Test script for Triton ternary matmul kernel.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import time

from app.quantization.packing import pack_for_triton
from app.quantization.kernels.ternary_matmul import (
    ternary_matmul,
    ternary_matmul_pytorch,
    TernaryLinear,
    is_triton_available,
)


def test_correctness():
    """Test that Triton kernel produces correct results."""
    print("Test: Kernel correctness")

    if not torch.cuda.is_available():
        print("  SKIPPED (no GPU)\n")
        return

    # Create test data
    M, K, N = 32, 128, 64
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")

    # Create ternary weights
    w_ternary = torch.randint(-1, 2, (N, K), dtype=torch.int8, device="cuda")
    scale = torch.tensor([0.1], dtype=torch.float16, device="cuda")

    # Pack weights
    w_packed, _ = pack_for_triton(w_ternary)

    # Compute reference (PyTorch)
    w_float = w_ternary.to(torch.float16)
    ref = F.linear(x, w_float, None) * scale

    # Compute with ternary_matmul
    out = ternary_matmul(x, w_packed, scale, original_k=K)

    # Compare
    max_diff = (ref - out).abs().max().item()
    mean_diff = (ref - out).abs().mean().item()

    print(f"  Shape: x={x.shape}, w_packed={w_packed.shape}, out={out.shape}")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    # Allow small numerical differences
    assert max_diff < 1e-2, f"Max diff too large: {max_diff}"
    print("  PASSED\n")


def test_correctness_with_bias():
    """Test kernel with bias."""
    print("Test: Kernel with bias")

    if not torch.cuda.is_available():
        print("  SKIPPED (no GPU)\n")
        return

    M, K, N = 32, 128, 64
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w_ternary = torch.randint(-1, 2, (N, K), dtype=torch.int8, device="cuda")
    scale = torch.tensor([0.1], dtype=torch.float16, device="cuda")
    bias = torch.randn(N, dtype=torch.float16, device="cuda")

    w_packed, _ = pack_for_triton(w_ternary)

    # Reference
    w_float = w_ternary.to(torch.float16)
    ref = F.linear(x, w_float, None) * scale + bias

    # Triton
    out = ternary_matmul(x, w_packed, scale, bias=bias, original_k=K)

    max_diff = (ref - out).abs().max().item()
    assert max_diff < 1e-2, f"Max diff too large: {max_diff}"
    print(f"  Max difference: {max_diff:.6f}")
    print("  PASSED\n")


def test_batched_input():
    """Test with batched input."""
    print("Test: Batched input")

    if not torch.cuda.is_available():
        print("  SKIPPED (no GPU)\n")
        return

    B, M, K, N = 4, 32, 128, 64
    x = torch.randn(B, M, K, dtype=torch.float16, device="cuda")
    w_ternary = torch.randint(-1, 2, (N, K), dtype=torch.int8, device="cuda")
    scale = torch.tensor([0.1], dtype=torch.float16, device="cuda")

    w_packed, _ = pack_for_triton(w_ternary)

    # Reference
    w_float = w_ternary.to(torch.float16)
    ref = F.linear(x, w_float, None) * scale

    # Triton
    out = ternary_matmul(x, w_packed, scale, original_k=K)

    max_diff = (ref - out).abs().max().item()
    assert max_diff < 1e-2, f"Max diff too large: {max_diff}"
    print(f"  Shape: {x.shape} → {out.shape}")
    print(f"  Max difference: {max_diff:.6f}")
    print("  PASSED\n")


def test_ternary_linear():
    """Test TernaryLinear layer."""
    print("Test: TernaryLinear layer")

    if not torch.cuda.is_available():
        print("  SKIPPED (no GPU)\n")
        return

    # Create original linear
    linear = torch.nn.Linear(128, 64, bias=True).cuda().half()

    # Test both modes
    for use_triton in [False, True]:
        mode_name = "Triton" if use_triton else "Cached"

        # Convert to TernaryLinear
        ternary = TernaryLinear.from_linear(
            linear,
            use_triton=use_triton,
            cache_weights=not use_triton
        )

        # Test forward
        x = torch.randn(32, 128, dtype=torch.float16, device="cuda")
        out = ternary(x)

        assert out.shape == (32, 64), f"Wrong shape: {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

        # Check memory savings
        savings = ternary.get_memory_savings()
        print(f"  [{mode_name}] Output shape: {out.shape}")
        print(f"  [{mode_name}] Storage savings: {savings['storage_ratio']:.2f}x")
        print(f"  [{mode_name}] Runtime savings: {savings['runtime_ratio']:.2f}x")

    print("  PASSED\n")


def test_performance():
    """Benchmark all modes vs PyTorch."""
    print("Test: Performance benchmark")

    if not torch.cuda.is_available():
        print("  SKIPPED (no GPU)\n")
        return

    # Typical sizes for SD U-Net attention
    M, K, N = 4096, 320, 320
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w_ternary = torch.randint(-1, 2, (N, K), dtype=torch.int8, device="cuda")
    scale = torch.tensor([0.1], dtype=torch.float16, device="cuda")

    w_packed, _ = pack_for_triton(w_ternary)
    w_float = w_ternary.to(torch.float16)

    # Create TernaryLinear layers for both modes
    linear_fp16 = torch.nn.Linear(K, N, bias=False).cuda().half()
    ternary_cached = TernaryLinear.from_linear(linear_fp16, use_triton=False, cache_weights=True)
    ternary_triton = TernaryLinear.from_linear(linear_fp16, use_triton=True, cache_weights=False) if is_triton_available() else None

    iterations = 100

    # Warmup
    for _ in range(10):
        _ = ternary_cached(x)
        if ternary_triton:
            _ = ternary_triton(x)
        _ = F.linear(x, w_float, None) * scale
    torch.cuda.synchronize()

    # Benchmark TernaryLinear (cached mode)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = ternary_cached(x)
    torch.cuda.synchronize()
    cached_time = (time.perf_counter() - start) / iterations * 1000

    # Benchmark TernaryLinear (Triton mode)
    triton_time = float('inf')
    if ternary_triton and is_triton_available():
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            _ = ternary_triton(x)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / iterations * 1000

    # Benchmark PyTorch (int8 → float16 conversion)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        w_float = w_ternary.to(torch.float16)
        _ = F.linear(x, w_float, None) * scale
    torch.cuda.synchronize()
    pytorch_int8_time = (time.perf_counter() - start) / iterations * 1000

    # Benchmark PyTorch (float16 weights, no conversion)
    w_fp16 = torch.randn(N, K, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = F.linear(x, w_fp16, None) * scale
    torch.cuda.synchronize()
    pytorch_fp16_time = (time.perf_counter() - start) / iterations * 1000

    print(f"  Matrix size: [{M}, {K}] @ [{N}, {K}]")
    print(f"  TernaryLinear (cached): {cached_time:.3f} ms")
    if is_triton_available():
        print(f"  TernaryLinear (Triton): {triton_time:.3f} ms")
    print(f"  PyTorch (int8→fp16):    {pytorch_int8_time:.3f} ms")
    print(f"  PyTorch (fp16):         {pytorch_fp16_time:.3f} ms")
    print()
    print(f"  Cached vs int8:  {pytorch_int8_time / cached_time:.2f}x")
    print(f"  Cached vs fp16:  {pytorch_fp16_time / cached_time:.2f}x")
    if is_triton_available():
        print(f"  Triton vs int8:  {pytorch_int8_time / triton_time:.2f}x")
        print(f"  Triton vs fp16:  {pytorch_fp16_time / triton_time:.2f}x")
    print("  PASSED\n")


def test_various_sizes():
    """Test kernel with various matrix sizes."""
    print("Test: Various sizes")

    if not torch.cuda.is_available():
        print("  SKIPPED (no GPU)\n")
        return

    sizes = [
        (32, 64, 64),
        (64, 128, 128),
        (128, 256, 256),
        (256, 512, 512),
        (512, 1024, 1024),
        (33, 65, 129),  # Non-aligned
        (1, 320, 320),   # Single row
    ]

    for M, K, N in sizes:
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        w_ternary = torch.randint(-1, 2, (N, K), dtype=torch.int8, device="cuda")
        scale = torch.tensor([0.1], dtype=torch.float16, device="cuda")

        w_packed, _ = pack_for_triton(w_ternary)

        # Reference
        w_float = w_ternary.to(torch.float16)
        ref = F.linear(x, w_float, None) * scale

        # Triton
        out = ternary_matmul(x, w_packed, scale, original_k=K)

        max_diff = (ref - out).abs().max().item()
        assert max_diff < 1e-2, f"Size ({M},{K},{N}): max diff {max_diff}"

    print(f"  Tested {len(sizes)} size combinations")
    print("  PASSED\n")


def run_all_tests():
    """Run all kernel tests."""
    print("=" * 60)
    print("Triton Kernel Tests")
    print(f"Triton available: {is_triton_available()}")
    print("=" * 60 + "\n")

    test_correctness()
    test_correctness_with_bias()
    test_batched_input()
    test_ternary_linear()
    test_various_sizes()
    test_performance()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
