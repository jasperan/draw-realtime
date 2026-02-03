#!/usr/bin/env python3
"""
Unit tests for 2-bit weight packing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from app.quantization.packing import (
    pack_ternary_weights,
    unpack_ternary_weights,
    pack_for_triton,
    unpack_for_triton,
    verify_packing_correctness,
    verify_triton_packing_correctness,
    get_packed_size,
)


def test_basic_packing():
    """Test basic pack/unpack round-trip."""
    print("Test: Basic packing round-trip")

    # Create test tensor with all ternary values
    weight = torch.tensor([-1, 0, 1, -1, 0, 1, 1, 0], dtype=torch.int8)

    packed, shape = pack_ternary_weights(weight)
    unpacked = unpack_ternary_weights(packed, shape)

    assert torch.equal(weight, unpacked), f"Mismatch: {weight} vs {unpacked}"
    assert packed.shape[0] == 2, f"Expected 2 bytes, got {packed.shape[0]}"

    print(f"  Original: {weight.tolist()}")
    print(f"  Packed:   {packed.tolist()} ({packed.shape[0]} bytes)")
    print(f"  Unpacked: {unpacked.tolist()}")
    print("  PASSED\n")


def test_odd_size():
    """Test packing with non-multiple-of-4 sizes."""
    print("Test: Odd size packing")

    for size in [1, 3, 5, 7, 13, 17, 100, 127]:
        weight = torch.randint(-1, 2, (size,), dtype=torch.int8)
        packed, shape = pack_ternary_weights(weight)
        unpacked = unpack_ternary_weights(packed, shape)

        assert torch.equal(weight, unpacked), f"Mismatch for size {size}"

    print(f"  Tested sizes: 1, 3, 5, 7, 13, 17, 100, 127")
    print("  PASSED\n")


def test_2d_packing():
    """Test packing 2D tensors."""
    print("Test: 2D tensor packing")

    weight = torch.tensor([
        [-1, 0, 1, -1],
        [1, 1, 0, 0],
        [0, -1, -1, 1],
    ], dtype=torch.int8)

    packed, shape = pack_ternary_weights(weight)
    unpacked = unpack_ternary_weights(packed, shape)

    assert torch.equal(weight, unpacked), f"Mismatch"
    print(f"  Shape: {weight.shape} → packed {packed.shape} → {unpacked.shape}")
    print("  PASSED\n")


def test_triton_format():
    """Test Triton-optimized packing format."""
    print("Test: Triton format packing")

    # Typical weight matrix shape
    weight = torch.randint(-1, 2, (128, 256), dtype=torch.int8)

    packed, shape = pack_for_triton(weight)
    unpacked = unpack_for_triton(packed, shape)

    assert torch.equal(weight, unpacked), "Mismatch"
    assert packed.shape == (128, 64), f"Expected (128, 64), got {packed.shape}"

    print(f"  Shape: {weight.shape} → packed {packed.shape}")
    print(f"  Memory: {weight.numel()} bytes → {packed.numel()} bytes (4x reduction)")
    print("  PASSED\n")


def test_triton_format_odd_size():
    """Test Triton format with non-aligned sizes."""
    print("Test: Triton format odd sizes")

    for out_f, in_f in [(64, 100), (100, 64), (33, 77), (128, 513)]:
        weight = torch.randint(-1, 2, (out_f, in_f), dtype=torch.int8)
        packed, shape = pack_for_triton(weight)
        unpacked = unpack_for_triton(packed, shape)

        assert torch.equal(weight, unpacked), f"Mismatch for ({out_f}, {in_f})"

    print("  Tested shapes: (64,100), (100,64), (33,77), (128,513)")
    print("  PASSED\n")


def test_memory_savings():
    """Verify memory savings from packing."""
    print("Test: Memory savings calculation")

    # Simulate SD-Turbo U-Net size (~865M weights)
    weight = torch.randint(-1, 2, (1024, 1024), dtype=torch.int8)

    original_bytes = weight.numel() * weight.element_size()
    packed, _ = pack_ternary_weights(weight)
    packed_bytes = packed.numel() * packed.element_size()

    savings = original_bytes / packed_bytes

    print(f"  Original: {original_bytes:,} bytes")
    print(f"  Packed:   {packed_bytes:,} bytes")
    print(f"  Savings:  {savings:.1f}x")

    assert savings >= 3.9, f"Expected ~4x savings, got {savings}x"
    print("  PASSED\n")


def test_gpu_packing():
    """Test packing on GPU."""
    print("Test: GPU packing")

    if not torch.cuda.is_available():
        print("  SKIPPED (no GPU)\n")
        return

    weight = torch.randint(-1, 2, (512, 512), dtype=torch.int8, device="cuda")

    # Test general packing
    packed, shape = pack_ternary_weights(weight)
    unpacked = unpack_ternary_weights(packed, shape)
    assert torch.equal(weight, unpacked), "GPU general packing mismatch"

    # Test Triton format
    packed, shape = pack_for_triton(weight)
    unpacked = unpack_for_triton(packed, shape)
    assert torch.equal(weight, unpacked), "GPU Triton format mismatch"

    print(f"  GPU packing verified on {torch.cuda.get_device_name()}")
    print("  PASSED\n")


def test_all_value_patterns():
    """Test all possible 4-value patterns."""
    print("Test: All value patterns")

    # Generate all 81 possible patterns of 4 ternary values (3^4 = 81)
    patterns_tested = 0
    for v0 in [-1, 0, 1]:
        for v1 in [-1, 0, 1]:
            for v2 in [-1, 0, 1]:
                for v3 in [-1, 0, 1]:
                    weight = torch.tensor([v0, v1, v2, v3], dtype=torch.int8)
                    packed, shape = pack_ternary_weights(weight)
                    unpacked = unpack_ternary_weights(packed, shape)

                    assert torch.equal(weight, unpacked), f"Failed for pattern [{v0},{v1},{v2},{v3}]"
                    patterns_tested += 1

    print(f"  Tested {patterns_tested} patterns (all 3^4 combinations)")
    print("  PASSED\n")


def run_all_tests():
    """Run all packing tests."""
    print("=" * 60)
    print("2-bit Packing Unit Tests")
    print("=" * 60 + "\n")

    test_basic_packing()
    test_odd_size()
    test_2d_packing()
    test_triton_format()
    test_triton_format_odd_size()
    test_memory_savings()
    test_gpu_packing()
    test_all_value_patterns()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
