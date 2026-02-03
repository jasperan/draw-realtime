"""
2-bit weight packing utilities for 1.58-bit ternary quantization.

Packing format:
- Ternary values {-1, 0, +1} are encoded as 2-bit values {0b10, 0b00, 0b01}
- 4 ternary values are packed into each byte
- Bit layout: [v3|v2|v1|v0] where bits 0-1 = v0, bits 2-3 = v1, etc.

Memory savings: 4x reduction (int8 → 2-bit packed)
"""

import torch
from typing import Tuple


# Encoding: ternary value → 2-bit code
# -1 → 0b10 (2)
#  0 → 0b00 (0)
# +1 → 0b01 (1)
TERNARY_TO_CODE = {-1: 2, 0: 0, 1: 1}
CODE_TO_TERNARY = {2: -1, 0: 0, 1: 1}


def pack_ternary_weights(weight_ternary: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Pack ternary weights from int8 to 2-bit representation.

    Each ternary value {-1, 0, +1} is stored in 2 bits, packing 4 values per byte.

    Args:
        weight_ternary: Int8 tensor with values in {-1, 0, +1}

    Returns:
        Tuple of (packed_weights, original_shape):
        - packed_weights: uint8 tensor with 4 values per byte
        - original_shape: original tensor shape for unpacking
    """
    original_shape = weight_ternary.shape
    device = weight_ternary.device

    # Flatten for packing
    flat = weight_ternary.flatten().to(torch.int8)

    # Pad to multiple of 4 if needed
    numel = flat.numel()
    pad_size = (4 - numel % 4) % 4
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size, dtype=torch.int8, device=device)])

    # Map ternary {-1, 0, +1} to 2-bit codes {2, 0, 1}
    # This allows simple bit operations:
    #   -1 → 0b10 (sign bit set, value bit clear)
    #    0 → 0b00 (both bits clear)
    #   +1 → 0b01 (sign bit clear, value bit set)
    codes = torch.zeros_like(flat, dtype=torch.uint8)
    codes[flat == -1] = 2
    codes[flat == 1] = 1
    # codes[flat == 0] remains 0

    # Pack 4 values per byte
    # Reshape to [N/4, 4] for vectorized packing
    codes = codes.reshape(-1, 4)

    # Pack: byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
    packed = (
        codes[:, 0] |
        (codes[:, 1] << 2) |
        (codes[:, 2] << 4) |
        (codes[:, 3] << 6)
    )

    return packed, original_shape


def unpack_ternary_weights(packed: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Unpack 2-bit packed weights back to int8 ternary format.

    Args:
        packed: uint8 tensor with 4 values per byte
        original_shape: original tensor shape

    Returns:
        Int8 tensor with values in {-1, 0, +1}
    """
    device = packed.device

    # Calculate total elements needed
    total_elements = 1
    for dim in original_shape:
        total_elements *= dim

    # Unpack 4 values per byte
    # Extract each 2-bit value using masks and shifts
    v0 = packed & 0b00000011
    v1 = (packed >> 2) & 0b00000011
    v2 = (packed >> 4) & 0b00000011
    v3 = (packed >> 6) & 0b00000011

    # Interleave back to original order
    unpacked = torch.stack([v0, v1, v2, v3], dim=1).flatten()

    # Trim padding and reshape
    unpacked = unpacked[:total_elements]

    # Map codes back to ternary: {2, 0, 1} → {-1, 0, +1}
    result = torch.zeros_like(unpacked, dtype=torch.int8)
    result[unpacked == 2] = -1
    result[unpacked == 1] = 1
    # result[unpacked == 0] remains 0

    return result.reshape(original_shape)


def pack_for_triton(weight_ternary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """
    Pack weights in a format optimized for Triton kernel access.

    For matrix multiplication W @ x where W is [out_features, in_features],
    we want efficient access patterns. This packs along the in_features dimension.

    Args:
        weight_ternary: Int8 tensor of shape [out_features, in_features]

    Returns:
        Tuple of (packed_weights, scale, original_shape):
        - packed_weights: [out_features, in_features // 4] as uint8
        - metadata tensor with padding info
        - original_shape for unpacking
    """
    assert weight_ternary.dim() == 2, "Expected 2D weight tensor"
    out_features, in_features = weight_ternary.shape
    device = weight_ternary.device

    # Pad in_features to multiple of 4
    pad_size = (4 - in_features % 4) % 4
    if pad_size > 0:
        weight_ternary = torch.nn.functional.pad(weight_ternary, (0, pad_size), value=0)

    padded_in_features = weight_ternary.shape[1]

    # Map to codes
    codes = torch.zeros_like(weight_ternary, dtype=torch.uint8)
    codes[weight_ternary == -1] = 2
    codes[weight_ternary == 1] = 1

    # Reshape for packing: [out_features, in_features/4, 4]
    codes = codes.reshape(out_features, padded_in_features // 4, 4)

    # Pack 4 values per byte
    packed = (
        codes[:, :, 0] |
        (codes[:, :, 1] << 2) |
        (codes[:, :, 2] << 4) |
        (codes[:, :, 3] << 6)
    )

    return packed, (out_features, in_features)


def unpack_for_triton(packed: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Unpack weights from Triton-optimized format.

    Args:
        packed: [out_features, in_features // 4] as uint8
        original_shape: (out_features, in_features)

    Returns:
        Int8 tensor of shape [out_features, in_features]
    """
    out_features, in_features = original_shape
    device = packed.device

    # Unpack
    v0 = packed & 0b00000011
    v1 = (packed >> 2) & 0b00000011
    v2 = (packed >> 4) & 0b00000011
    v3 = (packed >> 6) & 0b00000011

    # Stack and reshape: [out_features, packed_in, 4] → [out_features, padded_in]
    unpacked = torch.stack([v0, v1, v2, v3], dim=2).reshape(out_features, -1)

    # Trim padding
    unpacked = unpacked[:, :in_features]

    # Map codes to ternary
    result = torch.zeros_like(unpacked, dtype=torch.int8)
    result[unpacked == 2] = -1
    result[unpacked == 1] = 1

    return result


def get_packed_size(original_shape: Tuple[int, ...]) -> int:
    """Calculate the size of packed tensor in bytes."""
    total_elements = 1
    for dim in original_shape:
        total_elements *= dim
    # 4 elements per byte, rounded up
    return (total_elements + 3) // 4


def verify_packing_correctness(weight_ternary: torch.Tensor) -> bool:
    """
    Verify that pack/unpack round-trip is correct.

    Args:
        weight_ternary: Original ternary weights

    Returns:
        True if round-trip is lossless
    """
    packed, shape = pack_ternary_weights(weight_ternary)
    unpacked = unpack_ternary_weights(packed, shape)
    return torch.equal(weight_ternary, unpacked)


def verify_triton_packing_correctness(weight_ternary: torch.Tensor) -> bool:
    """
    Verify that Triton-format pack/unpack round-trip is correct.

    Args:
        weight_ternary: Original ternary weights [out_features, in_features]

    Returns:
        True if round-trip is lossless
    """
    packed, shape = pack_for_triton(weight_ternary)
    unpacked = unpack_for_triton(packed, shape)
    return torch.equal(weight_ternary, unpacked)
