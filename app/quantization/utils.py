"""
Utility functions for saving and loading quantized models.

This module provides:
- save_quantized_model: Save quantized U-Net state dict and config
- load_quantized_model: Load quantized U-Net and replace layers
- Weight packing utilities (optional, for further compression)
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path

from .bitlinear import BitLinear
from .quantize import replace_linear_with_bitlinear


def save_quantized_model(
    unet: nn.Module,
    output_dir: str,
    base_model_id: str,
    quantized_layers: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a quantized U-Net model to disk.

    Saves:
    - unet_quantized.pt: State dict with ternary weights and scales
    - config.json: Metadata about the quantization

    Args:
        unet: Quantized U-Net model
        output_dir: Directory to save to
        base_model_id: Original model identifier (e.g., "stabilityai/sd-turbo")
        quantized_layers: List of layer names that were quantized
        metadata: Optional additional metadata

    Returns:
        Path to the saved model directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save state dict
    state_dict_path = output_path / "unet_quantized.pt"
    torch.save(unet.state_dict(), state_dict_path)

    # Build config
    config = {
        "base_model_id": base_model_id,
        "quantization_type": "bitnet_1.58bit",
        "quantized_layers": quantized_layers,
        "num_quantized_layers": len(quantized_layers),
        "format_version": "1.0",
    }

    if metadata:
        config["metadata"] = metadata

    # Save config
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved quantized model to {output_path}")
    print(f"  State dict: {state_dict_path}")
    print(f"  Config: {config_path}")

    return str(output_path)


def load_quantized_model(
    unet: nn.Module,
    quantized_dir: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load quantized weights into a U-Net model.

    This function:
    1. Reads the config to get list of quantized layers
    2. Replaces those layers with BitLinear
    3. Loads the quantized state dict

    Args:
        unet: U-Net model to load into (will be modified in-place)
        quantized_dir: Directory containing unet_quantized.pt and config.json
        device: Target device
        strict: Whether to require all keys to match

    Returns:
        Config dict from the quantized model
    """
    quantized_path = Path(quantized_dir)

    # Load config
    config_path = quantized_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Replace Linear layers with BitLinear based on config
    quantized_layers = config.get("quantized_layers", [])
    print(f"Replacing {len(quantized_layers)} layers with BitLinear...")

    for layer_name in quantized_layers:
        try:
            # Navigate to the layer and check if it's Linear
            parts = layer_name.split(".")
            module = unet
            for part in parts:
                module = getattr(module, part)

            if isinstance(module, nn.Linear):
                # Create empty BitLinear (weights will be loaded from state dict)
                bit_linear = BitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    device=device,
                    dtype=module.weight.dtype,
                )

                # Replace in parent
                parent = unet
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], bit_linear)

        except AttributeError as e:
            if strict:
                raise ValueError(f"Could not find layer {layer_name}: {e}")
            else:
                print(f"Warning: Could not find layer {layer_name}, skipping")

    # Load state dict
    state_dict_path = quantized_path / "unet_quantized.pt"
    state_dict = torch.load(state_dict_path, map_location=device)

    # Load with strict=False to handle any minor mismatches
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)

    if missing and strict:
        print(f"Warning: Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
    if unexpected and strict:
        print(f"Warning: Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")

    print(f"Loaded quantized model from {quantized_path}")

    return config


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate the total size of a model's parameters in MB.

    Args:
        model: PyTorch model

    Returns:
        Size in megabytes
    """
    total_bytes = 0

    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()

    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()

    return total_bytes / (1024 * 1024)


def pack_ternary_weights(weight_ternary: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary weights from int8 to 2-bit representation.

    Each ternary value {-1, 0, +1} can be stored in 2 bits:
    - 00 = 0
    - 01 = +1
    - 10 = -1 (or 11, implementation choice)

    This packs 4 values into each byte.

    Args:
        weight_ternary: Int8 tensor with values in {-1, 0, +1}

    Returns:
        Packed uint8 tensor (4x smaller)
    """
    # Flatten for packing
    flat = weight_ternary.flatten()

    # Pad to multiple of 4
    pad_size = (4 - len(flat) % 4) % 4
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size, dtype=torch.int8, device=flat.device)])

    # Map {-1, 0, +1} to {2, 0, 1} for 2-bit encoding
    # -1 -> 10 (binary) = 2
    #  0 -> 00 (binary) = 0
    # +1 -> 01 (binary) = 1
    mapped = flat.clone()
    mapped[flat == -1] = 2
    mapped[flat == 1] = 1
    mapped[flat == 0] = 0

    # Pack 4 values per byte
    packed_len = len(mapped) // 4
    packed = torch.zeros(packed_len, dtype=torch.uint8, device=flat.device)

    for i in range(4):
        packed |= (mapped[i::4].to(torch.uint8) << (i * 2))

    return packed


def unpack_ternary_weights(
    packed: torch.Tensor,
    original_shape: tuple,
) -> torch.Tensor:
    """
    Unpack 2-bit packed weights back to int8 ternary.

    Args:
        packed: Packed uint8 tensor
        original_shape: Shape of the original weight tensor

    Returns:
        Int8 tensor with values in {-1, 0, +1}
    """
    total_elements = 1
    for dim in original_shape:
        total_elements *= dim

    # Unpack 4 values per byte
    unpacked = torch.zeros(len(packed) * 4, dtype=torch.int8, device=packed.device)

    for i in range(4):
        unpacked[i::4] = ((packed >> (i * 2)) & 0b11).to(torch.int8)

    # Map back: {2, 0, 1} -> {-1, 0, +1}
    result = unpacked[:total_elements].clone()
    result[unpacked[:total_elements] == 2] = -1
    result[unpacked[:total_elements] == 1] = 1
    result[unpacked[:total_elements] == 0] = 0

    return result.reshape(original_shape)


def verify_quantized_model(
    original_unet: nn.Module,
    quantized_unet: nn.Module,
    test_input: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Verify that a quantized model produces reasonable outputs.

    Args:
        original_unet: Original FP16 U-Net
        quantized_unet: Quantized U-Net
        test_input: Optional test input tensor
        device: Device to run on

    Returns:
        Dict with verification results
    """
    if test_input is None:
        # Create dummy input matching U-Net expected shape
        # Typical: (batch, channels, height, width) = (1, 4, 64, 64) for latents
        test_input = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)

    # Create dummy timestep
    timestep = torch.tensor([500], device=device)

    # Get encoder hidden size from UNet config
    # SD-Turbo uses cross_attention_dim of 1024, SD 1.5 uses 768
    cross_attention_dim = getattr(original_unet.config, 'cross_attention_dim', 768)
    if isinstance(cross_attention_dim, (list, tuple)):
        cross_attention_dim = cross_attention_dim[-1]  # Use last value if it's a list
    encoder_hidden_states = torch.randn(1, 77, cross_attention_dim, device=device, dtype=torch.float16)

    original_unet.eval()
    quantized_unet.eval()

    with torch.no_grad():
        try:
            # Run both models
            original_output = original_unet(
                test_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            quantized_output = quantized_unet(
                test_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            # Calculate differences
            mse = ((original_output - quantized_output) ** 2).mean().item()
            max_diff = (original_output - quantized_output).abs().max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                original_output.flatten().unsqueeze(0),
                quantized_output.flatten().unsqueeze(0),
            ).item()

            # Check for NaN/Inf
            has_nan = torch.isnan(quantized_output).any().item()
            has_inf = torch.isinf(quantized_output).any().item()

            return {
                "success": not (has_nan or has_inf),
                "mse": mse,
                "max_diff": max_diff,
                "cosine_similarity": cos_sim,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "output_shape": list(quantized_output.shape),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
