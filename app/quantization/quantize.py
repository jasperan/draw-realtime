"""
Quantization functions for converting U-Net models to 1.58-bit.

This module provides functions to:
- Find all nn.Linear layers in a U-Net
- Replace them with BitLinear equivalents
- Quantize an entire U-Net in-place or create a quantized copy
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Set, Optional, Dict, Any
from .bitlinear import BitLinear


def find_linear_layers(
    model: nn.Module,
    prefix: str = "",
    skip_patterns: Optional[Set[str]] = None,
) -> List[Tuple[str, nn.Linear]]:
    """
    Recursively find all nn.Linear layers in a model.

    Args:
        model: PyTorch model to search
        prefix: Current path prefix for naming
        skip_patterns: Set of name patterns to skip (e.g., {"lm_head", "embed"})

    Returns:
        List of (name, layer) tuples for all Linear layers
    """
    skip_patterns = skip_patterns or set()
    linear_layers = []

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Check if this layer should be skipped
        should_skip = any(pattern in full_name for pattern in skip_patterns)

        if isinstance(module, nn.Linear) and not should_skip:
            linear_layers.append((full_name, module))
        else:
            # Recurse into children
            linear_layers.extend(
                find_linear_layers(module, full_name, skip_patterns)
            )

    return linear_layers


def replace_linear_with_bitlinear(
    model: nn.Module,
    layer_name: str,
    device: Optional[torch.device] = None,
) -> BitLinear:
    """
    Replace a single nn.Linear layer with BitLinear in-place.

    Args:
        model: Model containing the layer
        layer_name: Dot-separated path to the layer (e.g., "down_blocks.0.attentions.0.to_q")
        device: Target device for the new layer

    Returns:
        The new BitLinear layer
    """
    # Navigate to parent module
    parts = layer_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    # Get the original layer
    original_layer = getattr(parent, parts[-1])

    if not isinstance(original_layer, nn.Linear):
        raise ValueError(f"Layer {layer_name} is not nn.Linear, got {type(original_layer)}")

    # Create BitLinear replacement
    bit_linear = BitLinear.from_linear(original_layer, device=device)

    # Replace in parent
    setattr(parent, parts[-1], bit_linear)

    return bit_linear


def quantize_unet(
    unet: nn.Module,
    skip_patterns: Optional[Set[str]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Quantize all Linear layers in a U-Net to 1.58-bit.

    This function:
    1. Finds all nn.Linear layers in the U-Net
    2. Replaces each with a BitLinear equivalent
    3. Returns statistics about the quantization

    Args:
        unet: The U-Net model to quantize (modified in-place)
        skip_patterns: Layer name patterns to skip (default: none)
        device: Target device for quantized layers
        verbose: Whether to print progress

    Returns:
        Dict with quantization statistics:
        - num_layers: Number of layers quantized
        - layer_names: List of quantized layer names
        - original_bytes: Total original weight memory
        - quantized_bytes: Total quantized weight memory
        - savings_ratio: Memory savings ratio
    """
    # Default skip patterns - typically we want to quantize everything in U-Net
    # but keep certain projection layers if needed
    skip_patterns = skip_patterns or set()

    # Find all Linear layers
    linear_layers = find_linear_layers(unet, skip_patterns=skip_patterns)

    if verbose:
        print(f"Found {len(linear_layers)} Linear layers to quantize")

    # Track statistics
    total_original_bytes = 0
    total_quantized_bytes = 0
    quantized_layer_names = []

    # Replace each layer
    for i, (name, layer) in enumerate(linear_layers):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Quantizing layer {i + 1}/{len(linear_layers)}: {name}")

        # Replace with BitLinear
        bit_linear = replace_linear_with_bitlinear(unet, name, device=device)
        quantized_layer_names.append(name)

        # Track memory
        savings = bit_linear.get_memory_savings()
        total_original_bytes += savings["original_bytes"]
        total_quantized_bytes += savings["quantized_bytes"]

    # Calculate overall savings
    savings_ratio = total_original_bytes / total_quantized_bytes if total_quantized_bytes > 0 else 0

    stats = {
        "num_layers": len(quantized_layer_names),
        "layer_names": quantized_layer_names,
        "original_bytes": total_original_bytes,
        "quantized_bytes": total_quantized_bytes,
        "savings_ratio": savings_ratio,
        "original_mb": total_original_bytes / (1024 * 1024),
        "quantized_mb": total_quantized_bytes / (1024 * 1024),
    }

    if verbose:
        print(f"\nQuantization complete:")
        print(f"  Layers quantized: {stats['num_layers']}")
        print(f"  Original size: {stats['original_mb']:.2f} MB")
        print(f"  Quantized size: {stats['quantized_mb']:.2f} MB")
        print(f"  Savings ratio: {stats['savings_ratio']:.2f}x")

    return stats


def get_unet_from_pipeline(pipeline) -> nn.Module:
    """
    Extract U-Net from various pipeline types.

    Args:
        pipeline: A diffusion pipeline (StreamDiffusion, Diffusers, etc.)

    Returns:
        The U-Net module
    """
    # StreamDiffusion wrapper
    if hasattr(pipeline, "stream") and hasattr(pipeline.stream, "unet"):
        return pipeline.stream.unet

    # StreamDiffusion direct
    if hasattr(pipeline, "unet"):
        return pipeline.unet

    # Diffusers pipelines
    if hasattr(pipeline, "pipe") and hasattr(pipeline.pipe, "unet"):
        return pipeline.pipe.unet

    raise ValueError(f"Could not find U-Net in pipeline of type {type(pipeline)}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in a model, separating quantized and non-quantized.

    Args:
        model: PyTorch model

    Returns:
        Dict with total, quantized, and fp16 parameter counts
    """
    total_params = 0
    quantized_params = 0
    fp_params = 0

    for module in model.modules():
        if isinstance(module, BitLinear):
            # Count ternary weights
            quantized_params += module.weight_ternary.numel()
            if module.bias is not None:
                fp_params += module.bias.numel()
        elif isinstance(module, nn.Linear):
            fp_params += module.weight.numel()
            if module.bias is not None:
                fp_params += module.bias.numel()

    total_params = quantized_params + fp_params

    return {
        "total": total_params,
        "quantized": quantized_params,
        "fp16": fp_params,
        "quantized_ratio": quantized_params / total_params if total_params > 0 else 0,
    }
