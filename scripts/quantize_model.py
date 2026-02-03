#!/usr/bin/env python3
"""
One-time PTQ script to convert SD-Turbo and SD15-LCM models to 1.58-bit.

Usage:
    python scripts/quantize_model.py --model sd-turbo --output models/quantized/sd-turbo-1.58bit
    python scripts/quantize_model.py --model sd15-lcm --output models/quantized/sd15-lcm-1.58bit
    python scripts/quantize_model.py --model all  # Quantize both models

This script:
1. Loads the original model using diffusers
2. Extracts the U-Net
3. Quantizes all Linear layers to 1.58-bit ternary
4. Saves the quantized state dict and config
5. Verifies the quantized model with a forward pass
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image

from app.quantization import quantize_unet, save_quantized_model
from app.quantization.utils import get_model_size_mb, verify_quantized_model
from app.config import MODELS


# Model configurations for quantization
QUANTIZABLE_MODELS = {
    "sd-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "description": "SD-Turbo 1.58-bit quantized",
    },
    "sd15-lcm": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "description": "SD 1.5 + LCM-LoRA 1.58-bit quantized",
        "lcm_lora": "latent-consistency/lcm-lora-sdv1-5",
    },
}


def load_unet(model_key: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    """
    Load U-Net from a model configuration.

    Args:
        model_key: Key from QUANTIZABLE_MODELS
        device: Target device
        dtype: Data type for loading

    Returns:
        Tuple of (unet, model_id)
    """
    if model_key not in QUANTIZABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(QUANTIZABLE_MODELS.keys())}")

    config = QUANTIZABLE_MODELS[model_key]
    model_id = config["model_id"]

    print(f"\nLoading model: {model_id}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")

    # Load the full pipeline to get the U-Net
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )

    # Load LCM-LoRA if specified and fuse it
    if "lcm_lora" in config:
        print(f"  Loading LCM-LoRA: {config['lcm_lora']}")
        pipe.load_lora_weights(config["lcm_lora"])
        pipe.fuse_lora()
        print("  LoRA fused into base weights")

    # Extract U-Net and move to device
    unet = pipe.unet.to(device)

    # Clean up the rest of the pipeline
    del pipe.vae
    del pipe.text_encoder
    del pipe.tokenizer
    del pipe.scheduler
    del pipe
    torch.cuda.empty_cache()

    return unet, model_id


def quantize_and_save(
    model_key: str,
    output_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    verify: bool = True,
):
    """
    Load, quantize, and save a model.

    Args:
        model_key: Key from QUANTIZABLE_MODELS
        output_dir: Directory to save quantized model
        device: Target device
        dtype: Data type
        verify: Whether to run verification after quantization
    """
    print(f"\n{'='*60}")
    print(f"Quantizing: {model_key}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    start_time = time.time()

    # Load U-Net
    unet, model_id = load_unet(model_key, device, dtype)

    # Get original size
    original_size = get_model_size_mb(unet)
    print(f"\nOriginal U-Net size: {original_size:.2f} MB")

    # Keep a copy for verification if needed
    if verify:
        print("\nCreating copy for verification...")
        original_state = {k: v.clone() for k, v in unet.state_dict().items()}

    # Quantize
    print("\nQuantizing U-Net...")
    stats = quantize_unet(unet, device=device, verbose=True)

    # Get quantized size
    quantized_size = get_model_size_mb(unet)
    print(f"\nQuantized U-Net size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")

    # Save quantized model
    print(f"\nSaving to {output_dir}...")
    save_quantized_model(
        unet=unet,
        output_dir=output_dir,
        base_model_id=model_id,
        quantized_layers=stats["layer_names"],
        metadata={
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": original_size / quantized_size,
            "source_model_key": model_key,
            "quantization_stats": {
                "num_layers": stats["num_layers"],
                "original_bytes": stats["original_bytes"],
                "quantized_bytes": stats["quantized_bytes"],
            },
        },
    )

    # Verification
    if verify:
        print("\nRunning verification...")
        # Reload original U-Net
        original_unet, _ = load_unet(model_key, device, dtype)

        # Run verification
        results = verify_quantized_model(original_unet, unet, device=device)

        print(f"\nVerification results:")
        print(f"  Success: {results.get('success', False)}")
        if results.get("success"):
            print(f"  MSE: {results['mse']:.6f}")
            print(f"  Max diff: {results['max_diff']:.6f}")
            print(f"  Cosine similarity: {results['cosine_similarity']:.6f}")
        else:
            print(f"  Error: {results.get('error', 'Unknown error')}")

        del original_unet
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Quantize diffusion models to 1.58-bit (BitNet style)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(QUANTIZABLE_MODELS.keys()) + ["all"],
        help="Model to quantize (or 'all' for both)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: models/quantized/<model>-1.58bit)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Data type (default: float16)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step",
    )

    args = parser.parse_args()

    # Parse dtype
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Determine which models to quantize
    if args.model == "all":
        models_to_quantize = list(QUANTIZABLE_MODELS.keys())
    else:
        models_to_quantize = [args.model]

    # Quantize each model
    for model_key in models_to_quantize:
        # Determine output directory
        if args.output and len(models_to_quantize) == 1:
            output_dir = args.output
        else:
            output_dir = str(project_root / "models" / "quantized" / f"{model_key}-1.58bit")

        # Run quantization
        quantize_and_save(
            model_key=model_key,
            output_dir=output_dir,
            device=args.device,
            dtype=dtype,
            verify=not args.no_verify,
        )

    print("\n" + "="*60)
    print("All quantization complete!")
    print("="*60)


if __name__ == "__main__":
    main()
