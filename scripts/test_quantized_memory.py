#!/usr/bin/env python3
"""
Test script to measure memory usage of quantized vs original models.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "StreamDiffusion" / "src"))

import torch
import gc
from PIL import Image
import numpy as np


def measure_memory(model_key: str, num_inferences: int = 5):
    """Measure memory usage for a model."""
    from app.pipeline import Pipeline
    from app.config import PROMPT_PRESETS

    print(f"\n{'='*60}")
    print(f"Measuring memory for: {model_key}")
    print(f"{'='*60}")

    # Clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Measure baseline
    baseline_allocated = torch.cuda.memory_allocated() / (1024**2)
    print(f"Baseline memory: {baseline_allocated:.2f} MB")

    # Load model
    print("Loading model...")
    pipeline = Pipeline(model_key=model_key, device="cuda")
    pipeline.update_prompt(PROMPT_PRESETS["anime-ghibli"].prompt)

    after_load_allocated = torch.cuda.memory_allocated() / (1024**2)
    after_load_peak = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"After load - Allocated: {after_load_allocated:.2f} MB, Peak: {after_load_peak:.2f} MB")

    # Create test image
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

    # Run inferences
    print(f"Running {num_inferences} inferences...")
    torch.cuda.reset_peak_memory_stats()

    for i in range(num_inferences):
        output = pipeline.predict(test_image)
        if i == 0:
            first_inference_peak = torch.cuda.max_memory_allocated() / (1024**2)

    final_allocated = torch.cuda.memory_allocated() / (1024**2)
    final_peak = torch.cuda.max_memory_allocated() / (1024**2)

    print(f"After inference - Allocated: {final_allocated:.2f} MB, Peak: {final_peak:.2f} MB")

    # Cleanup
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_key,
        "model_load_mb": after_load_allocated,
        "inference_peak_mb": final_peak,
        "first_inference_peak_mb": first_inference_peak,
    }


def main():
    results = []

    # Test original model
    results.append(measure_memory("sd-turbo", num_inferences=5))

    # Test quantized model
    results.append(measure_memory("sd-turbo-1.58bit", num_inferences=5))

    # Print comparison
    print("\n" + "="*70)
    print("MEMORY COMPARISON")
    print("="*70)
    print(f"{'Model':<25} {'Model Load (MB)':>18} {'Inference Peak (MB)':>20}")
    print("-"*70)
    for r in results:
        print(f"{r['model']:<25} {r['model_load_mb']:>18.2f} {r['inference_peak_mb']:>20.2f}")

    # Calculate savings
    if len(results) == 2:
        original = results[0]
        quantized = results[1]
        load_savings = (1 - quantized['model_load_mb'] / original['model_load_mb']) * 100
        peak_savings = (1 - quantized['inference_peak_mb'] / original['inference_peak_mb']) * 100
        print("-"*70)
        print(f"{'Memory Savings':<25} {load_savings:>17.1f}% {peak_savings:>19.1f}%")

    print("="*70)


if __name__ == "__main__":
    main()
