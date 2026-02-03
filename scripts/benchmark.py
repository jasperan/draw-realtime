#!/usr/bin/env python3
"""
Benchmark script to compare original vs 1.58-bit quantized models.

Measures:
- FPS (frames per second)
- Latency (ms per frame)
- VRAM usage (peak and average)
- Quality metrics (LPIPS, PSNR, SSIM)
- Model size on disk

Usage:
    python scripts/benchmark.py --model sd-turbo --iterations 100
    python scripts/benchmark.py --model sd15-lcm --iterations 100
    python scripts/benchmark.py --all --iterations 50

Output:
    Prints comparison table and saves detailed results to benchmark_results.json
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image

# Optional quality metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. Quality metrics will be limited.")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    quantized: bool
    iterations: int

    # Speed metrics
    fps: float
    latency_ms: float
    latency_std_ms: float

    # Memory metrics
    vram_peak_mb: float
    vram_allocated_mb: float
    model_size_mb: float

    # Quality metrics (only for quantized vs original comparison)
    lpips_score: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None

    # Additional info
    device: str = "cuda"
    resolution: str = "512x512"


def get_vram_usage() -> Dict[str, float]:
    """Get current VRAM usage in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "peak": 0}

    return {
        "allocated": torch.cuda.memory_allocated() / (1024 * 1024),
        "peak": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }


def reset_vram_stats():
    """Reset VRAM peak statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def create_test_images(
    num_images: int = 10,
    resolution: tuple = (512, 512),
    seed: int = 42,
) -> List[Image.Image]:
    """Create test images for benchmarking."""
    np.random.seed(seed)
    images = []

    for i in range(num_images):
        # Create varied test images
        if i % 3 == 0:
            # Gradient image
            arr = np.zeros((*resolution, 3), dtype=np.uint8)
            for c in range(3):
                arr[:, :, c] = np.tile(
                    np.linspace(0, 255, resolution[1], dtype=np.uint8),
                    (resolution[0], 1)
                )
        elif i % 3 == 1:
            # Random noise
            arr = np.random.randint(0, 256, (*resolution, 3), dtype=np.uint8)
        else:
            # Patterns
            arr = np.zeros((*resolution, 3), dtype=np.uint8)
            arr[::8, :, 0] = 255
            arr[:, ::8, 1] = 255

        images.append(Image.fromarray(arr))

    return images


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR between two images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate simplified SSIM between two images."""
    # Simple SSIM approximation
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.std()
    sigma2 = img2.std()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))

    return float(ssim)


def benchmark_pipeline(
    pipeline,
    test_images: List[Image.Image],
    iterations: int,
    warmup: int = 10,
) -> Dict[str, Any]:
    """
    Benchmark a pipeline's inference speed.

    Args:
        pipeline: Pipeline with predict() method
        test_images: List of test images
        iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        Dict with timing statistics
    """
    latencies = []

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for i in range(warmup):
        img = test_images[i % len(test_images)]
        _ = pipeline.predict(img)

    # Benchmark
    print(f"  Running benchmark ({iterations} iterations)...")
    reset_vram_stats()

    for i in range(iterations):
        img = test_images[i % len(test_images)]

        torch.cuda.synchronize()
        start = time.perf_counter()

        _ = pipeline.predict(img)

        torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 20 == 0:
            print(f"    Progress: {i + 1}/{iterations}")

    # Calculate statistics
    latencies = np.array(latencies)
    vram = get_vram_usage()

    return {
        "latencies": latencies.tolist(),
        "latency_mean_ms": float(latencies.mean()),
        "latency_std_ms": float(latencies.std()),
        "latency_min_ms": float(latencies.min()),
        "latency_max_ms": float(latencies.max()),
        "fps": 1000.0 / latencies.mean(),
        "vram_peak_mb": vram["peak"],
        "vram_allocated_mb": vram["allocated"],
    }


def compare_outputs(
    original_pipeline,
    quantized_pipeline,
    test_images: List[Image.Image],
    num_comparisons: int = 10,
) -> Dict[str, float]:
    """
    Compare outputs between original and quantized pipelines.

    Args:
        original_pipeline: Original FP16 pipeline
        quantized_pipeline: Quantized 1.58-bit pipeline
        test_images: Test images
        num_comparisons: Number of image pairs to compare

    Returns:
        Dict with quality metrics
    """
    lpips_scores = []
    psnr_scores = []
    ssim_scores = []

    # Initialize LPIPS if available
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').cuda()

    print(f"  Comparing {num_comparisons} output pairs...")

    for i in range(min(num_comparisons, len(test_images))):
        img = test_images[i]

        # Generate outputs
        original_out = original_pipeline.predict(img)
        quantized_out = quantized_pipeline.predict(img)

        if original_out is None or quantized_out is None:
            continue

        # Convert to numpy
        orig_arr = np.array(original_out)
        quant_arr = np.array(quantized_out)

        # PSNR and SSIM
        psnr_scores.append(calculate_psnr(orig_arr, quant_arr))
        ssim_scores.append(calculate_ssim(orig_arr, quant_arr))

        # LPIPS
        if lpips_fn is not None:
            orig_tensor = torch.from_numpy(orig_arr).permute(2, 0, 1).unsqueeze(0).float().cuda() / 127.5 - 1
            quant_tensor = torch.from_numpy(quant_arr).permute(2, 0, 1).unsqueeze(0).float().cuda() / 127.5 - 1
            with torch.no_grad():
                lpips_scores.append(lpips_fn(orig_tensor, quant_tensor).item())

    results = {
        "psnr_mean": float(np.mean(psnr_scores)) if psnr_scores else None,
        "ssim_mean": float(np.mean(ssim_scores)) if ssim_scores else None,
    }

    if lpips_scores:
        results["lpips_mean"] = float(np.mean(lpips_scores))

    return results


def get_model_size(model_path: str) -> float:
    """Get model size in MB from disk."""
    if os.path.isdir(model_path):
        total = 0
        for f in os.listdir(model_path):
            fp = os.path.join(model_path, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
        return total / (1024 * 1024)
    elif os.path.isfile(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0


def run_benchmark(
    model_key: str,
    iterations: int = 100,
    compare_quality: bool = True,
    device: str = "cuda",
) -> Dict[str, BenchmarkResult]:
    """
    Run full benchmark for a model (original and quantized).

    Args:
        model_key: Model key (sd-turbo or sd15-lcm)
        iterations: Number of benchmark iterations
        compare_quality: Whether to compare quality metrics
        device: Target device

    Returns:
        Dict with 'original' and 'quantized' BenchmarkResult objects
    """
    from app.pipeline import Pipeline
    from app.config import MODELS

    results = {}

    # Create test images
    test_images = create_test_images(num_images=20)

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_key}")
    print(f"{'='*60}")

    # Benchmark original model
    print(f"\n[1/2] Original {model_key} (FP16)")
    print("-" * 40)

    original_pipeline = Pipeline(model_key=model_key, device=device)
    original_stats = benchmark_pipeline(
        original_pipeline,
        test_images,
        iterations=iterations,
    )

    results["original"] = BenchmarkResult(
        model_name=model_key,
        quantized=False,
        iterations=iterations,
        fps=original_stats["fps"],
        latency_ms=original_stats["latency_mean_ms"],
        latency_std_ms=original_stats["latency_std_ms"],
        vram_peak_mb=original_stats["vram_peak_mb"],
        vram_allocated_mb=original_stats["vram_allocated_mb"],
        model_size_mb=0,  # Could calculate from HF cache
        device=device,
    )

    # Benchmark quantized model
    quantized_key = f"{model_key}-1.58bit"
    quantized_path = project_root / "models" / "quantized" / f"{model_key}-1.58bit"

    if not quantized_path.exists():
        print(f"\nWarning: Quantized model not found at {quantized_path}")
        print("Run scripts/quantize_model.py first")
        return results

    print(f"\n[2/2] Quantized {model_key} (1.58-bit)")
    print("-" * 40)

    # Need to add quantized model loading to pipeline first
    # For now, show placeholder
    print("  Note: Quantized pipeline integration pending")
    print("  Run this benchmark after integrating quantized models into pipeline.py")

    return results


def print_comparison_table(results: Dict[str, Dict[str, BenchmarkResult]]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    headers = ["Model", "Type", "FPS", "Latency (ms)", "VRAM Peak (MB)", "Speedup"]
    print(f"\n{headers[0]:<15} {headers[1]:<12} {headers[2]:<8} {headers[3]:<15} {headers[4]:<15} {headers[5]:<10}")
    print("-" * 80)

    for model_key, model_results in results.items():
        original = model_results.get("original")
        quantized = model_results.get("quantized")

        if original:
            speedup = "-"
            print(f"{model_key:<15} {'Original':<12} {original.fps:<8.1f} "
                  f"{original.latency_ms:<15.2f} {original.vram_peak_mb:<15.1f} {speedup:<10}")

        if quantized:
            speedup = f"{quantized.fps / original.fps:.2f}x" if original else "-"
            print(f"{'':<15} {'1.58-bit':<12} {quantized.fps:<8.1f} "
                  f"{quantized.latency_ms:<15.2f} {quantized.vram_peak_mb:<15.1f} {speedup:<10}")

    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark original vs quantized models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["sd-turbo", "sd15-lcm", "all"],
        default="sd-turbo",
        help="Model to benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip quality comparison (faster)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to benchmark on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for detailed results",
    )

    args = parser.parse_args()

    # Determine models to benchmark
    if args.model == "all":
        models = ["sd-turbo", "sd15-lcm"]
    else:
        models = [args.model]

    # Run benchmarks
    all_results = {}
    for model_key in models:
        all_results[model_key] = run_benchmark(
            model_key=model_key,
            iterations=args.iterations,
            compare_quality=not args.no_quality,
            device=args.device,
        )

    # Print comparison table
    print_comparison_table(all_results)

    # Save detailed results
    output_data = {
        model_key: {
            k: asdict(v) for k, v in model_results.items()
        }
        for model_key, model_results in all_results.items()
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
