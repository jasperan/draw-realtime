#!/usr/bin/env python3
"""
Test TernaryLinear layer on full quantized SD-Turbo model.
Compares BitLinear (current) vs TernaryLinear (new optimized) performance.
"""

import os
import sys
import time
import gc
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import cv2
import numpy as np
from PIL import Image

from app.pipeline import Pipeline
from app.config import MODELS, PROMPT_PRESETS
from app.quantization import BitLinear, TernaryLinear


def convert_bitlinear_to_ternary(model, use_triton=False, cache_weights=True):
    """
    Convert all BitLinear layers in a model to TernaryLinear.

    Args:
        model: Model containing BitLinear layers
        use_triton: Whether to use Triton kernel (slower but memory-efficient)
        cache_weights: Whether to cache unpacked weights (faster inference)

    Returns:
        Number of layers converted
    """
    converted = 0

    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Convert to TernaryLinear
            ternary = TernaryLinear.from_bitlinear(
                module,
                use_triton=use_triton,
                cache_weights=cache_weights
            )

            setattr(parent, attr_name, ternary)
            converted += 1

    return converted


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def process_video(
    input_path: str,
    output_path: str,
    pipeline,
    prompt: str,
    max_frames: int = None,
):
    """
    Process a video with the given pipeline.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        pipeline: Loaded Pipeline instance
        prompt: Style prompt
        max_frames: Maximum frames to process (None = all frames)

    Returns:
        dict with stats
    """
    # Update prompt
    pipeline.update_prompt(prompt)

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames is None:
        max_frames = total_frames

    print(f"  Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    print(f"  Processing {min(max_frames, total_frames)} frames")

    # Get output resolution
    out_w, out_h = pipeline.get_output_resolution()

    # Create output video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Process frames
    frame_count = 0
    total_inference_time = 0

    # Warmup
    print("  Warming up...")
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        for _ in range(3):
            _ = pipeline.predict(pil_image)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_gpu_memory()

    print("  Processing frames...")
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Process frame
        torch.cuda.synchronize()
        start_infer = time.perf_counter()
        output_image = pipeline.predict(pil_image)
        torch.cuda.synchronize()
        infer_time = time.perf_counter() - start_infer
        total_inference_time += infer_time

        if output_image is None:
            continue

        # Convert back to BGR for OpenCV
        output_array = np.array(output_image)
        output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
        out.write(output_bgr)

        frame_count += 1
        if frame_count % 50 == 0:
            avg_fps = frame_count / total_inference_time
            print(f"    Frame {frame_count}/{min(max_frames, total_frames)} - FPS: {avg_fps:.2f}")

    # Cleanup
    cap.release()
    out.release()

    mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Calculate stats
    avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
    avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0

    return {
        "frames": frame_count,
        "total_time": total_inference_time,
        "avg_fps": avg_fps,
        "avg_latency_ms": avg_inference_time * 1000,
        "mem_peak_mb": mem_peak,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test TernaryLinear on full model")
    parser.add_argument(
        "--input",
        type=str,
        default="videos/1732530451289.mp4",  # 22 second video
        help="Input video path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/test_ternary",
        help="Output directory",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="anime-ghibli",
        help="Style preset",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,  # ~7 seconds at 30fps
        help="Maximum frames to process",
    )
    parser.add_argument(
        "--use-triton",
        action="store_true",
        help="Use Triton kernel (slower but memory-efficient)",
    )

    args = parser.parse_args()

    # Get prompt from style
    if args.style in PROMPT_PRESETS:
        prompt = PROMPT_PRESETS[args.style].prompt
    else:
        prompt = args.style

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # =========================================================================
    # Test 1: BitLinear (current implementation)
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: BitLinear (current quantized implementation)")
    print("="*70)

    print("\nLoading sd-turbo-1.58bit with BitLinear...")
    start_load = time.time()
    pipeline_bitlinear = Pipeline(model_key="sd-turbo-1.58bit", device="cuda")
    load_time = time.time() - start_load
    print(f"Loaded in {load_time:.2f}s")

    # Access the unet via stream.stream (StreamDiffusionWrapper wraps StreamDiffusion)
    unet = pipeline_bitlinear.stream.stream.unet

    # Count BitLinear layers
    bitlinear_count = sum(1 for m in unet.modules() if isinstance(m, BitLinear))
    print(f"BitLinear layers: {bitlinear_count}")

    output_path = output_dir / f"bitlinear_{args.style}.mp4"
    results["BitLinear"] = process_video(
        input_path=args.input,
        output_path=str(output_path),
        pipeline=pipeline_bitlinear,
        prompt=prompt,
        max_frames=args.max_frames,
    )
    results["BitLinear"]["output"] = str(output_path)

    print(f"\n  Results: {results['BitLinear']['avg_fps']:.2f} FPS, "
          f"{results['BitLinear']['avg_latency_ms']:.2f}ms latency, "
          f"{results['BitLinear']['mem_peak_mb']:.0f}MB peak")

    # =========================================================================
    # Test 2: TernaryLinear (cached mode - fastest)
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: TernaryLinear (cached mode - optimized)")
    print("="*70)

    print("\nConverting BitLinear → TernaryLinear (cached)...")
    start_convert = time.time()
    converted = convert_bitlinear_to_ternary(
        unet,
        use_triton=False,
        cache_weights=True
    )
    convert_time = time.time() - start_convert
    print(f"Converted {converted} layers in {convert_time:.2f}s")

    # Count TernaryLinear layers
    ternary_count = sum(1 for m in unet.modules() if isinstance(m, TernaryLinear))
    print(f"TernaryLinear layers: {ternary_count}")

    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()

    output_path = output_dir / f"ternary_cached_{args.style}.mp4"
    results["TernaryLinear (cached)"] = process_video(
        input_path=args.input,
        output_path=str(output_path),
        pipeline=pipeline_bitlinear,
        prompt=prompt,
        max_frames=args.max_frames,
    )
    results["TernaryLinear (cached)"]["output"] = str(output_path)

    print(f"\n  Results: {results['TernaryLinear (cached)']['avg_fps']:.2f} FPS, "
          f"{results['TernaryLinear (cached)']['avg_latency_ms']:.2f}ms latency, "
          f"{results['TernaryLinear (cached)']['mem_peak_mb']:.0f}MB peak")

    # =========================================================================
    # Test 3: TernaryLinear with Triton (optional)
    # =========================================================================
    if args.use_triton:
        print("\n" + "="*70)
        print("TEST 3: TernaryLinear (Triton mode - memory-efficient)")
        print("="*70)

        # Need to reload model for fair comparison
        del pipeline_bitlinear
        torch.cuda.empty_cache()
        gc.collect()

        print("\nReloading model for Triton test...")
        pipeline_triton = Pipeline(model_key="sd-turbo-1.58bit", device="cuda")

        print("Converting BitLinear → TernaryLinear (Triton)...")
        unet_triton = pipeline_triton.stream.stream.unet
        converted = convert_bitlinear_to_ternary(
            unet_triton,
            use_triton=True,
            cache_weights=False
        )
        print(f"Converted {converted} layers")

        torch.cuda.empty_cache()
        gc.collect()

        output_path = output_dir / f"ternary_triton_{args.style}.mp4"
        results["TernaryLinear (Triton)"] = process_video(
            input_path=args.input,
            output_path=str(output_path),
            pipeline=pipeline_triton,
            prompt=prompt,
            max_frames=args.max_frames,
        )
        results["TernaryLinear (Triton)"]["output"] = str(output_path)

        print(f"\n  Results: {results['TernaryLinear (Triton)']['avg_fps']:.2f} FPS, "
              f"{results['TernaryLinear (Triton)']['avg_latency_ms']:.2f}ms latency, "
              f"{results['TernaryLinear (Triton)']['mem_peak_mb']:.0f}MB peak")

        del pipeline_triton

    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Implementation':<30} {'FPS':>10} {'Latency (ms)':>15} {'Peak VRAM (MB)':>15}")
    print("-"*70)

    baseline_fps = results["BitLinear"]["avg_fps"]
    for name, r in results.items():
        speedup = r["avg_fps"] / baseline_fps
        speedup_str = f"({speedup:.2f}x)" if name != "BitLinear" else "(baseline)"
        print(f"{name:<30} {r['avg_fps']:>10.2f} {r['avg_latency_ms']:>15.2f} {r['mem_peak_mb']:>15.0f} {speedup_str}")

    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    for name, r in results.items():
        print(f"  {name}: {r['output']}")

    # Convert to h264 for better compatibility
    print("\n" + "="*70)
    print("Converting outputs to H.264...")
    print("="*70)
    for name, r in results.items():
        input_mp4 = r['output']
        output_h264 = input_mp4.replace('.mp4', '_h264.mp4')
        cmd = f'ffmpeg -y -i "{input_mp4}" -c:v libx264 -crf 18 -preset fast "{output_h264}" 2>/dev/null'
        os.system(cmd)
        print(f"  {name}: {output_h264}")

    print("\nDone!")


if __name__ == "__main__":
    main()
