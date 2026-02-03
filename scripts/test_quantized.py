#!/usr/bin/env python3
"""
Test script for quantized models.
Processes a sample video and saves the output for visualization.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from PIL import Image

from app.pipeline import Pipeline
from app.config import MODELS, PROMPT_PRESETS


def process_video(
    input_path: str,
    output_path: str,
    model_key: str,
    prompt: str,
    max_frames: int = 100,
):
    """
    Process a video with the specified model.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        model_key: Model key (e.g., "sd-turbo-1.58bit")
        prompt: Style prompt
        max_frames: Maximum frames to process (for testing)
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_key}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps} fps, {total_frames} frames")
    print(f"Processing up to {min(max_frames, total_frames)} frames")

    # Load model
    print(f"\nLoading model: {model_key}...")
    start_load = time.time()

    pipeline = Pipeline(model_key=model_key, device="cuda")
    pipeline.update_prompt(prompt)

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")

    # Get output resolution
    out_w, out_h = pipeline.get_output_resolution()
    print(f"Output resolution: {out_w}x{out_h}")

    # Create output video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Process frames
    frame_count = 0
    total_inference_time = 0

    print(f"\nProcessing frames...")

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Process frame
        start_infer = time.time()
        output_image = pipeline.predict(pil_image)
        infer_time = time.time() - start_infer
        total_inference_time += infer_time

        if output_image is None:
            print(f"Frame {frame_count}: Failed to generate output")
            continue

        # Convert back to BGR for OpenCV
        output_array = np.array(output_image)
        output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)

        # Write frame
        out.write(output_bgr)

        frame_count += 1
        if frame_count % 10 == 0:
            avg_fps = frame_count / total_inference_time
            print(f"  Frame {frame_count}/{min(max_frames, total_frames)} - "
                  f"Avg FPS: {avg_fps:.2f}")

    # Cleanup
    cap.release()
    out.release()

    # Print stats
    avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
    avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total inference time: {total_inference_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average latency: {avg_inference_time*1000:.2f}ms")
    print(f"  Output saved to: {output_path}")
    print(f"{'='*60}\n")

    return {
        "frames": frame_count,
        "total_time": total_inference_time,
        "avg_fps": avg_fps,
        "avg_latency_ms": avg_inference_time * 1000,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test quantized models")
    parser.add_argument(
        "--input",
        type=str,
        default="videos/time_lapse_video_of_night_sky (Original).mp4",
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/test_quantized",
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sd-turbo-1.58bit",
        help="Model to test",
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
        default=50,
        help="Maximum frames to process",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also process with original model for comparison",
    )

    args = parser.parse_args()

    # Get prompt from style
    if args.style in PROMPT_PRESETS:
        prompt = PROMPT_PRESETS[args.style].prompt
    else:
        prompt = args.style

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Test quantized model
    output_path = output_dir / f"{args.model}_{args.style}.mp4"
    results[args.model] = process_video(
        input_path=args.input,
        output_path=str(output_path),
        model_key=args.model,
        prompt=prompt,
        max_frames=args.max_frames,
    )

    # Test original model for comparison if requested
    if args.compare:
        # Get base model key
        base_model = args.model.replace("-1.58bit", "")
        if base_model in MODELS:
            output_path = output_dir / f"{base_model}_{args.style}.mp4"
            results[base_model] = process_video(
                input_path=args.input,
                output_path=str(output_path),
                model_key=base_model,
                prompt=prompt,
                max_frames=args.max_frames,
            )

            # Print comparison
            print("\n" + "="*60)
            print("COMPARISON")
            print("="*60)
            print(f"{'Model':<25} {'FPS':>10} {'Latency (ms)':>15}")
            print("-"*60)
            for model_key, r in results.items():
                print(f"{model_key:<25} {r['avg_fps']:>10.2f} {r['avg_latency_ms']:>15.2f}")
            print("="*60)

    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
