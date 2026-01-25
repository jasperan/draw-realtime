#!/usr/bin/env python3
"""
StreamDiffusion Video-to-Video CLI

Transform videos with AI-powered diffusion from the command line.

Usage:
    python cli.py input.mp4 -o output.mp4 -p "cyberpunk style"
    python cli.py input.mp4 --model sd15-lcm --prompt "oil painting"
    python cli.py --list-videos
    python cli.py --process-all -p "anime style"
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich import print as rprint

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import config, MODELS, DEFAULT_MODEL, DEFAULT_PROMPT
from app.video_source import get_available_videos, get_video_path

console = Console()


def get_pipeline():
    """Lazy load pipeline to avoid import time."""
    from app.pipeline import get_pipeline as _get_pipeline
    return _get_pipeline()


def list_models():
    """List available models."""
    table = Table(title="Available Models")
    table.add_column("Key", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Default", style="yellow")

    for key, model in MODELS.items():
        default = "✓" if key == DEFAULT_MODEL else ""
        table.add_row(key, model.description, default)

    console.print(table)


def list_videos():
    """List available server videos."""
    videos = get_available_videos()

    if not videos:
        console.print("[yellow]No videos found in videos/ directory[/yellow]")
        return

    table = Table(title="Available Videos")
    table.add_column("Name", style="cyan")
    table.add_column("Duration", style="green", justify="right")
    table.add_column("FPS", style="yellow", justify="right")

    for video in videos:
        table.add_row(
            video["name"],
            f"{video['duration']}s",
            str(video["fps"])
        )

    console.print(table)


def process_video(
    input_path: str,
    output_path: Optional[str] = None,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
) -> bool:
    """Process a single video."""

    # Validate input
    if not os.path.exists(input_path):
        # Check if it's a server video name
        server_path = get_video_path(input_path)
        if server_path:
            input_path = server_path
        else:
            console.print(f"[red]Error: Video not found: {input_path}[/red]")
            return False

    # Generate output path if not provided
    if output_path is None:
        input_name = Path(input_path).stem
        output_path = f"outputs/{input_name}_processed.mp4"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        console.print(f"[red]Error: Cannot open video: {input_path}[/red]")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    console.print(f"\n[bold cyan]Processing Video[/bold cyan]")
    console.print(f"  Input:    {input_path}")
    console.print(f"  Output:   {output_path}")
    console.print(f"  Duration: {duration:.1f}s ({total_frames} frames @ {fps:.1f} fps)")
    console.print(f"  Model:    {MODELS[model].description}")
    console.print(f"  Prompt:   {prompt}\n")

    # Load pipeline
    with console.status("[bold green]Loading model...") as status:
        pipeline = get_pipeline()

        if model != pipeline.get_current_model():
            status.update(f"[bold green]Switching to {model}...")
            pipeline.load_model(model)

        pipeline.update_prompt(prompt)

    # Create output video writer (temp file first)
    temp_path = output_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (config.width, config.height))

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green", finished_style="bright_green"),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing frames", total=total_frames)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Process through pipeline
            output_image = pipeline.predict(pil_image)

            if output_image is not None:
                # Convert back to BGR for OpenCV
                output_array = np.array(output_image)
                output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
                out.write(output_bgr)

            frame_idx += 1
            progress.update(task, advance=1)

    cap.release()
    out.release()

    # Re-encode with ffmpeg for browser compatibility
    console.print("\n[bold green]Encoding to H.264...[/bold green]")

    import subprocess
    result = subprocess.run([
        'ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '23', '-pix_fmt', 'yuv420p',
        output_path
    ], capture_output=True, text=True)

    if result.returncode == 0:
        os.remove(temp_path)
    else:
        # If ffmpeg fails, keep the temp file as output
        console.print("[yellow]Warning: ffmpeg encoding failed, using raw output[/yellow]")
        os.rename(temp_path, output_path)

    elapsed = time.time() - start_time
    fps_actual = total_frames / elapsed

    console.print(f"\n[bold green]✓ Complete![/bold green]")
    console.print(f"  Time:     {elapsed:.1f}s ({fps_actual:.1f} fps)")
    console.print(f"  Output:   {output_path}")

    # File size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    console.print(f"  Size:     {size_mb:.1f} MB\n")

    return True


def process_all_videos(prompt: str = DEFAULT_PROMPT, model: str = DEFAULT_MODEL):
    """Process all server videos."""
    videos = get_available_videos()

    if not videos:
        console.print("[yellow]No videos found in videos/ directory[/yellow]")
        return

    console.print(f"\n[bold cyan]Processing {len(videos)} videos[/bold cyan]\n")

    success = 0
    for i, video in enumerate(videos, 1):
        console.print(f"[bold]({i}/{len(videos)}) {video['name']}[/bold]")

        input_path = get_video_path(video["name"])
        if input_path and process_video(input_path, prompt=prompt, model=model):
            success += 1

    console.print(f"\n[bold green]Completed: {success}/{len(videos)} videos[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="StreamDiffusion Video-to-Video CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mp4 -o output.mp4 -p "cyberpunk style"
  %(prog)s video.mp4 --model sd15-lcm --prompt "oil painting"
  %(prog)s --list-videos
  %(prog)s --list-models
  %(prog)s --process-all -p "anime style, studio ghibli"
        """
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Input video file path or server video name"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output video file path (default: outputs/<input>_processed.mp4)"
    )
    parser.add_argument(
        "-p", "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Style prompt (default: '{DEFAULT_PROMPT}')"
    )
    parser.add_argument(
        "-m", "--model",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--list-videos",
        action="store_true",
        help="List available server videos"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process all server videos"
    )

    args = parser.parse_args()

    # Handle info commands
    if args.list_videos:
        list_videos()
        return

    if args.list_models:
        list_models()
        return

    # Handle processing commands
    if args.process_all:
        process_all_videos(prompt=args.prompt, model=args.model)
        return

    if args.input:
        success = process_video(
            input_path=args.input,
            output_path=args.output,
            prompt=args.prompt,
            model=args.model,
        )
        sys.exit(0 if success else 1)

    # No input provided
    parser.print_help()


if __name__ == "__main__":
    main()
