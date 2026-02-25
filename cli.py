#!/usr/bin/env python3
"""
StreamDiffusion Video-to-Video CLI

Transform videos with AI-powered diffusion from the command line.

Usage:
    python cli.py input.mp4 -s anime-ghibli
    python cli.py input.mp4 --style cyberpunk-neon --model sd15-lcm
    python cli.py input.mp4 -p "custom prompt here"
    python cli.py --list-styles
    python cli.py --list-videos
    python cli.py --process-all -s oil-painting

MonarchRT Text-to-Video Generation:
    python cli.py generate "a cat playing in a garden"
    python cli.py generate "ocean waves at sunset" -m monarchrt-wan --frames 81
    python cli.py generate "robot dancing" -o output.mp4 --seed 42

Multi-Style Mode (LLaVA + FLUX):
    python cli.py multistyle input.mp4
    python cli.py multistyle input.mp4 --description "a cat playing"
    python cli.py multistyle input.mp4 --output-dir ./my-outputs
"""

import argparse
import os
import shutil
import subprocess
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

from app.config import config, MODELS, DEFAULT_MODEL, DEFAULT_PROMPT, PROMPT_PRESETS, DEFAULT_PRESET
from app.video_source import get_available_videos, get_video_path
from app.multistyle import STYLES as MULTISTYLE_STYLES

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


def list_styles():
    """List available style presets."""
    table = Table(title="Available Style Presets")
    table.add_column("Key", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Default", style="yellow")

    for key, preset in PROMPT_PRESETS.items():
        default = "✓" if key == DEFAULT_PRESET else ""
        table.add_row(key, preset.description, default)

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
    out_w, out_h = pipeline.get_output_resolution()
    base, ext = os.path.splitext(output_path)
    temp_path = f"{base}_temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (out_w, out_h))

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


def generate_video(
    prompt: str,
    output_path: Optional[str] = None,
    model: str = "monarchrt-sf",
    num_frames: int = 21,
    seed: int = -1,
) -> bool:
    """Generate a video from text using MonarchRT."""
    from app.monarchrt_pipeline import is_monarchrt_available

    if not is_monarchrt_available():
        console.print("[red]Error: MonarchRT is not installed.[/red]")
        console.print("Install it with:")
        console.print("  git clone https://github.com/Infini-AI-Lab/MonarchRT.git")
        console.print("  cd MonarchRT && pip install -r requirements.txt")
        console.print("  pip install flash-attn --no-build-isolation")
        console.print("  python setup.py develop")
        return False

    if model not in MODELS:
        console.print(f"[red]Error: Unknown model: {model}[/red]")
        return False

    model_config = MODELS[model]
    if model_config.pipeline_type != "monarchrt":
        console.print(f"[red]Error: Model '{model}' is not a MonarchRT model.[/red]")
        console.print("Available MonarchRT models: monarchrt-sf, monarchrt-wan")
        return False

    # Generate output path if not provided
    if output_path is None:
        os.makedirs("outputs", exist_ok=True)
        import uuid
        gen_id = str(uuid.uuid4())[:8]
        output_path = f"outputs/generated_{gen_id}.mp4"

    out_w = model_config.width or 832
    out_h = model_config.height or 480
    fps = 16.0 if model_config.monarchrt_mode == "causal" else 24.0

    console.print(f"\n[bold cyan]MonarchRT Video Generation[/bold cyan]")
    console.print(f"  Model:    {model_config.description}")
    console.print(f"  Prompt:   {prompt}")
    console.print(f"  Frames:   {num_frames}")
    console.print(f"  Size:     {out_w}x{out_h}")
    console.print(f"  Output:   {output_path}\n")

    # Load pipeline
    with console.status("[bold green]Loading MonarchRT model...") as status:
        pipeline = get_pipeline()
        if model != pipeline.get_current_model():
            status.update(f"[bold green]Switching to {model}...")
            if not pipeline.load_model(model):
                console.print("[red]Error: Failed to load MonarchRT model[/red]")
                return False

    start_time = time.time()

    # Generate video
    with console.status("[bold green]Generating video frames...") as status:
        frames = pipeline.generate_video(
            prompt=prompt,
            num_frames=num_frames,
            width=out_w,
            height=out_h,
            seed=seed,
        )

    if not frames:
        console.print("[red]Error: Generation returned no frames[/red]")
        return False

    console.print(f"  Generated {len(frames)} frames")

    # Write to video file
    base, ext = os.path.splitext(output_path)
    temp_path = f"{base}_temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (out_w, out_h))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green", finished_style="bright_green"),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Writing frames", total=len(frames))

        for frame in frames:
            output_array = np.array(frame)
            output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
            h, w = output_bgr.shape[:2]
            if w != out_w or h != out_h:
                output_bgr = cv2.resize(output_bgr, (out_w, out_h))
            out.write(output_bgr)
            progress.update(task, advance=1)

    out.release()

    # Re-encode with ffmpeg for browser compatibility
    console.print("\n[bold green]Encoding to H.264...[/bold green]")

    result = subprocess.run([
        'ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '23', '-pix_fmt', 'yuv420p',
        output_path
    ], capture_output=True, text=True)

    if result.returncode == 0:
        os.remove(temp_path)
    else:
        console.print("[yellow]Warning: ffmpeg encoding failed, using raw output[/yellow]")
        os.rename(temp_path, output_path)

    elapsed = time.time() - start_time
    fps_actual = len(frames) / elapsed

    console.print(f"\n[bold green]Complete![/bold green]")
    console.print(f"  Time:     {elapsed:.1f}s ({fps_actual:.1f} fps)")
    console.print(f"  Output:   {output_path}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    console.print(f"  Size:     {size_mb:.1f} MB\n")

    return True


def process_multistyle(
    input_path: str,
    output_dir: Optional[str] = None,
    description: Optional[str] = None,
) -> bool:
    """Process a video with multi-style FLUX generation."""
    from app.multistyle import get_multistyle_processor, STYLES

    # Validate input
    if not os.path.exists(input_path):
        # Check if it's a server video name
        server_path = get_video_path(input_path)
        if server_path:
            input_path = server_path
        else:
            console.print(f"[red]Error: Video not found: {input_path}[/red]")
            return False

    # Get video info
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        console.print(f"[red]Error: Cannot open video: {input_path}[/red]")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    console.print(f"\n[bold cyan]Multi-Style FLUX Generation[/bold cyan]")
    console.print(f"  Input:    {input_path}")
    console.print(f"  Duration: {duration:.1f}s ({total_frames} frames @ {fps:.1f} fps)")
    console.print(f"  Styles:   {', '.join(s[0] for s in STYLES)}\n")

    # Get processor
    processor = get_multistyle_processor()

    # Create job
    job = processor.create_job(input_path)

    if output_dir:
        job.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    console.print(f"  Output:   {job.output_dir}\n")

    start_time = time.time()

    # Phase 1: LLaVA analysis (or use provided description)
    if description:
        console.print(f"[bold green]Using provided description...[/bold green]")
        job.description = description
    else:
        with console.status("[bold green]Analyzing video with LLaVA...") as status:
            job.description = processor.analyze_video_with_llava(input_path)

    console.print(f"[bold]Description:[/bold] {job.description}\n")

    # Phase 2: Generate each style with FLUX
    console.print("[bold green]Loading FLUX model...[/bold green]")
    pipeline = get_pipeline()
    pipeline.load_model("flux2-klein")

    for i, (style_slug, style_desc) in enumerate(STYLES):
        prompt = f"{job.description}, detailed, vibrant colors, {style_desc}"

        console.print(f"\n[bold][{i + 1}/{len(STYLES)}] {style_slug}[/bold]")
        console.print(f"  Prompt: {prompt[:80]}...")

        pipeline.update_prompt(prompt)

        # Process video for this style
        input_name = Path(input_path).stem
        output_filename = f"{input_name}_{style_slug}.mp4"
        output_path = str(Path(job.output_dir) / output_filename)

        # Open input video
        cap = cv2.VideoCapture(input_path)
        out_w, out_h = pipeline.get_output_resolution()
        base_s, _ = os.path.splitext(output_path)
        temp_path = f"{base_s}_temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (out_w, out_h))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bright_green"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Processing {style_slug}", total=total_frames)

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                output_image = pipeline.predict(pil_image)

                if output_image is not None:
                    output_array = np.array(output_image)
                    output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
                    out.write(output_bgr)

                frame_idx += 1
                progress.update(task, advance=1)

        cap.release()
        out.release()

        # Re-encode with H.264
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
            os.rename(temp_path, output_path)

        job.completed_outputs.append(output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        console.print(f"  [green]✓[/green] {output_filename} ({size_mb:.1f} MB)")

    # Phase 3: Create comparison grid
    console.print("\n[bold green]Creating comparison grid...[/bold green]")
    grid_path = processor.create_comparison_grid(job)

    if grid_path:
        job.grid_output = grid_path
        size_mb = os.path.getsize(grid_path) / (1024 * 1024)
        console.print(f"  [green]✓[/green] {os.path.basename(grid_path)} ({size_mb:.1f} MB)")

    elapsed = time.time() - start_time

    console.print(f"\n[bold green]✓ Multi-Style Complete![/bold green]")
    console.print(f"  Time:     {elapsed:.1f}s")
    console.print(f"  Outputs:  {len(job.completed_outputs)} style videos + 1 grid")
    console.print(f"  Location: {job.output_dir}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="StreamDiffusion Video-to-Video CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mp4 -s anime-ghibli              # Use anime style preset
  %(prog)s input.mp4 -s oil-painting -m sd15-lcm  # Higher quality model
  %(prog)s input.mp4 -p "custom prompt here"      # Custom prompt
  %(prog)s --list-styles                          # Show all style presets
  %(prog)s --list-videos
  %(prog)s --list-models
  %(prog)s --process-all -s fantasy               # Process all with fantasy style

Multi-Style Mode (LLaVA + FLUX):
  %(prog)s multistyle input.mp4                   # Analyze with LLaVA, generate 5 styles
  %(prog)s multistyle input.mp4 -d "a cat playing" # Skip LLaVA, use description
  %(prog)s multistyle input.mp4 -o ./my-outputs   # Custom output directory
        """
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate subcommand (MonarchRT text-to-video)
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate video from text using MonarchRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MonarchRT models:
  monarchrt-sf    Self-Forcing (real-time 16fps, autoregressive)
  monarchrt-wan   Wan2.1 (high quality, bidirectional)

Examples:
  %(prog)s "a cat playing in a garden"
  %(prog)s "ocean waves at sunset" -m monarchrt-wan --frames 81
  %(prog)s "robot dancing" -o output.mp4 --seed 42
        """
    )
    generate_parser.add_argument(
        "prompt",
        help="Text description of the video to generate"
    )
    generate_parser.add_argument(
        "-o", "--output",
        help="Output video file path (default: outputs/generated_<id>.mp4)"
    )
    generate_parser.add_argument(
        "-m", "--model",
        choices=["monarchrt-sf", "monarchrt-wan"],
        default="monarchrt-sf",
        help="MonarchRT model to use (default: monarchrt-sf)"
    )
    generate_parser.add_argument(
        "--frames",
        type=int,
        default=21,
        help="Number of frames to generate (default: 21)"
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for random)"
    )

    # Multistyle subcommand
    multistyle_parser = subparsers.add_parser(
        "multistyle",
        help="Generate 5 artistic styles with LLaVA + FLUX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Styles generated:
  - Oil painting (rich textures, classic feel)
  - Watercolor (soft, flowing, dreamlike)
  - Impressionist (Monet-style brushstrokes)
  - Pop art (bold colors, Warhol-esque)
  - Ukiyo-e (Japanese woodblock print)

Outputs 5 individual videos + 1 comparison grid video.
        """
    )
    multistyle_parser.add_argument(
        "input",
        help="Input video file path or server video name"
    )
    multistyle_parser.add_argument(
        "-o", "--output-dir",
        help="Output directory (default: outputs/multistyle/<job-id>)"
    )
    multistyle_parser.add_argument(
        "-d", "--description",
        help="Skip LLaVA analysis and use this description instead"
    )

    # Main parser arguments (for non-subcommand usage)
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
        default=None,
        help="Custom style prompt (overrides --style)"
    )
    parser.add_argument(
        "-s", "--style",
        choices=list(PROMPT_PRESETS.keys()),
        default=DEFAULT_PRESET,
        help=f"Style preset (default: {DEFAULT_PRESET})"
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
        "--list-styles",
        action="store_true",
        help="List available style presets"
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process all server videos"
    )

    args = parser.parse_args()

    # Handle info commands first (no GPU/ffmpeg needed)
    if hasattr(args, 'list_videos') and args.list_videos:
        list_videos()
        return
    if hasattr(args, 'list_models') and args.list_models:
        list_models()
        return
    if hasattr(args, 'list_styles') and args.list_styles:
        list_styles()
        return

    # Pre-flight checks for processing commands
    needs_processing = (
        args.command == "multistyle"
        or getattr(args, 'process_all', False)
        or getattr(args, 'input', None)
    )

    if needs_processing:
        # Check ffmpeg
        if not shutil.which('ffmpeg'):
            console.print("[red]Error: ffmpeg not found. Install ffmpeg and ensure it's on your PATH.[/red]")
            sys.exit(1)

        # Check CUDA
        try:
            import torch
            if not torch.cuda.is_available():
                console.print("[red]Error: No CUDA GPU detected. This tool requires an NVIDIA GPU with CUDA support.[/red]")
                sys.exit(1)
        except ImportError:
            console.print("[red]Error: PyTorch not installed. Run: pip install torch[/red]")
            sys.exit(1)

    # Handle generate subcommand (MonarchRT)
    if args.command == "generate":
        success = generate_video(
            prompt=args.prompt,
            output_path=args.output,
            model=args.model,
            num_frames=args.frames,
            seed=args.seed,
        )
        sys.exit(0 if success else 1)

    # Handle multistyle subcommand
    if args.command == "multistyle":
        success = process_multistyle(
            input_path=args.input,
            output_dir=args.output_dir,
            description=args.description,
        )
        sys.exit(0 if success else 1)

    # Determine prompt: custom prompt takes priority, otherwise use style preset
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = PROMPT_PRESETS[args.style].prompt

    # Handle processing commands
    if args.process_all:
        process_all_videos(prompt=prompt, model=args.model)
        return

    if args.input:
        success = process_video(
            input_path=args.input,
            output_path=args.output,
            prompt=prompt,
            model=args.model,
        )
        sys.exit(0 if success else 1)

    # No input provided
    parser.print_help()


if __name__ == "__main__":
    main()
