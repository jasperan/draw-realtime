#!/usr/bin/env python3
"""Test MonarchRT video generation using official WanT2V pipeline."""
import sys, os, gc, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MonarchRT"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.empty_cache()
gc.collect()
torch.set_grad_enabled(False)

device = torch.device("cuda:0")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
free = torch.cuda.mem_get_info(0)
print(f"VRAM: {free[0]/1e9:.1f} GB free / {free[1]/1e9:.1f} GB total")
print()

from wan.configs import WAN_CONFIGS
from wan.text2video import WanT2V

config = WAN_CONFIGS['t2v-1.3B']
checkpoint_dir = os.path.join(os.path.dirname(__file__), "MonarchRT", "wan_models", "Wan2.1-T2V-1.3B")

print("Initializing WanT2V pipeline...")
t0 = time.time()
pipeline = WanT2V(
    config=config,
    checkpoint_dir=checkpoint_dir,
    device_id=0,
    rank=0,
    t5_cpu=True,
)
print(f"Pipeline initialized in {time.time()-t0:.1f}s")

free = torch.cuda.mem_get_info(0)
print(f"VRAM: {free[0]/1e9:.1f} GB free")
print()

prompt = "A golden retriever running through a sunlit meadow with wildflowers, cinematic, beautiful lighting"

print(f"Generating video...")
print(f"  Prompt: {prompt}")
print(f"  Size: 832x480, 21 frames, 30 steps")
gen_start = time.time()

video = pipeline.generate(
    input_prompt=prompt,
    size=(832, 480),
    frame_num=21,
    shift=5.0,
    sample_solver='unipc',
    sampling_steps=30,
    guide_scale=5.0,
    seed=42,
    offload_model=True,
)

gen_time = time.time() - gen_start
num_frames = 21

print(f"\nGeneration complete!")
print(f"  Output shape: {video.shape}")
print(f"  Value range: [{video.min():.3f}, {video.max():.3f}]")
print(f"  Generation time: {gen_time:.1f}s")
print(f"  FPS: {num_frames / gen_time:.2f}")
print(f"  Time per frame: {gen_time / num_frames:.2f}s")
print()

# Save
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/monarchrt_official.mp4"

from torchvision.io import write_video
from einops import rearrange

# video is [C, T, H, W] with values in [-1, 1]
video_out = rearrange(video, 'c t h w -> t h w c')
video_out = ((video_out + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu()
write_video(output_path, video_out, fps=16)

import subprocess
h264_path = "outputs/monarchrt_official_h264.mp4"
result = subprocess.run([
    'ffmpeg', '-y', '-i', output_path,
    '-c:v', 'libx264', '-preset', 'fast',
    '-crf', '23', '-pix_fmt', 'yuv420p',
    h264_path
], capture_output=True, text=True)

if result.returncode == 0:
    os.replace(h264_path, output_path)

file_size = os.path.getsize(output_path) / (1024 * 1024)
print(f"Saved to: {output_path} ({file_size:.1f} MB)")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated(0)/1e9:.1f} GB")
