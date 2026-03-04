#!/usr/bin/env python3
"""Test MonarchRT video generation on A10 GPU.

Uses manual memory management to fit in 23GB VRAM.
"""

import sys
import os
import time
import gc
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MonarchRT"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.cuda.amp as amp
from contextlib import contextmanager

torch.cuda.empty_cache()
gc.collect()
torch.set_grad_enabled(False)

device = torch.device("cuda:0")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
free = torch.cuda.mem_get_info(0)
print(f"VRAM: {free[0]/1e9:.1f} GB free / {free[1]/1e9:.1f} GB total")
print(f"System RAM: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9:.0f} GB")
print()

# Import Wan components
from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel

config = WAN_CONFIGS['t2v-1.3B']
checkpoint_dir = os.path.join(os.path.dirname(__file__), "MonarchRT", "wan_models", "Wan2.1-T2V-1.3B")

# ---- Step 1: Load T5 on CPU, encode text, then free it ----
print("Step 1: Loading T5 text encoder (CPU)...")
t1 = time.time()
text_encoder = T5EncoderModel(
    text_len=config.text_len,
    dtype=config.t5_dtype,
    device=torch.device('cpu'),
    checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
    tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
)

prompt = "A golden retriever running through a sunlit meadow with wildflowers, cinematic, beautiful lighting"
neg_prompt = config.sample_neg_prompt

context = text_encoder([prompt], torch.device('cpu'))
context_null = text_encoder([neg_prompt], torch.device('cpu'))
context = [t.to(device) for t in context]
context_null = [t.to(device) for t in context_null]

# Free T5 entirely
del text_encoder
gc.collect()
print(f"  T5 loaded and encoded in {time.time()-t1:.1f}s (freed from memory)")

# ---- Step 2: Load DiT model ----
print("Step 2: Loading DiT model (GPU)...")
t2 = time.time()
dit = WanModel.from_pretrained(checkpoint_dir)
dit.eval().requires_grad_(False)
dit.to(device)
print(f"  DiT loaded in {time.time()-t2:.1f}s")

free = torch.cuda.mem_get_info(0)
print(f"  VRAM: {free[0]/1e9:.1f} GB free")

# VAE will be loaded after DiT is offloaded to save VRAM
print()

# ---- Step 4: Generate video ----
num_frames = 21
width, height = 832, 480
vae_stride = config.vae_stride
patch_size = config.patch_size

target_shape = (
    16,  # z_dim
    (num_frames - 1) // vae_stride[0] + 1,
    height // vae_stride[1],
    width // vae_stride[2],
)

seq_len = math.ceil(
    (target_shape[2] * target_shape[3]) /
    (patch_size[1] * patch_size[2]) *
    target_shape[1]
)

print(f"Generating video:")
print(f"  Prompt: {prompt}")
print(f"  Frames: {num_frames}")
print(f"  Size:   {width}x{height}")
print(f"  Latent: {target_shape}")
print(f"  Seq len: {seq_len}")
print()

seed = 42
seed_g = torch.Generator(device=device)
seed_g.manual_seed(seed)
torch.manual_seed(seed)

noise = [
    torch.randn(
        target_shape[0], target_shape[1], target_shape[2], target_shape[3],
        dtype=torch.float32, device=device, generator=seed_g,
    )
]

# Set up scheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

sampling_steps = 30
shift = 5.0
guide_scale = 5.0

@contextmanager
def noop():
    yield

gen_start = time.time()

with amp.autocast(dtype=torch.bfloat16), torch.no_grad(), noop():
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
    timesteps = sample_scheduler.timesteps

    latents = noise

    arg_c = {'context': context, 'seq_len': seq_len}
    arg_null = {'context': context_null, 'seq_len': seq_len}

    from tqdm import tqdm
    for step_i, t in enumerate(tqdm(timesteps)):
        latent_model_input = latents
        timestep = torch.stack([t])

        noise_pred_cond = dit(latent_model_input, t=timestep, **arg_c)[0]
        noise_pred_uncond = dit(latent_model_input, t=timestep, **arg_null)[0]

        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

        temp_x0 = sample_scheduler.step(
            noise_pred.unsqueeze(0),
            t,
            latents[0].unsqueeze(0),
            return_dict=False,
            generator=seed_g,
        )[0]
        latents = [temp_x0.squeeze(0)]

    x0 = latents

    # Free DiT to make room for VAE decoding
    dit.cpu()
    del dit
    gc.collect()
    torch.cuda.empty_cache()

    free = torch.cuda.mem_get_info(0)
    print(f"\n  DiT offloaded. VRAM: {free[0]/1e9:.1f} GB free")

    # Load VAE now that DiT is freed
    print("  Loading VAE for decoding...")
    t3 = time.time()
    vae = WanVAE(
        vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
        device=device,
    )
    print(f"  VAE loaded in {time.time()-t3:.1f}s")

    videos = vae.decode(x0)

gen_time = time.time() - gen_start

print(f"\nGeneration complete!")
# videos is a list of tensors, each [C, T, H, W]
video_tensor = videos[0]  # first (and only) video
print(f"  Output shape: {video_tensor.shape}")
print(f"  Generation time: {gen_time:.1f}s")
print(f"  FPS: {num_frames / gen_time:.2f}")
print(f"  Time per frame: {gen_time / num_frames:.2f}s")
print()

# Save the video
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/monarchrt_sample.mp4"

from torchvision.io import write_video
from einops import rearrange

# video_tensor is [C, T, H, W] with values in [-1, 1]
video_out = rearrange(video_tensor, 'c t h w -> t h w c')
video_out = ((video_out + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu()

write_video(output_path, video_out, fps=16)
file_size = os.path.getsize(output_path) / (1024 * 1024)
print(f"Saved to: {output_path} ({file_size:.1f} MB)")

# Re-encode with ffmpeg
import subprocess
h264_path = "outputs/monarchrt_sample_h264.mp4"
result = subprocess.run([
    'ffmpeg', '-y', '-i', output_path,
    '-c:v', 'libx264', '-preset', 'fast',
    '-crf', '23', '-pix_fmt', 'yuv420p',
    h264_path
], capture_output=True, text=True)

if result.returncode == 0:
    os.replace(h264_path, output_path)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Re-encoded H.264: {output_path} ({file_size:.1f} MB)")

print(f"\nPeak VRAM used: {torch.cuda.max_memory_allocated(0)/1e9:.1f} GB")
