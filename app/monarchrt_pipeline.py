"""MonarchRT pipeline wrapper for real-time video generation.

MonarchRT uses Monarch matrix attention for efficient video generation
via Diffusion Transformers. Supports Self-Forcing (autoregressive) and
Wan2.1 (bidirectional) models.

Requires separate installation:
    git clone https://github.com/Infini-AI-Lab/MonarchRT.git
    cd MonarchRT
    conda create -n monarch_rt python=3.10 -y && conda activate monarch_rt
    pip install -r requirements.txt
    pip install flash-attn --no-build-isolation
    python setup.py develop
"""

import gc
import sys
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# MonarchRT root directory (sibling to project root, like StreamDiffusion)
MONARCHRT_DIR = Path(__file__).parent.parent / "MonarchRT"

# Availability flag - set on first check
_monarchrt_available: Optional[bool] = None


def is_monarchrt_available() -> bool:
    """Check if MonarchRT is installed and importable."""
    global _monarchrt_available
    if _monarchrt_available is not None:
        return _monarchrt_available

    try:
        import torch
        if not torch.cuda.is_available():
            _monarchrt_available = False
            return False
    except ImportError:
        _monarchrt_available = False
        return False

    # Check if MonarchRT is installed as a package
    try:
        import monarch_rt  # noqa: F401
        _monarchrt_available = True
        return True
    except ImportError:
        pass

    # Check if MonarchRT directory exists with expected structure
    if (MONARCHRT_DIR / "pipeline").is_dir() and (MONARCHRT_DIR / "wan").is_dir():
        _monarchrt_available = True
        return True

    _monarchrt_available = False
    return False


def _ensure_monarchrt_on_path():
    """Add MonarchRT to sys.path if not already importable."""
    try:
        import monarch_rt  # noqa: F401
        return
    except ImportError:
        pass

    monarchrt_str = str(MONARCHRT_DIR)
    if monarchrt_str not in sys.path:
        sys.path.insert(0, monarchrt_str)


class MonarchRTPipeline:
    """Wrapper around MonarchRT inference pipelines.

    Supports two modes:
    - "causal": Self-Forcing autoregressive generation (faster, real-time capable)
    - "bidirectional": Wan2.1 bidirectional generation (higher quality)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipeline = None
        self.wan_model = None
        self.mode: Optional[str] = None  # "causal" or "bidirectional"
        self.config = None
        self._loaded = False

    def load_causal_model(
        self,
        config_path: str,
        checkpoint_path: str,
    ) -> bool:
        """Load a Self-Forcing (causal/autoregressive) MonarchRT model.

        Args:
            config_path: Path to YAML config file (e.g. configs/self_forcing_monarch_dmd.yaml)
            checkpoint_path: Path to model checkpoint directory
        """
        try:
            import torch
            from omegaconf import OmegaConf

            _ensure_monarchrt_on_path()
            from pipeline.causal_inference import CausalInferencePipeline

            logger.info(f"Loading MonarchRT causal model from {checkpoint_path}")

            # Load config
            cfg = OmegaConf.load(config_path)

            # Override checkpoint path
            cfg.generator_checkpoint = checkpoint_path

            # Initialize pipeline
            self.pipeline = CausalInferencePipeline(
                args=cfg,
                device=torch.device(self.device),
            )

            # Load checkpoint weights
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.is_dir():
                # PyTorch distributed checkpoint
                from torch.distributed.checkpoint import load as dist_load
                dist_load(self.pipeline.state_dict(), checkpoint_id=str(ckpt_path))
            else:
                state_dict = torch.load(str(ckpt_path), map_location=self.device)
                if "ema" in state_dict:
                    self.pipeline.load_state_dict(state_dict["ema"], strict=False)
                elif "model" in state_dict:
                    self.pipeline.load_state_dict(state_dict["model"], strict=False)
                else:
                    self.pipeline.load_state_dict(state_dict, strict=False)

            self.pipeline.eval()
            self.config = cfg
            self.mode = "causal"
            self._loaded = True
            logger.info("MonarchRT causal model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load MonarchRT causal model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_bidirectional_model(
        self,
        model_name: str = "Wan2.1-T2V-1.3B",
        checkpoint_dir: str = "",
    ) -> bool:
        """Load a Wan2.1 bidirectional MonarchRT model.

        Args:
            model_name: Wan model name (e.g. "Wan2.1-T2V-1.3B" or "Wan2.1-T2V-14B")
            checkpoint_dir: Path to Wan model checkpoint directory
        """
        try:

            _ensure_monarchrt_on_path()
            from wan.text2video import WanT2V
            from wan.configs import WAN_CONFIGS

            logger.info(f"Loading MonarchRT bidirectional model: {model_name}")

            wan_config = WAN_CONFIGS.get(model_name)
            if wan_config is None:
                logger.error(f"Unknown Wan model: {model_name}. Available: {list(WAN_CONFIGS.keys())}")
                return False

            if not checkpoint_dir:
                checkpoint_dir = str(MONARCHRT_DIR / "checkpoints" / model_name)

            self.wan_model = WanT2V(
                config=wan_config,
                checkpoint_dir=checkpoint_dir,
                device_id=int(self.device.split(":")[-1]) if ":" in self.device else 0,
            )

            self.mode = "bidirectional"
            self._loaded = True
            logger.info(f"MonarchRT bidirectional model {model_name} loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load MonarchRT bidirectional model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_video(
        self,
        prompt: str,
        num_frames: int = 21,
        width: int = 832,
        height: int = 480,
        seed: int = -1,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 4,
        negative_prompt: str = "",
    ) -> Optional[List[Image.Image]]:
        """Generate a video from a text prompt.

        Args:
            prompt: Text description of the video to generate
            num_frames: Number of frames to generate (must be 4n+1 for Wan models)
            width: Output video width
            height: Output video height
            seed: Random seed (-1 for random)
            guidance_scale: Classifier-free guidance strength
            num_inference_steps: Number of denoising steps
            negative_prompt: Negative prompt for guidance

        Returns:
            List of PIL Images (one per frame), or None on failure.
        """
        if not self._loaded:
            logger.error("No model loaded. Call load_causal_model() or load_bidirectional_model() first.")
            return None

        try:

            if self.mode == "bidirectional":
                return self._generate_bidirectional(
                    prompt, num_frames, width, height, seed,
                    guidance_scale, num_inference_steps, negative_prompt,
                )
            elif self.mode == "causal":
                return self._generate_causal(
                    prompt, num_frames, width, height, seed,
                    guidance_scale, num_inference_steps,
                )
            else:
                logger.error(f"Unknown pipeline mode: {self.mode}")
                return None

        except Exception as e:
            logger.error(f"MonarchRT generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_bidirectional(
        self,
        prompt: str,
        num_frames: int,
        width: int,
        height: int,
        seed: int,
        guidance_scale: float,
        num_inference_steps: int,
        negative_prompt: str,
    ) -> Optional[List[Image.Image]]:
        """Generate video using Wan2.1 bidirectional model."""

        # Wan2.1 requires frame count in format 4n+1
        num_frames = self._align_frame_count(num_frames)

        video_tensor = self.wan_model.generate(
            input_prompt=prompt,
            size=(width, height),
            frame_num=num_frames,
            shift=5.0,
            sample_solver="unipc",
            sampling_steps=num_inference_steps,
            guide_scale=guidance_scale,
            n_prompt=negative_prompt,
            seed=seed,
            offload_model=True,
        )

        return self._tensor_to_frames(video_tensor)

    def _generate_causal(
        self,
        prompt: str,
        num_frames: int,
        width: int,
        height: int,
        seed: int,
        guidance_scale: float,
        num_inference_steps: int,
    ) -> Optional[List[Image.Image]]:
        """Generate video using Self-Forcing causal model."""
        import torch

        if seed >= 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Calculate latent dimensions (patch size is typically (1,2,2))
        latent_h = height // 8  # VAE downscale
        latent_w = width // 8
        latent_channels = 16  # Wan model latent channels

        noise = torch.randn(
            1, num_frames, latent_channels, latent_h, latent_w,
            device=self.device, dtype=torch.bfloat16,
        )

        video_tensor = self.pipeline.inference(
            noise=noise,
            text_prompts=[prompt],
        )

        return self._tensor_to_frames(video_tensor)

    @staticmethod
    def _align_frame_count(n: int) -> int:
        """Align frame count to 4n+1 format required by Wan models."""
        # Find nearest valid count: 5, 9, 13, 17, 21, 25, ..., 81
        remainder = (n - 1) % 4
        if remainder == 0:
            return n
        return n + (4 - remainder)

    @staticmethod
    def _tensor_to_frames(video_tensor) -> List[Image.Image]:
        """Convert a video tensor (C, N, H, W) or (B, C, N, H, W) to list of PIL Images."""

        if video_tensor is None:
            return []

        # Handle batch dimension
        if video_tensor.dim() == 5:
            video_tensor = video_tensor[0]  # Take first sample

        # Ensure (C, N, H, W) format
        if video_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor (C, N, H, W), got {video_tensor.dim()}D")

        # Clamp to [0, 1] and convert to uint8
        video_tensor = video_tensor.clamp(0, 1)

        frames = []
        num_frames = video_tensor.shape[1]
        for i in range(num_frames):
            # Extract frame: (C, H, W) -> (H, W, C)
            frame = video_tensor[:, i, :, :].permute(1, 2, 0)
            frame_np = (frame.cpu().float().numpy() * 255).astype(np.uint8)
            frames.append(Image.fromarray(frame_np))

        return frames

    def cleanup(self):
        """Free GPU memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        if self.wan_model is not None:
            del self.wan_model
            self.wan_model = None

        self.mode = None
        self._loaded = False

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_info(self) -> dict:
        """Get info about the loaded model."""
        return {
            "loaded": self._loaded,
            "mode": self.mode,
            "device": self.device,
            "monarchrt_available": is_monarchrt_available(),
        }
