# StreamDiffusion Real-Time Demo

Real-time AI video filter demo using [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion). Transform webcam feeds or video files with AI-generated styles in real-time.

## Features

- **Real-time processing** - 30+ FPS on modern GPUs with TensorRT acceleration
- **Multiple input sources**:
  - Webcam (live camera feed)
  - MP4 upload (from browser)
  - Server videos (pre-loaded files)
- **Model selection** - Switch between SD-Turbo and SD 1.5 + LCM-LoRA
- **Live prompt editing** - Change the style prompt without restarting
- **Web UI** - Clean, minimal interface accessible from any browser

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support (RTX 2060+ recommended)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Node.js 18+ (for frontend build)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jasperan/draw-realtime.git
   cd draw-realtime
   ```

2. **Create conda environment**
   ```bash
   conda create -n streamdiffusion python=3.10 -y
   conda activate streamdiffusion
   ```

3. **Install PyTorch with CUDA**
   ```bash
   # For CUDA 11.8
   pip install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install StreamDiffusion with TensorRT**
   ```bash
   pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]
   python -m streamdiffusion.tools.install-tensorrt
   ```

5. **Install web dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Build frontend**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

### Run the Demo

```bash
./start.sh
```

Then open http://localhost:7860 in your browser.

**First run notes:**
- Models will download automatically (~3GB for SD-Turbo)
- TensorRT engines compile on first run (5-10 minutes)
- Subsequent runs start instantly

## Usage

### Input Sources

| Source | Description |
|--------|-------------|
| **Webcam** | Live camera feed from your device |
| **Upload MP4** | Upload a video file from your computer |
| **Server Videos** | Select from pre-loaded videos on the server |

To add server-side videos, drop MP4 files in the `videos/` directory.

### Models

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| **SD-Turbo** | Fast | Good | Default, single-step inference |
| **SD 1.5 + LCM** | Medium | Higher | 4-step inference with LCM-LoRA |

### Prompts

Enter any text prompt to change the style. Examples:
- `cyberpunk robot, neon lights, highly detailed`
- `oil painting, impressionist style, vibrant colors`
- `anime character, studio ghibli style`
- `zombie horror, dark atmosphere, cinematic`

## Project Structure

```
draw-realtime/
├── app/                    # Python backend
│   ├── main.py            # FastAPI server
│   ├── pipeline.py        # StreamDiffusion wrapper
│   ├── video_source.py    # Video file handling
│   └── config.py          # Configuration
├── frontend/              # Svelte web UI
│   ├── src/
│   │   └── App.svelte     # Main UI component
│   └── build/             # Production build
├── videos/                # Server-side video files
├── engines/               # TensorRT cached engines
├── StreamDiffusion/       # StreamDiffusion library
├── requirements.txt
└── start.sh               # Launch script
```

## Performance

Expected FPS with TensorRT acceleration:

| GPU | Resolution | FPS |
|-----|------------|-----|
| RTX 4090 | 512x512 | 90-100 |
| RTX 3080 | 512x512 | 50-60 |
| RTX 3060 | 512x512 | 30-40 |
| RTX 2060 | 512x512 | 15-25 |

## Configuration

Environment variables (optional):

```bash
HOST=0.0.0.0          # Server host
PORT=7860             # Server port
VIDEOS_DIR=videos     # Video files directory
ENGINES_DIR=engines   # TensorRT engines directory
DEBUG=true            # Enable debug logging
```

## Troubleshooting

### TensorRT compilation fails
The system will automatically fall back to xformers acceleration (30-50% slower but still real-time).

### Out of memory
- Reduce resolution in `app/config.py` (default: 512x512)
- Use SD-Turbo instead of SD 1.5 + LCM

### Webcam not working
- Ensure browser has camera permissions
- Try a different browser (Chrome recommended)
- Check if another application is using the camera

## Technology

Built with:
- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) - Real-time diffusion pipeline
- [Stable Diffusion Turbo](https://huggingface.co/stabilityai/sd-turbo) - Fast image generation
- [TensorRT](https://developer.nvidia.com/tensorrt) - NVIDIA inference optimization
- [FastAPI](https://fastapi.tiangolo.com/) - Python web framework
- [Svelte](https://svelte.dev/) - Frontend framework

## References

- [StreamDiffusion Paper](https://arxiv.org/abs/2312.12491)
- [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
- [TinyVAE (TAESD)](https://huggingface.co/madebyollin/taesd)

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgements

- [cumulo-autumn/StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) for the core pipeline
- Stability AI for SD-Turbo
- The Hugging Face community
