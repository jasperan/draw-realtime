# StreamDiffusion Video-to-Video Demo

Transform videos with AI-powered diffusion using [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion). Upload or select a video, apply style prompts, and compare input/output side-by-side.

## Features

- **Video-to-Video processing** - Transform entire videos with AI styles
- **Side-by-side comparison** - Input and output videos with synchronized playback
- **Multiple input sources**:
  - Upload MP4 from browser
  - Server-side video library
- **Model selection** - Switch between SD-Turbo and SD 1.5 + LCM-LoRA
- **Custom prompts** - Any text prompt to define the transformation style
- **Progress tracking** - Real-time progress bar during processing
- **Job history** - View and reload previous processing results

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support (RTX 2060+ recommended)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Node.js 18+ (for frontend build)
- ffmpeg (for video encoding)

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

### Workflow

1. **Select a video** - Upload an MP4 or choose from server videos
2. **Configure settings** - Select model and enter a style prompt
3. **Process** - Click "Process Video" and wait for completion
4. **Compare** - Play both videos side-by-side with synchronized controls

### Input Sources

| Source | Description |
|--------|-------------|
| **Upload MP4** | Upload a video file from your computer |
| **Server Videos** | Select from pre-loaded videos on the server |

To add server-side videos, drop MP4 files in the `videos/` directory.

### Models

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| **SD-Turbo** | Fast | Good | Default, single-step inference |
| **SD 1.5 + LCM** | Medium | Higher | 4-step inference with LCM-LoRA |
| **Hyper-SDXL** | Fastest | SDXL | 1-step SDXL with Hyper-SD LoRA |

### Style Presets

Choose from 12 built-in style presets:

| Preset | Description |
|--------|-------------|
| `anime-ghibli` | Studio Ghibli inspired, soft colors |
| `anime-cyberpunk` | Anime + cyberpunk, neon, Makoto Shinkai |
| `cyberpunk-neon` | Cyberpunk city, neon lights, rain |
| `oil-painting` | Classical oil painting, rich colors |
| `watercolor` | Soft watercolor, flowing colors |
| `fantasy` | Magical fantasy art, ethereal |
| `dark-gothic` | Dark gothic, moody atmosphere |
| `comic-pop` | Comic book / pop art style |
| `photorealistic` | Ultra-detailed photorealistic |
| `impressionist` | Impressionist painting, Monet style |
| `pixel-art` | 16-bit retro pixel art |
| `sketch` | Pencil sketch, detailed linework |

### Custom Prompts

You can also enter custom prompts. Examples:
- `cyberpunk robot, neon lights, highly detailed`
- `oil painting, impressionist style, vibrant colors`
- `anime character, studio ghibli style`
- `zombie horror, dark atmosphere, cinematic`

### Playback Controls

- **Play Both** - Start both videos simultaneously
- **Pause Both** - Pause both videos
- **Restart** - Reset both to beginning and play
- **Sync Playback** - Toggle synchronized seeking

## Project Structure

```
draw-realtime/
├── app/                    # Python backend
│   ├── main.py            # FastAPI server
│   ├── pipeline.py        # StreamDiffusion wrapper
│   ├── video_source.py    # Video file handling
│   ├── video_processor.py # Batch video processing
│   └── config.py          # Configuration
├── frontend/              # Svelte web UI
│   ├── src/
│   │   └── App.svelte     # Main UI component
│   └── build/             # Production build
├── videos/                # Server-side video files
├── uploads/               # User uploaded videos
├── outputs/               # Processed output videos
├── engines/               # TensorRT cached engines
├── StreamDiffusion/       # StreamDiffusion library
├── requirements.txt
└── start.sh               # Launch script
```

## CLI Usage

Process videos from the command line:

```bash
# Use a style preset
python cli.py input.mp4 -s anime-ghibli

# Use higher quality model with preset
python cli.py input.mp4 -s oil-painting -m sd15-lcm

# Custom prompt
python cli.py input.mp4 -p "your custom prompt here"

# Process all server videos
python cli.py --process-all -s fantasy

# List available options
python cli.py --list-styles
python cli.py --list-models
python cli.py --list-videos
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/settings` | GET | Get app configuration |
| `/api/videos` | GET | List server-side videos |
| `/api/upload` | POST | Upload a video file |
| `/api/process` | POST | Start video processing |
| `/api/job/{id}` | GET | Get job status |
| `/api/jobs` | GET | List all jobs |
| `/api/output/{file}` | GET | Download processed video |
| `/api/input/{file}` | GET | Download input video |
| `/api/preview/{file}` | GET | Get real-time preview frame |

## Performance

Processing time depends on video length and GPU:

| GPU | 10s Video @ 30fps | Notes |
|-----|-------------------|-------|
| RTX 4090 | ~30s | Full TensorRT |
| RTX 3080 | ~60s | Full TensorRT |
| RTX 3060 | ~120s | xformers fallback |
| RTX 2060 | ~180s | xformers fallback |

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
The system will automatically fall back to xformers acceleration (slower but still functional).

### Out of memory
- Reduce resolution in `app/config.py` (default: 512x512)
- Use SD-Turbo instead of SD 1.5 + LCM
- Process shorter video clips

### Video won't play in browser
- Ensure ffmpeg is installed for H.264 encoding
- Try a different browser (Chrome recommended)

### Processing is slow
- Check if TensorRT acceleration is working (see logs)
- Reduce video length or resolution
- Use SD-Turbo model

## Technology

Built with:
- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) - Real-time diffusion pipeline
- [Stable Diffusion Turbo](https://huggingface.co/stabilityai/sd-turbo) - Fast image generation
- [TensorRT](https://developer.nvidia.com/tensorrt) - NVIDIA inference optimization
- [FastAPI](https://fastapi.tiangolo.com/) - Python web framework
- [Svelte](https://svelte.dev/) - Frontend framework
- [OpenCV](https://opencv.org/) - Video processing
- [ffmpeg](https://ffmpeg.org/) - Video encoding

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
