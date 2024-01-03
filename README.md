# StreamDiffusion

[English](./StreamDiffusion/README.md) | [日本語](./StreamDiffusion/README-ja.md)

<p align="center">
  <img src="./assets/demo_07.gif" width=90%>
  <img src="./assets/demo_09.gif" width=90%>
</p>

# StreamDiffusion: A Pipeline-Level Solution for Real-Time Interactive Generation

StreamDiffusion is an innovative diffusion pipeline designed for real-time interactive generation. It introduces significant performance enhancements to current diffusion-based image generation techniques.

[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2312.12491)
[![Hugging Face Papers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-papers-yellow)](https://huggingface.co/papers/2312.12491)


## Key Features

1. **Stream Batch**
   - Streamlined data processing through efficient batch operations.

2. **Residual Classifier-Free Guidance** - [Learn More](#residual-cfg-rcfg)
   - Improved guidance mechanism that minimizes computational redundancy.

3. **Stochastic Similarity Filter** - [Learn More](#stochastic-similarity-filter)
   - Improves GPU utilization efficiency through advanced filtering techniques.

4. **IO Queues**
   - Efficiently manages input and output operations for smoother execution.

5. **Pre-Computation for KV-Caches**
   - Optimizes caching strategies for accelerated processing.

6. **Model Acceleration Tools**
   - Utilizes various tools for model optimization and performance boost.

When images are produced using our proposed StreamDiffusion pipeline in an environment with **GPU: RTX 4090**, **CPU: Core i9-13900K**, and **OS: Ubuntu 22.04.3 LTS**.

|model                | Denoising Step      |  fps on Txt2Img      |  fps on Img2Img      |
|:-------------------:|:-------------------:|:--------------------:|:--------------------:|
|SD-turbo             | 1              | 106.16                    | 93.897               |
|LCM-LoRA <br>+<br> KohakuV2| 4        | 38.023                    | 37.133               |


## Installation

### Step0: clone this repository

```bash
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
```

### Step1: Make Environment

You can install StreamDiffusion via pip, conda, or Docker(explanation below).

```bash
conda create -n streamdiffusion python=3.10
conda activate streamdiffusion
```

OR

```cmd
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux
source .venv/bin/activate
```

### Step2: Install PyTorch

Select the appropriate version for your system.

CUDA 11.8

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.1

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
```
details: https://pytorch.org/

### Step3: Install StreamDiffusion

#### For User

Install StreamDiffusion

```bash
#for Latest Version (recommended)
pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]


#or


#for Stable Version
pip install streamdiffusion[tensorrt]
```

Install TensorRT extension

```bash
python -m streamdiffusion.tools.install-tensorrt
```
(Only for Windows) You may need to install pywin32 additionally, if you installed Stable Version(`pip install streamdiffusion[tensorrt]`).
```bash
pip install --force-reinstall pywin32
```

#### For Developer

```bash
python setup.py develop easy_install streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
```

### Docker Installation (TensorRT Ready)

```bash
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
cd StreamDiffusion
docker build -t stream-diffusion:latest -f Dockerfile .
docker run --gpus all -it -v $(pwd):/home/ubuntu/streamdiffusion stream-diffusion:latest
```

## Quick Start

You can try StreamDiffusion in [`examples`](./examples) directory.

| ![画像3](./assets/demo_02.gif) | ![画像4](./assets/demo_03.gif) |
|:--------------------:|:--------------------:|
| ![画像5](./assets/demo_04.gif) | ![画像6](./assets/demo_05.gif) |

## Acknowledgements

jasperan
