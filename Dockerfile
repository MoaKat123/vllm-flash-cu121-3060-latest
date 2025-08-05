# Use CUDA 11.8 base image compatible with RTX 3060 and FlashAttention v2 wheels
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Install Python 3.10 and essential system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential cmake wget curl ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip and install Python build tools
RUN python3 -m pip install --upgrade pip setuptools wheel packaging build

# Install PyTorch with CUDA 11.8 support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install FlashAttention v2.4.2 via prebuilt wheel (no source build)
RUN pip install flash-attn==2.4.2 --extra-index-url https://pypi.nvidia.com

# Install vLLM from PyPI (no source build)
RUN pip install vllm==0.2.4

# Optional: Set working directory for running server or future configs
WORKDIR /vllm

# Expose the vLLM server port
EXPOSE 8000

# Run vLLM's OpenAI-compatible API server
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
