# Use CUDA 11.8 base image for RTX 3060 compatibility
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Install Python 3.10 and essential build tools
RUN apt-get update && apt-get install -y \
    git python3.10 python3.10-dev python3-pip \
    build-essential cmake wget curl ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip and install build-related Python packages
RUN python3 -m pip install --upgrade pip setuptools wheel packaging build

# Install PyTorch and torchvision CUDA 11.8 wheels (without +cu118 suffix)
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install FlashAttention v2.6.2 prebuilt wheel from NVIDIA PyPI (no build isolation to avoid source build)
RUN pip install flash-attn==2.6.2 --extra-index-url https://pypi.nvidia.com --no-build-isolation

# Install vLLM from PyPI
RUN pip install vllm==0.2.4

# Set working directory for vLLM server
WORKDIR /vllm

# Expose vLLM API port
EXPOSE 8000

# Run the OpenAI-compatible vLLM API server
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
