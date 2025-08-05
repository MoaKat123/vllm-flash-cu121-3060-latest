# Use official NVIDIA CUDA base image with cuDNN for CUDA 12.1 and Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Install system dependencies, Python 3.10, pip, and dev tools
RUN apt-get update && apt-get install -y \
    git \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential cmake wget curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip and install build tools required by flash-attention
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# Install PyTorch with CUDA 12.1 support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install FlashAttention v2 from source
RUN git clone --branch v2.4.2 https://github.com/Dao-AILab/flash-attention.git /flash-attention \
    && cd /flash-attention \
    && python3 setup.py install

# Clone vLLM repo and install it directly (don't use requirements.txt)
RUN git clone https://github.com/vllm-project/vllm.git /vllm \
    && pip install /vllm

# Set working directory to vLLM
WORKDIR /vllm

# Expose the default API port
EXPOSE 8000

# Run vLLM OpenAI-compatible server
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
