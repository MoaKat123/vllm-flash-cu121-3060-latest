# Use official NVIDIA CUDA 11.8 image (ideal for RTX 3060 and FlashAttention wheels)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# ----------------------
# Install Python 3.10 and tools
# ----------------------
RUN apt-get update && apt-get install -y \
    git \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential cmake wget curl ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip and install required Python tools
RUN python3 -m pip install --upgrade pip setuptools wheel packaging build ninja
RUN ninja --version  # sanity check

# ----------------------
# Install PyTorch (CUDA 11.8 build)
# ----------------------
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ----------------------
# Install FlashAttention v2.3.5 (has wheel for Python 3.10 + CUDA 11.8)
# Avoid source build by disabling build isolation
# ----------------------
RUN pip install flash-attn==2.3.5 --extra-index-url https://pypi.nvidia.com --no-build-isolation

# ----------------------
# Install vLLM from PyPI
# ----------------------
RUN pip install vllm==0.2.4

# ----------------------
# Set working dir & expose port
# ----------------------
WORKDIR /vllm
EXPOSE 8000

# ----------------------
# Start the OpenAI-compatible server
# You can override this when running to specify a model
# ----------------------
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
