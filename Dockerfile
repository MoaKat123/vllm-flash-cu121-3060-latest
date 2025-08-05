# Use CUDA 11.8 base image (ideal for Ampere GPUs)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Install Python 3.10 and essential tools
RUN apt-get update && apt-get install -y \
    git python3.10 python3.10-dev python3-pip \
    build-essential cmake wget curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip and Python build tooling
RUN python3 -m pip install --upgrade pip setuptools wheel packaging build ninja
RUN ninja --version

# Install PyTorch with CUDA 11.8 (adjust version if needed)
RUN pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 -f https://download.pytorch.org/whl/cu118

# Configure parallel compile jobs to avoid memory overload
ENV MAX_JOBS=1

# Install FlashAttention v2 via pip with no build isolation to prefer prebuilt wheels
RUN pip install flash-attn==2.6.2 --extra-index-url https://pypi.nvidia.com --no-build-isolation

# If pip still tries to build from source (check logs), uncomment the manual fallback:
# COPY flash_attn‑2.6.2‑cp310‑cp310‑linux_x86_64.whl /tmp/
# RUN pip install /tmp/flash_attn‑2.6.2‑cp310‑cp310‑linux_x86_64.whl

# Install vLLM from PyPI
RUN pip install vllm==0.2.4

WORKDIR /vllm
EXPOSE 8000

# Start vLLM server (OpenAI-compatible API)
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
