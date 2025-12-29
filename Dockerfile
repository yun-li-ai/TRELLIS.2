# Modifications Copyright (c) Applied Intuition, Inc.

# To build it with predownloaded models, run the following command:
# docker build --build-arg HF_TOKEN="<your_huggingface_token>" -t quay.io/applied_dev/applied2:trellis2-1.0.0 .
# You will need to have a HuggingFace account and get the token from your HuggingFace account settings.
# Your HuggingFace account should ask permission and get approval for the following gated models:
# - https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
# - https://huggingface.co/briaai/RMBG-2.0
# My experience is that you should wait for a few minutes to get the approval.


# Use PyTorch base image with CUDA 12.4
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Set working directory
WORKDIR /app

# Set environment variables early
ENV PYTHONPATH=/app
# If you really need to append to an existing PYTHONPATH, use:
# ENV PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}/app"
ENV ATTN_BACKEND=flash_attn
ENV SPCONV_ALGO=native
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
ENV CXX=/usr/local/bin/gxx-wrapper

# Create temporary directory needed by headless_app.py
RUN mkdir -p /tmp/Trellis2-demo

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    build-essential \
    git \
    python3-onnx \
    rdfind \
    ninja-build \
    curl \
    libjpeg-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files
COPY . /app/

# Initialize git submodules
RUN cd /app && \
    git init && \
    git submodule init && \
    git submodule update --init --recursive && \
    git submodule update --recursive && \
    rm -rf .git */.git **/.git

# Setup CUDA environment for JIT compilation
RUN printf '#!/usr/bin/env bash\nexec /usr/bin/g++ -I/usr/local/cuda/include -I/usr/local/cuda/include/crt "$@"\n' > /usr/local/bin/gxx-wrapper && \
    chmod +x /usr/local/bin/gxx-wrapper

# Before making setup.sh executable, we should add shebang
RUN echo '#!/bin/bash' | cat - setup.sh > temp && mv temp setup.sh && \
    chmod +x setup.sh

# Install PyTorch 2.6.0 with CUDA 12.4
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install basic dependencies
RUN pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    scipy ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard \
    pillow-simd kornia timm huggingface_hub

# Install utils3d
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Install flash-attn
RUN pip install flash-attn==2.7.3

# Install nvdiffrast
RUN mkdir -p /tmp/extensions && \
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast --no-build-isolation && \
    rm -rf /tmp/extensions/nvdiffrast

# Install nvdiffrec
RUN mkdir -p /tmp/extensions && \
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
    pip install /tmp/extensions/nvdiffrec --no-build-isolation && \
    rm -rf /tmp/extensions/nvdiffrec

# Install cumesh
RUN mkdir -p /tmp/extensions && \
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive && \
    pip install /tmp/extensions/CuMesh --no-build-isolation && \
    rm -rf /tmp/extensions/CuMesh

# Install flexgemm
RUN mkdir -p /tmp/extensions && \
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive && \
    pip install /tmp/extensions/FlexGEMM --no-build-isolation && \
    rm -rf /tmp/extensions/FlexGEMM

# Install o-voxel
RUN mkdir -p /tmp/extensions && \
    cp -r o-voxel /tmp/extensions/o-voxel && \
    pip install /tmp/extensions/o-voxel --no-build-isolation && \
    rm -rf /tmp/extensions/o-voxel

# Install FastAPI dependencies
RUN pip install fastapi uvicorn python-multipart optree>=0.13.0

# Replace the verification RUN command with a simpler one
RUN python -c "import torch; import flash_attn; print(f'PyTorch {torch.__version__}, Flash-Attn {flash_attn.__version__}')"

# Add a startup script
COPY <<EOF /app/verify_cuda.sh
#!/bin/bash
set -e  # Exit on any error

check_cuda() {
    echo "Checking CUDA setup..."
    if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
        echo "ERROR: CUDA is not available"
        exit 1
    fi
    
    if ! python -c "import torch; assert torch.version.cuda == '12.4', f'Wrong CUDA version: {torch.version.cuda}'"; then
        echo "ERROR: Wrong CUDA version"
        exit 1
    fi
    
    echo "CUDA setup verified successfully"
}

check_dependencies() {
    echo "Checking dependencies..."
    python -c "
import sys
import torch
import flash_attn
import nvdiffrast
import o_voxel
from trellis2.representations import Mesh, Voxel, MeshWithVoxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline

versions = {
    'PyTorch': torch.__version__,
    'flash_attn': flash_attn.__version__,
}

print('Versions:', versions)

# Verify CUDA setup
assert torch.cuda.is_available(), 'CUDA not available'
assert torch.cuda.device_count() > 0, 'No GPU devices found'
device = torch.cuda.current_device()
print(f'Using GPU: {torch.cuda.get_device_name(device)}')"
}

main() {
    check_cuda
    check_dependencies
    echo "All checks passed successfully!"
}

main
EOF
RUN chmod +x /app/verify_cuda.sh

# Cleanup
RUN conda clean -a -y

# Build argument for HuggingFace token (optional, passed at build time)
ARG HF_TOKEN

# Authenticate with HuggingFace if token provided
RUN if [ -n "$HF_TOKEN" ]; then \
        echo "Authenticating with HuggingFace..."; \
        huggingface-cli login --token $HF_TOKEN --add-to-git-credential; \
    else \
        echo "No HF_TOKEN provided, skipping authentication"; \
    fi

# Pre-download TRELLIS.2 models using huggingface-cli (doesn't require GPU)
RUN if [ -n "$HF_TOKEN" ]; then \
        echo "Pre-downloading TRELLIS.2 models..."; \
        huggingface-cli download microsoft/TRELLIS.2-4B --local-dir /root/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B; \
        huggingface-cli download facebook/dinov3-vitl16-pretrain-lvd1689m --local-dir /root/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m; \
        echo "Models downloaded successfully"; \
    else \
        echo "Skipping model download (no HF_TOKEN)"; \
    fi

# Expose the port that FastAPI runs on
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Modify CMD to run verification first
CMD ["/bin/bash", "-c", "./verify_cuda.sh && python headless_app.py"]

