# TRELLIS.2 Docker Setup

This document describes the Docker setup created for TRELLIS.2, based on the original TRELLIS implementation.

## Quick Start

### Build with Pre-downloaded Models (Recommended)

```bash
# Build with HuggingFace token to pre-download models
docker build \
  --build-arg HF_TOKEN="your_huggingface_token" \
  -t quay.io/applied_dev/applied2:trellis2-1.0.0 \
  .

# Run without needing token at runtime
docker run -d --gpus all -p 8000:8000 \
  --name trellis2-app \
  -v /tmp/sensor-sim-trellis2-shared:/tmp/sensor-sim-trellis2-shared \
  quay.io/applied_dev/applied2:trellis2-1.0.0
```

### Prerequisites

1. **HuggingFace Account**: Create account at https://huggingface.co
2. **Get Access Token**: https://huggingface.co/settings/tokens
3. **Request Access to Gated Models**:
   - https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
   - https://huggingface.co/briaai/RMBG-2.0
   - Note: Approval may take a few minutes

## Files Created

### 1. Dockerfile
Main Dockerfile for building the complete TRELLIS.2 image from scratch.

**Key differences from TRELLIS Dockerfile:**
- Uses PyTorch 2.6.0 with CUDA 12.4 (upgraded from 2.4.0 with CUDA 11.8)
- Uses `flash_attn` instead of `xformers` as the attention backend
- Removes Kaolin, spconv, and gaussian rasterization dependencies (not needed for TRELLIS.2)
- Adds TRELLIS.2-specific dependencies:
  - `utils3d` for 3D utilities
  - `pillow-simd` for faster image processing
  - `kornia` and `timm` for image feature extraction
  - `nvdiffrast` and `nvdiffrec` for rendering
  - `cumesh` for mesh processing
  - `flexgemm` for efficient matrix operations
  - `o-voxel` for voxel processing and GLB export
  - `libwebp-dev` for WebP texture compression support
- Supports optional HuggingFace authentication and model pre-downloading
- Updates TORCH_CUDA_ARCH_LIST to include architecture 9.0
- Changes temporary directory to `/tmp/Trellis2-demo`

### 2. Dockerfile.update_app
Lightweight Dockerfile for updating the application code without rebuilding dependencies.

**Usage:**
```bash
# Build updated image
docker build -f Dockerfile.update_app -t quay.io/applied_dev/applied2:trellis2-1.0.1 .

# Push to registry
docker login --username=<quay username> --password=<encrypted password> quay.io
docker push quay.io/applied_dev/applied2:trellis2-1.0.1
```

### 3. headless_app.py
FastAPI-based REST API service for TRELLIS.2.

**Endpoints:**
- `GET /health` - Health check endpoint
- `POST /create-3d-model-from-paths` - Generate 3D model from image(s)
- `GET /download/{filename}` - Download generated files

**Key features:**
- Supports all TRELLIS.2 pipeline parameters (resolution, guidance strengths, sampling steps, etc.)
- Generates both GLB files and render videos
- Automatic cleanup of old files (24 hour retention)
- CORS enabled for cross-origin requests
- GPU memory management
- Single-image inference (TRELLIS.2 limitation - if multiple images provided, uses first one)

## Building the Docker Image

### Option 1: Build with Pre-downloaded Models (Recommended)

This pre-downloads ~7-8 GB of models during build, resulting in faster startup and no runtime authentication needed.

```bash
cd /home/applied-owner/Documents/code/TRELLIS.2

docker build \
  --build-arg HF_TOKEN="hf_your_token_here" \
  -t quay.io/applied_dev/applied2:trellis2-1.0.0 \
  .
```

**Benefits:**
- ✅ No HuggingFace token needed at runtime
- ✅ Faster startup (models already cached)
- ✅ Works offline after initial build
- ✅ Simpler run command

**Note:** Build time will be ~30-40 minutes due to model downloads.

### Option 2: Build without Pre-downloading (Faster build)

```bash
docker build -t quay.io/applied_dev/applied2:trellis2-1.0.0 .
```

**Note:** You'll need to provide HF_TOKEN at runtime and wait for model downloads on first run.

## Running the Container

### If Built with Pre-downloaded Models

```bash
# Simple run
docker run -d --gpus all -p 8000:8000 \
  --name trellis2-app \
  -v /tmp/sensor-sim-trellis2-shared:/tmp/sensor-sim-trellis2-shared \
  quay.io/applied_dev/applied2:trellis2-1.0.0
```

### If Built without Pre-downloading

```bash
docker run -d --gpus all -p 8000:8000 \
  --name trellis2-app \
  -e HF_TOKEN="your_token" \
  -v /tmp/sensor-sim-trellis2-shared:/tmp/sensor-sim-trellis2-shared \
  quay.io/applied_dev/applied2:trellis2-1.0.0 \
  bash -c "huggingface-cli login --token \$HF_TOKEN && python headless_app.py"
```

### Production Setup

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  --name trellis2-prod \
  --restart unless-stopped \
  -v /tmp/sensor-sim-trellis2-shared:/tmp/sensor-sim-trellis2-shared \
  quay.io/applied_dev/applied2:trellis2-1.0.0
```

### Development Setup (Mount Code)

```bash
docker run -d --gpus all -p 8000:8000 \
  --name trellis2-dev \
  -v /tmp/sensor-sim-trellis2-shared:/tmp/sensor-sim-trellis2-shared \
  -v /home/applied-owner/Documents/code/TRELLIS.2/headless_app.py:/app/headless_app.py \
  -v /home/applied-owner/Documents/code/TRELLIS.2/assets:/app/assets \
  quay.io/applied_dev/applied2:trellis2-1.0.0
```

## Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Create 3D model
curl -X POST http://localhost:8000/create-3d-model-from-paths \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": ["/path/to/image.png"],
    "job_id": "test-job-1",
    "seed": 42,
    "randomize_seed": false,
    "resolution": "1024"
  }'
```

## Managing the Container

```bash
# View logs
docker logs -f trellis2-app

# Check status
docker ps

# Stop container
docker stop trellis2-app

# Start container
docker start trellis2-app

# Restart container
docker restart trellis2-app

# Remove container
docker stop trellis2-app && docker rm trellis2-app
```

## Environment Variables

The following environment variables are set in the container:
- `PYTHONPATH=/app` - Python module search path
- `ATTN_BACKEND=flash_attn` - Use Flash Attention for efficiency
- `SPCONV_ALGO=native` - Sparse convolution algorithm
- `TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"` - Supported CUDA architectures
- `CXX=/usr/local/bin/gxx-wrapper` - Custom C++ compiler wrapper for CUDA
- `OPENCV_IO_ENABLE_OPENEXR=1` - Enable OpenEXR support in OpenCV
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - GPU memory optimization

## Port Configuration

- Port 8000: FastAPI REST API service

## Volume Mounts

- `/tmp/sensor-sim-trellis2-shared/output` - Output directory for generated files

## Healthcheck

The container includes a healthcheck that:
- Runs every 30 seconds
- Has a 30-second timeout
- Waits 5 seconds before first check
- Retries 3 times before marking as unhealthy
- Checks the `/health` endpoint

## Differences from TRELLIS

1. **Model Architecture**: TRELLIS.2 uses Octree representations instead of Gaussian splatting
2. **Pipeline**: Uses `Trellis2ImageTo3DPipeline` with separate stages for sparse structure, shape, and texture
3. **Output Format**: Generates PBR materials with proper texture maps (basecolor, metallic, roughness, normal)
4. **Dependencies**: Different set of CUDA extensions optimized for TRELLIS.2
5. **API Parameters**: More granular control over each generation stage
6. **Multi-Image**: TRELLIS.2 does NOT support multi-image inference (only uses first image if multiple provided)

## System Requirements

- **GPU**: NVIDIA GPU with CUDA compute capability 7.0 or higher
- **GPU Memory**: 
  - Minimum 16GB for 512/1024 resolution
  - 24GB+ recommended for 1536 resolution
- **Disk Space**: ~20GB for image with pre-downloaded models (~13GB without)
- **NVIDIA Driver**: Compatible with CUDA 12.4

## Troubleshooting

### GPU Not Accessible

```bash
# Install nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Container Unhealthy

```bash
# Check logs for errors
docker logs trellis2-app

# Common issues:
# - Missing HuggingFace token (if models not pre-downloaded)
# - Insufficient GPU memory
# - Port 8000 already in use
```

### Models Not Found (Runtime Error)

If you didn't pre-download models during build, you need to either:
1. Rebuild with `--build-arg HF_TOKEN="..."`
2. Provide token at runtime (see "If Built without Pre-downloading" section)

## Docker Image Tagging

```bash
# Tag image
docker tag quay.io/applied_dev/applied2:trellis2-1.0.0 quay.io/applied_dev/applied2:trellis2-latest

# Push to registry
docker login quay.io
docker push quay.io/applied_dev/applied2:trellis2-1.0.0
docker push quay.io/applied_dev/applied2:trellis2-latest
```

## Notes

- WebP texture compression is supported (requires `libwebp-dev` in Dockerfile)
- Models cached in `/root/.cache/huggingface` inside container
- First run without pre-downloaded models downloads ~7-8 GB
- GLB export uses WebP compression for smaller file sizes (2-5 MB vs 4-8 MB with JPEG)
