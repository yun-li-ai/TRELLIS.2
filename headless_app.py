# Modifications Copyright (c) Applied Intuition, Inc.

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from typing import *
import torch
import numpy as np
import imageio
import time
import cv2
from easydict import EasyDict as edict
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.representations import MeshWithVoxel
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

print(f"Using attention backend: {os.environ.get('ATTN_BACKEND', 'default')}")


class Create3DModelRequestModel(BaseModel):
    image_paths: List[str]
    job_id: str
    seed: int = 0
    randomize_seed: bool = True
    resolution: str = "1024"
    ss_guidance_strength: float = 7.5
    ss_guidance_rescale: float = 0.7
    ss_sampling_steps: int = 12
    ss_rescale_t: float = 5.0
    shape_slat_guidance_strength: float = 7.5
    shape_slat_guidance_rescale: float = 0.5
    shape_slat_sampling_steps: int = 12
    shape_slat_rescale_t: float = 3.0
    tex_slat_guidance_strength: float = 1.0
    tex_slat_guidance_rescale: float = 0.0
    tex_slat_sampling_steps: int = 12
    tex_slat_rescale_t: float = 3.0


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = "/tmp/sensor-sim-trellis-shared/output"
os.makedirs(TMP_DIR, exist_ok=True)


def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """
    Remove files older than max_age_hours from the specified directory.

    Args:
        directory: Directory to clean up.
        max_age_hours: Maximum age of files in hours.
    """
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                    print(f"Removed old file: {filepath}")
                except Exception as e:
                    print(f"Error removing file {filepath}: {e}")


def preprocess_images(images: List[Image.Image]) -> List[Image.Image]:
    """
    Preprocess the input images.

    Args:
        images: List of input images.

    Returns:
        List of preprocessed images.

    Note:
        TRELLIS.2 only supports single-image inference.
        If multiple images are provided, only the first will be used.
    """
    return [pipeline.preprocess_image(images[0])]


def image_to_3d(
    images: List[Image.Image],
    seed: int = 0,
    resolution: str = "1024",
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
) -> MeshWithVoxel:
    """
    Convert image(s) to 3D model.

    Args:
        images: List of input images.
        seed: Random seed.
        resolution: Output resolution ("512", "1024", or "1536").
        ss_guidance_strength: Sparse structure guidance strength.
        ss_guidance_rescale: Sparse structure guidance rescale.
        ss_sampling_steps: Sparse structure sampling steps.
        ss_rescale_t: Sparse structure rescale t.
        shape_slat_guidance_strength: Shape guidance strength.
        shape_slat_guidance_rescale: Shape guidance rescale.
        shape_slat_sampling_steps: Shape sampling steps.
        shape_slat_rescale_t: Shape rescale t.
        tex_slat_guidance_strength: Texture guidance strength.
        tex_slat_guidance_rescale: Texture guidance rescale.
        tex_slat_sampling_steps: Texture sampling steps.
        tex_slat_rescale_t: Texture rescale t.

    Returns:
        The generated 3D mesh.
    """
    # Preprocess images (TRELLIS.2 only supports single image)
    # If multiple images provided, use only the first one
    processed_images = preprocess_images(images[:1])

    # Use first image for generation
    image = processed_images[0]

    # Run the pipeline
    outputs = pipeline.run(
        image,
        seed=seed,
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
        },
        shape_slat_sampler_params={
            "steps": shape_slat_sampling_steps,
            "guidance_strength": shape_slat_guidance_strength,
            "guidance_rescale": shape_slat_guidance_rescale,
            "rescale_t": shape_slat_rescale_t,
        },
        tex_slat_sampler_params={
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
        },
        pipeline_type={
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[resolution],
    )

    mesh = outputs[0]
    mesh.simplify(16777216)  # nvdiffrast limit

    return mesh


def extract_glb(
    mesh: MeshWithVoxel,
    job_id: str,
    decimation_target: int = 1000000,
    texture_size: int = 4096,
) -> str:
    """
    Extract a GLB file from the 3D model.

    Args:
        mesh: The mesh to export.
        job_id: Job ID for organizing output files.
        decimation_target: Target face count for decimation.
        texture_size: Texture resolution.

    Returns:
        Path to the exported GLB file.
    """
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=mesh.grid_size if hasattr(mesh, "grid_size") else 1024,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=False,
    )

    # Generate filename using job_id directory for consistency with TRELLIS
    glb_path = os.path.join(TMP_DIR, job_id, "model.glb")
    glb.export(glb_path, extension_webp=True)

    return glb_path


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}


@app.post("/create-3d-model-from-paths")
async def create_3d_model_from_paths(request: Create3DModelRequestModel):
    """
    Create a 3D model from image(s).

    Args:
        request: Request model with image paths and parameters.

    Returns:
        Dictionary with paths to generated files.
    """
    job_id = request.job_id
    try:
        # Create job-specific directory
        job_dir = os.path.join(TMP_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        # Cleanup old files periodically
        cleanup_old_files(TMP_DIR)

        # Get seed
        seed = (
            np.random.randint(0, MAX_SEED) if request.randomize_seed else request.seed
        )

        # Load images
        errors = []
        images = []
        output_image_paths = []
        for idx, image_path in enumerate(request.image_paths):
            if not os.path.exists(image_path):
                error_msg = f"File not found: {image_path}"
                errors.append(error_msg)
                continue

            try:
                image = Image.open(image_path)
                images.append(image)

                # Save a copy of the input image to the job directory
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                saved_image_path = os.path.join(job_dir, f"{base_name}.png")
                image.save(saved_image_path)
                output_image_paths.append(saved_image_path)
            except Exception as e:
                error_msg = f"Error opening file {image_path}: {str(e)}"
                errors.append(error_msg)

        # Check if we have any valid images
        if not images:
            return {"error": "No valid images found", "details": errors}

        # TRELLIS.2 only supports single-image inference
        if len(images) > 1:
            print(
                f"Warning: Multiple images provided ({len(images)}), but TRELLIS.2 only supports single-image inference. Using only the first image."
            )

        # Generate 3D model
        mesh = image_to_3d(
            images=images,
            seed=seed,
            resolution=request.resolution,
            ss_guidance_strength=request.ss_guidance_strength,
            ss_guidance_rescale=request.ss_guidance_rescale,
            ss_sampling_steps=request.ss_sampling_steps,
            ss_rescale_t=request.ss_rescale_t,
            shape_slat_guidance_strength=request.shape_slat_guidance_strength,
            shape_slat_guidance_rescale=request.shape_slat_guidance_rescale,
            shape_slat_sampling_steps=request.shape_slat_sampling_steps,
            shape_slat_rescale_t=request.shape_slat_rescale_t,
            tex_slat_guidance_strength=request.tex_slat_guidance_strength,
            tex_slat_guidance_rescale=request.tex_slat_guidance_rescale,
            tex_slat_sampling_steps=request.tex_slat_sampling_steps,
            tex_slat_rescale_t=request.tex_slat_rescale_t,
        )

        # Render video
        video_path = os.path.join(job_dir, "preview.mp4")
        video = render_utils.make_pbr_vis_frames(
            render_utils.render_video(mesh, envmap=envmap)
        )
        imageio.mimsave(video_path, video, fps=15)

        # Extract GLB
        glb_path = extract_glb(mesh, job_id)

        # Clean up GPU memory
        torch.cuda.empty_cache()

        return {
            "job_id": request.job_id,
            "preview_video": video_path,
            "glb_file": glb_path,
            "output_images": output_image_paths,
            "errors": errors,  # Include any errors encountered
        }

    except Exception as e:
        import traceback

        error_msg = traceback.format_exc()
        print(f"Error in create_3d_model: {error_msg}")
        return {"success": False, "error": str(e), "traceback": error_msg}


@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """
    Download a generated file.

    Args:
        job_id: Job ID for the file.
        filename: Name of the file to download (e.g., 'model.glb', 'preview.mp4').

    Returns:
        FileResponse with the requested file.
    """
    filepath = os.path.join(TMP_DIR, job_id, filename)
    if not os.path.exists(filepath):
        return {"error": f"File not found: {job_id}/{filename}"}
    return FileResponse(filepath, filename=f"{job_id}_{filename}")


if __name__ == "__main__":
    import uvicorn

    # Load pipeline
    print("Loading TRELLIS.2 pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    print("Pipeline loaded successfully")

    # Load environment map
    print("Loading environment map...")
    envmap = EnvMap(
        torch.tensor(
            cv2.cvtColor(
                cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED),
                cv2.COLOR_BGR2RGB,
            ),
            dtype=torch.float32,
            device="cuda",
        )
    )
    print("Environment map loaded successfully")

    # Run server
    print("Starting server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
