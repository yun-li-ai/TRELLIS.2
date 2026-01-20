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
    resolution: str = "1024"  # 512, 1024, 1536
    decimation_target: int = (
        300000  # Target face count for mesh simplification (100000-500000 recommended)
    )
    texture_size: int = 4096  # 1024, 2048, 4096
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

    TRELLIS.2 uses a three-stage generation pipeline:
    1. **Sparse Structure Generation**: Creates a sparse 3D voxel occupancy map (binary mask indicating
       which 3D locations contain geometry). This is a "field-free" representation - only occupied voxel
       coordinates are stored, making it memory-efficient. Think of it as a 3D skeleton/scaffold that
       defines where the object exists in 3D space.
    2. **Shape Generation**: Generates the detailed 3D mesh geometry (vertices, faces) within the
       sparse structure coordinates.
    3. **Texture Generation**: Adds PBR materials and textures to the generated mesh.

    **What is Classifier-Free Guidance (CFG)?**
    Guidance is a technique used in diffusion models to control how strongly the generation follows
    the input conditioning (your image) versus generating unconditionally. The model makes two predictions:
    - **Conditioned prediction**: What the model thinks should be generated given your input image.
    - **Unconditioned prediction**: What the model would generate without any input (pure noise/random).

    The final output is a weighted combination:
        output = guidance_strength × conditioned + (1 - guidance_strength) × unconditioned

    - **guidance_strength = 1.0**: No guidance, uses only conditioned prediction (but may be less stable).
    - **guidance_strength > 1.0**: Amplifies the difference between conditioned and unconditioned, making
      the output more faithful to the input image. Higher values = stronger adherence to input.
    - **guidance_rescale**: Rescales the guidance output to match the statistics of unconditioned predictions,
      preventing artifacts from overly strong guidance (like over-saturation or unrealistic details).

    Args:
        images: List of input images. TRELLIS.2 only supports single-image inference, so if multiple
            images are provided, only the first one will be used.
        seed: Random seed for controlling the stochastic generation process (default: 0).
            The seed initializes PyTorch's random number generator (`torch.manual_seed(seed)`), providing
            reproducible results when the same seed is used with the same input image and parameters.
            - Same seed + same image + same parameters = very similar output (mostly deterministic).
            - Different seeds = different variations of the 3D model (useful for exploring options).
            - Set randomize_seed=True in the request to automatically use a random seed for each generation.
            Important notes on reproducibility:
            - Outputs are highly reproducible but may not be bit-identical due to:
              * CUDA operations can have non-deterministic behavior (especially reduction operations)
              * Mesh simplification algorithms may have slight variations
              * Floating point precision differences across GPU architectures/CUDA versions
            - For maximum reproducibility, use the same GPU model, CUDA version, and PyTorch version.
            - The generated geometry and overall structure will be very similar, but vertex positions
              may have minor floating-point differences (< 1e-6 typically).
        resolution: Output resolution for the 3D model generation (default: "1024", options: "512", "1024", "1536").
            Controls the internal voxel grid resolution used during generation (affects detail quality, not final vertex count).
            - **"512"**: Fastest generation, lower detail. Good for quick previews or simple objects.
                **Guaranteed**: Uses dedicated 512-resolution models, always produces 512 resolution output.
            - **"1024"**: Balanced quality and speed (recommended). Good detail for most use cases.
                **Mostly guaranteed**: Uses cascade pipeline (512→1024) targeting 1024. May be reduced if geometry
                complexity exceeds token limits (max_num_tokens=49152), but minimum is 1024.
            - **"1536"**: Highest quality, slowest generation. Best for final production models with fine details.
                **Best effort**: Uses cascade pipeline (512→1024→1536) targeting 1536 resolution. However, if the
                generated geometry exceeds token limits (max_num_tokens=49152), the resolution may be automatically reduced
                in 128-pixel increments down to a minimum of 1024. A warning message will be printed if reduction occurs.
            Important notes:
            - Resolution affects internal voxel grid detail and texture quality, but NOT the final vertex/face count.
            - Final mesh vertex/face count is controlled by `decimation_target` parameter (default: 300,000, range: 100,000-500,000 faces).
            - The mesh is simplified to nvdiffrast limit (16M faces) first, then decimated to target face count.
            - Higher resolution = more detail in the generated geometry before decimation, but final mesh size is similar.
            - Check the generated mesh's `grid_size` attribute for the actual internal resolution used.

        # Sparse Structure (SS) Parameters - Control the initial 3D structure generation.
        ss_guidance_strength: Classifier-free guidance strength for sparse structure generation (default: 7.5).
            Controls how strongly the model follows the input image conditioning vs. unconditional generation.
            - Higher values (7-10): Stronger adherence to input image, more deterministic output.
            - Lower values (1-3): More creative variation, less faithful to input.
            - Value of 1.0: No guidance (unconditional generation).
        ss_guidance_rescale: Guidance rescale factor for sparse structure (default: 0.7, range: 0.0-1.0).
            Rescales the guidance output to match the standard deviation of unconditional predictions.
            Helps prevent over-saturation and artifacts from high guidance strength.
            - 0.0: No rescaling (standard CFG).
            - 0.5-0.8: Recommended range for stable generation.
            - 1.0: Full rescaling (may reduce guidance effectiveness).
        ss_sampling_steps: Number of diffusion steps for sparse structure generation (default: 12).
            More steps = higher quality but slower generation.
            - 8-12: Fast, good quality (recommended).
            - 20-50: Higher quality, slower.
        ss_rescale_t: Time rescaling factor for sparse structure sampling schedule (default: 5.0).
            Rescales the diffusion timestep schedule to focus more on early or late denoising steps.
            - Lower values (1.0-3.0): More uniform timestep distribution.
            - Higher values (5.0-6.0): Focuses more on early denoising (recommended for structure).
            Affects the balance between structure detail and overall coherence.

        # Shape SLat (Structured Latent) Parameters - Control the 3D mesh shape generation.
        shape_slat_guidance_strength: Classifier-free guidance strength for shape generation (default: 7.5).
            Similar to ss_guidance_strength but specifically for the 3D mesh shape.
            Controls how closely the generated shape matches the input image's geometry.
            - Higher values: More faithful to input image geometry.
            - Lower values: More shape variation and creativity.
        shape_slat_guidance_rescale: Guidance rescale factor for shape (default: 0.5, range: 0.0-1.0).
            Prevents shape artifacts from high guidance strength.
            - 0.0-0.5: Standard range for stable shape generation.
        shape_slat_sampling_steps: Number of diffusion steps for shape generation (default: 12).
            More steps improve shape quality but increase generation time.
        shape_slat_rescale_t: Time rescaling factor for shape sampling schedule (default: 3.0).
            Lower than ss_rescale_t because shape benefits from more uniform denoising.
            - 2.0-4.0: Recommended range for balanced shape generation.

        # Texture SLat Parameters - Control the texture/material generation.
        tex_slat_guidance_strength: Classifier-free guidance strength for texture generation (default: 1.0).
            Lower default value because textures benefit from more variation.
            - 1.0: Minimal guidance (allows natural texture variation).
            - 3.0-5.0: Moderate guidance (more faithful to input colors/materials).
            - 7.0+: Strong guidance (may reduce texture diversity).
        tex_slat_guidance_rescale: Guidance rescale factor for texture (default: 0.0).
            Usually set to 0.0 because texture guidance is already low.
            - 0.0: No rescaling (standard for textures).
            - 0.3-0.5: Can help if using higher guidance_strength.
        tex_slat_sampling_steps: Number of diffusion steps for texture generation (default: 12).
            Similar to other sampling steps - more steps = better quality but slower.
        tex_slat_rescale_t: Time rescaling factor for texture sampling schedule (default: 3.0).
            Similar to shape_slat_rescale_t for balanced texture generation.

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
    decimation_target: int = 300000,  # Target face count for mesh simplification (100000-500000 recommended)
    texture_size: int = 4096,  # Texture resolution in pixels (1024, 2048, or 4096)
) -> str:
    """
    Extract a GLB file from the 3D model.

    Args:
        mesh: The mesh to export.
        job_id: Job ID for organizing output files.
        decimation_target: Target number of faces for mesh simplification (default: 300,000, range: 100,000-500,000).
            This parameter controls the final polygon count of the exported mesh. The decimation process:
            1. First simplifies aggressively to 3× target (900,000 faces by default) to remove fine details
            2. Cleans up topology (removes duplicates, repairs non-manifold edges, fills holes)
            3. Simplifies again to the exact target face count
            4. Performs final cleanup
            Lower values (100,000-200,000): Smaller file size, faster loading, less detail.
            Medium values (200,000-400,000): Balanced quality and file size (recommended).
            Higher values (400,000-500,000): Larger file size, more geometric detail preserved.
            Note: The actual vertex count will be approximately half the face count (for triangular meshes).
        texture_size: Texture resolution (default: 2048). Size of the baked texture maps in pixels.
            Options: 1024, 2048, 4096. Higher values = sharper textures but larger file size.

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

    # Move pivot to bottom of the asset (matching TRELLIS behavior)
    # Apply AFTER GLB creation to preserve texture/attribute mapping
    # NOTE: o_voxel.postprocess.to_glb() already converts Z-up to Y-up with inverted Y
    # So we need to adjust Y axis (index 1), not Z axis
    vertices = glb.vertices
    min_y = vertices[:, 1].min()
    vertices[:, 1] -= min_y
    glb.vertices = vertices

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
