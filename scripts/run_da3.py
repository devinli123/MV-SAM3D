"""
Depth Anything 3 (DA3) Runner Script

This script runs DA3 on a folder of images and outputs:
- depth maps
- pointmaps (3D coordinates in camera space)
- camera extrinsics and intrinsics
- visualization files (optional)

The outputs can be used as input to MV-SAM3D for improved 3D reconstruction.

IMPORTANT: By default, this script now uses the GLOBAL point cloud from scene.glb
for all views, ensuring geometric consistency across views. This provides higher
quality pointmaps compared to per-view depth maps.

Usage:
    # Default: Use global point cloud from scene.glb (RECOMMENDED)
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example

    # Legacy mode: Use per-view depth maps (NOT recommended)
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example --no_global_pointcloud

    # With custom resolution
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example --process_res 756

    # Without visualization (faster)
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example --no_vis
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

# ============================================================================
# Path setup: DA3 should be a sibling directory to MV-SAM3D
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # MV-SAM3D root
DA3_ROOT = PROJECT_ROOT.parent / "Depth-Anything-3"

if not DA3_ROOT.exists():
    raise FileNotFoundError(
        f"Depth-Anything-3 not found at {DA3_ROOT}. "
        f"Please ensure DA3 is installed as a sibling directory to MV-SAM3D:\n"
        f"  parent_dir/\n"
        f"  ├── MV-SAM3D/\n"
        f"  └── Depth-Anything-3/"
    )

sys.path.insert(0, str(DA3_ROOT / "src"))

# Now we can import DA3
from depth_anything_3.api import DepthAnything3


def depth_to_pointmap(
    depth: np.ndarray, 
    intrinsics: np.ndarray,
) -> np.ndarray:
    """
    Convert depth map to pointmap (3D coordinates in camera space).
    
    NOTE: This outputs in STANDARD CAMERA SPACE (same as MoGe raw output):
        - x: right direction
        - y: down direction  
        - z: forward direction (away from camera, positive depth)
    
    SAM3D's compute_pointmap() will apply the PyTorch3D coordinate transform
    internally, so we should NOT do the transform here.
    
    Args:
        depth: (H, W) depth map, values are distances from camera
        intrinsics: (3, 3) camera intrinsic matrix
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
    
    Returns:
        pointmap: (H, W, 3) point cloud map, each pixel is (x, y, z) coordinate
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Create pixel coordinate grids
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Unproject to 3D (standard camera space)
    # z is positive (depth values are positive, pointing away from camera)
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    pointmap = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    return pointmap


def pointmap_to_sam3d_format(pointmap: np.ndarray) -> np.ndarray:
    """
    Convert pointmap to SAM3D expected format.

    Args:
        pointmap: (H, W, 3) pointmap in PyTorch3D coordinates

    Returns:
        pointmap_sam3d: (3, H, W) pointmap ready for SAM3D
    """
    # SAM3D expects (3, H, W) format (channel-first)
    return pointmap.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)


def project_global_pointcloud_to_view(
    global_points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple,
    use_global_pointcloud: bool = True,
) -> np.ndarray:
    """
    Project global point cloud to a specific view's camera space, generating a dense pointmap.

    This function takes the high-quality global point cloud from DA3's scene.glb and projects
    it into each view's camera space, ensuring all views share the same geometric representation.

    Args:
        global_points: (M, 3) global point cloud in world coordinates
        extrinsic: (3, 4) or (4, 4) camera extrinsic matrix (world-to-camera)
        intrinsic: (3, 3) camera intrinsic matrix
        image_shape: (H, W) target image shape
        use_global_pointcloud: If True, use global points; if False, return None

    Returns:
        pointmap: (H, W, 3) dense pointmap in camera space, or None if disabled
    """
    if not use_global_pointcloud or global_points is None:
        return None

    H, W = image_shape

    # Convert extrinsic to (4, 4) if needed
    if extrinsic.shape == (3, 4):
        extrinsic_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
    else:
        extrinsic_4x4 = extrinsic

    # Transform points from world to camera space
    # global_points: (M, 3), add homogeneous coordinate
    points_homo = np.hstack([global_points, np.ones((global_points.shape[0], 1))])  # (M, 4)
    points_cam_homo = (extrinsic_4x4 @ points_homo.T).T  # (M, 4)
    points_cam = points_cam_homo[:, :3]  # (M, 3)

    # Filter points behind camera
    valid_mask = points_cam[:, 2] > 0
    points_cam = points_cam[valid_mask]

    if len(points_cam) == 0:
        print(f"  Warning: No points in front of camera")
        return np.zeros((H, W, 3), dtype=np.float32)

    # Project to image plane
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    x_img = (points_cam[:, 0] * fx / points_cam[:, 2]) + cx
    y_img = (points_cam[:, 1] * fy / points_cam[:, 2]) + cy

    # Round to nearest pixel first
    u = np.round(x_img).astype(int)
    v = np.round(y_img).astype(int)

    # Filter points within image bounds (AFTER rounding to avoid edge cases)
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_bounds]
    v = v[in_bounds]
    points_cam = points_cam[in_bounds]

    if len(points_cam) == 0:
        print(f"  Warning: No points project within image bounds")
        return np.zeros((H, W, 3), dtype=np.float32)

    # Create dense pointmap using vectorized depth buffer
    pointmap = np.zeros((H, W, 3), dtype=np.float32)
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)

    # Vectorized depth test: for each projected point, keep the closest one
    for i in range(len(points_cam)):
        if points_cam[i, 2] < depth_buffer[v[i], u[i]]:
            depth_buffer[v[i], u[i]] = points_cam[i, 2]
            pointmap[v[i], u[i]] = points_cam[i]

    # Check coverage
    valid_pixels = np.sum(depth_buffer < np.inf)
    total_pixels = H * W
    coverage = valid_pixels / total_pixels * 100
    print(f"    Point cloud coverage: {coverage:.1f}% ({valid_pixels}/{total_pixels} pixels)")

    # If coverage is too low, projection quality is poor
    if coverage < 10:
        print(f"    WARNING: Coverage too low ({coverage:.1f}%), projection may fail")
        print(f"    This view will fallback to depth map")
        return None  # Signal caller to use depth map instead

    # For sparse areas, apply limited inpainting to avoid singular matrices
    # Only fill small holes (1-2 pixel dilation) to maintain geometric diversity
    if coverage < 80:
        try:
            from scipy.ndimage import distance_transform_edt, binary_dilation
            valid_mask = depth_buffer < np.inf

            # Only fill very small holes (1-2 pixel dilation)
            dilated = binary_dilation(valid_mask, iterations=2)
            small_holes = dilated & (~valid_mask)

            if np.sum(small_holes) > 0:
                indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
                pointmap[small_holes] = pointmap[indices[0][small_holes], indices[1][small_holes]]
                filled_pixels = np.sum(small_holes)
                print(f"    Filled {filled_pixels} small holes")
        except Exception as e:
            print(f"    Warning: Failed to fill holes: {e}")

    return pointmap


def load_global_pointcloud_from_scene(scene_glb_path: Path) -> Optional[np.ndarray]:
    """
    Load global point cloud from DA3's scene.glb file.

    Args:
        scene_glb_path: Path to scene.glb

    Returns:
        points: (M, 3) global point cloud in world coordinates, or None if file doesn't exist
    """
    try:
        import trimesh
    except ImportError:
        print("Warning: trimesh not installed, cannot load scene.glb")
        return None

    if not scene_glb_path.exists():
        print(f"Warning: scene.glb not found at {scene_glb_path}")
        return None

    print(f"Loading global point cloud from {scene_glb_path}")
    scene = trimesh.load(str(scene_glb_path))

    # Extract point cloud from scene
    # scene.glb typically contains a point cloud as a mesh or points
    if hasattr(scene, 'vertices'):
        points = np.array(scene.vertices)
    elif hasattr(scene, 'geometry'):
        # It's a Scene object, extract all vertices
        all_vertices = []
        for geom in scene.geometry.values():
            if hasattr(geom, 'vertices'):
                all_vertices.append(np.array(geom.vertices))
        if all_vertices:
            points = np.vstack(all_vertices)
        else:
            print("Warning: No vertices found in scene.glb")
            return None
    else:
        print("Warning: Unexpected scene.glb format")
        return None

    print(f"  Loaded {len(points)} points from scene.glb")
    print(f"  Point cloud bounds: x=[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
          f"y=[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
          f"z=[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

    return points


def extract_object_pointcloud_from_scene(
    scene_glb_path: Path,
    view_masks: List[np.ndarray],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple,
) -> Optional[np.ndarray]:
    """
    Extract object point cloud from DA3's scene.glb using masks from multiple views.

    This function projects the global point cloud to each view, checks which points
    are inside any mask, and returns those points in world coordinates.

    Args:
        scene_glb_path: Path to scene.glb
        view_masks: List of (H, W) binary masks, one per view
        extrinsics: (N, 3, 4) or (N, 4, 4) camera extrinsics
        intrinsics: (N, 3, 3) camera intrinsics
        image_shape: (H, W) image shape

    Returns:
        object_points: (M, 3) object point cloud in world coordinates, or None if failed
    """
    # Load global point cloud
    global_points = load_global_pointcloud_from_scene(scene_glb_path)
    if global_points is None:
        return None

    H, W = image_shape
    N = len(view_masks)

    print(f"\nExtracting object point cloud from {N} views...")
    object_point_mask = np.zeros(len(global_points), dtype=bool)

    for i in range(N):
        mask = view_masks[i]
        extrinsic = extrinsics[i]
        intrinsic = intrinsics[i]

        # Convert extrinsic to (4, 4)
        if extrinsic.shape == (3, 4):
            extrinsic_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
        else:
            extrinsic_4x4 = extrinsic

        # Transform points from world to camera space
        points_homo = np.hstack([global_points, np.ones((global_points.shape[0], 1))])
        points_cam_homo = (extrinsic_4x4 @ points_homo.T).T
        points_cam = points_cam_homo[:, :3]

        # Filter points in front of camera
        front_mask = points_cam[:, 2] > 0

        # Project to image plane
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        x_img = (points_cam[:, 0] * fx / points_cam[:, 2]) + cx
        y_img = (points_cam[:, 1] * fy / points_cam[:, 2]) + cy

        # Round to pixel coordinates
        u = np.round(x_img).astype(int)
        v = np.round(y_img).astype(int)

        # Check which points are in bounds
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & front_mask

        # For points in bounds, check if they're inside the mask
        valid_indices = np.where(in_bounds)[0]
        for idx in valid_indices:
            if mask[v[idx], u[idx]] > 0.5:  # Inside mask
                object_point_mask[idx] = True

        visible_count = np.sum(in_bounds)
        mask_count = np.sum(object_point_mask)
        print(f"  View {i+1}/{N}: {visible_count} points visible, {mask_count} total in object")

    # Extract object points
    object_points = global_points[object_point_mask]

    if len(object_points) == 0:
        print("  WARNING: No points found inside any mask!")
        return None

    print(f"\n  Extracted {len(object_points)} object points from scene.glb")
    print(f"  Object bounds: x=[{object_points[:, 0].min():.3f}, {object_points[:, 0].max():.3f}], "
          f"y=[{object_points[:, 1].min():.3f}, {object_points[:, 1].max():.3f}], "
          f"z=[{object_points[:, 2].min():.3f}, {object_points[:, 2].max():.3f}]")

    return object_points


def project_pointcloud_to_camera(
    world_points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple,
) -> np.ndarray:
    """
    Project world-space point cloud to a specific camera's image plane to create a pointmap.

    Args:
        world_points: (M, 3) points in world coordinates
        extrinsic: (3, 4) or (4, 4) camera extrinsic matrix (world-to-camera)
        intrinsic: (3, 3) camera intrinsic matrix
        image_shape: (H, W) target image shape

    Returns:
        pointmap: (H, W, 3) pointmap in camera space
    """
    H, W = image_shape

    # Convert extrinsic to (4, 4)
    if extrinsic.shape == (3, 4):
        extrinsic_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
    else:
        extrinsic_4x4 = extrinsic

    # Transform from world to camera space
    points_homo = np.hstack([world_points, np.ones((world_points.shape[0], 1))])
    points_cam_homo = (extrinsic_4x4 @ points_homo.T).T
    points_cam = points_cam_homo[:, :3]

    # Filter points in front of camera
    front_mask = points_cam[:, 2] > 0
    points_cam = points_cam[front_mask]

    if len(points_cam) == 0:
        return np.zeros((H, W, 3), dtype=np.float32)

    # Project to image plane
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    x_img = (points_cam[:, 0] * fx / points_cam[:, 2]) + cx
    y_img = (points_cam[:, 1] * fy / points_cam[:, 2]) + cy

    # Round to pixel coordinates
    u = np.round(x_img).astype(int)
    v = np.round(y_img).astype(int)

    # Filter points within bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_bounds]
    v = v[in_bounds]
    points_cam = points_cam[in_bounds]

    # Create pointmap with depth buffering
    pointmap = np.zeros((H, W, 3), dtype=np.float32)
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)

    for i in range(len(points_cam)):
        if points_cam[i, 2] < depth_buffer[v[i], u[i]]:
            depth_buffer[v[i], u[i]] = points_cam[i, 2]
            pointmap[v[i], u[i]] = points_cam[i]

    return pointmap


def run_da3_inference(
    image_dir: str,
    output_dir: str,
    model_path: Optional[str] = None,
    process_res: int = 504,
    save_visualization: bool = True,
    device: str = "cuda",
    use_global_pointcloud: bool = True,
) -> Dict[str, Any]:
    """
    Run DA3 on a folder of images.

    Args:
        image_dir: Path to folder containing input images
        output_dir: Path to output directory
        model_path: Path to DA3 model checkpoint (default: auto-detect)
        process_res: Processing resolution (default: 504)
        save_visualization: Whether to save GLB and depth visualizations
        device: Device to run on ('cuda' or 'cpu')
        use_global_pointcloud: Whether to use global point cloud from scene.glb for all views (default: True)

    Returns:
        Dictionary containing:
            - depth: (N, H, W) depth maps
            - pointmaps: (N, H, W, 3) point cloud maps (from global scene if use_global_pointcloud=True)
            - extrinsics: (N, 3, 4) or (N, 4, 4) camera extrinsics
            - intrinsics: (N, 3, 3) camera intrinsics
            - image_files: List of input image paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect model path if not provided
    if model_path is None:
        # Check common locations
        possible_paths = [
            DA3_ROOT / "checkpoints" / "DA3NESTED-GIANT-LARGE",
            DA3_ROOT / "checkpoints" / "DA3-GIANT-LARGE",
            Path.home() / ".cache" / "huggingface" / "hub" / "models--depth-anything--DA3NESTED-GIANT-LARGE",
        ]
        for p in possible_paths:
            if p.exists():
                model_path = str(p)
                break
        
        if model_path is None:
            # Fall back to HuggingFace model ID
            model_path = "depth-anything/DA3NESTED-GIANT-LARGE"
            print(f"No local model found, will download from HuggingFace: {model_path}")
    
    print(f"Loading DA3 model from: {model_path}")
    model = DepthAnything3.from_pretrained(model_path).to(device)
    
    # Collect images
    image_dir = Path(image_dir)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(ext))
    
    # Sort with natural number ordering (consistent with inference code)
    # This ensures "2.jpg" comes before "10.jpg" for numeric filenames
    def natural_sort_key(path):
        """Sort key that handles numeric filenames correctly."""
        stem = path.stem
        try:
            return (0, int(stem), stem)  # Numeric names first, sorted numerically
        except ValueError:
            return (1, 0, stem)  # Non-numeric names after, sorted alphabetically
    
    image_files = sorted(image_files, key=natural_sort_key)
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_files)} images:")
    for f in image_files:
        print(f"  - {f.name}")
    
    # Build export format
    export_format = "mini_npz"
    if save_visualization:
        export_format += "-glb-depth_vis"
    
    # Run inference
    print(f"\nRunning DA3 inference (process_res={process_res})...")
    prediction = model.inference(
        image=[str(f) for f in image_files],
        process_res=process_res,
        export_dir=str(output_path),
        export_format=export_format,
        show_cameras=True,
    )
    
    # Extract results
    depth = prediction.depth           # (N, H, W)
    extrinsics = prediction.extrinsics # (N, 3, 4) or (N, 4, 4)
    intrinsics = prediction.intrinsics # (N, 3, 3)
    
    print(f"\nDA3 Output:")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"  Extrinsics shape: {extrinsics.shape}")
    print(f"  Intrinsics shape: {intrinsics.shape}")

    # Load global point cloud from scene.glb if requested
    N = depth.shape[0]
    H, W = depth.shape[1], depth.shape[2]
    global_points = None

    if use_global_pointcloud:
        print(f"\n{'='*60}")
        print(f"Using GLOBAL point cloud from scene.glb for all views")
        print(f"{'='*60}")
        scene_glb_path = output_path / "scene.glb"

        # Wait a bit for scene.glb to be written (DA3 writes it asynchronously)
        import time
        max_wait = 10
        for i in range(max_wait):
            if scene_glb_path.exists():
                break
            print(f"  Waiting for scene.glb to be generated... ({i+1}/{max_wait})")
            time.sleep(1)

        global_points = load_global_pointcloud_from_scene(scene_glb_path)

        if global_points is None:
            print(f"  WARNING: Failed to load global point cloud, falling back to per-view depth maps")
            use_global_pointcloud = False
    else:
        print(f"\nUsing per-view depth maps (legacy mode)")

    # Convert depth to pointmaps
    # Two formats:
    # 1. pointmaps: (N, H, W, 3) - standard camera space, for visualization
    # 2. pointmaps_sam3d: (N, 3, H, W) - channel-first format for SAM3D input
    #
    # NOTE: We output in STANDARD CAMERA SPACE (z positive = away from camera)
    # SAM3D's compute_pointmap() applies the PyTorch3D transform internally
    pointmaps = []
    pointmaps_sam3d = []

    print(f"\nGenerating pointmaps for {N} views...")
    for i in range(N):
        if use_global_pointcloud and global_points is not None:
            # Project global point cloud to this view
            print(f"  View {i+1}/{N}: Projecting global point cloud...")
            pm = project_global_pointcloud_to_view(
                global_points,
                extrinsics[i],
                intrinsics[i],
                (H, W),
                use_global_pointcloud=True
            )
            if pm is None:
                # Fallback to depth map if projection failed
                print(f"    Projection failed, using depth map instead")
                pm = depth_to_pointmap(depth[i], intrinsics[i])
        else:
            # Legacy: Convert depth to pointmap (standard camera space, no coordinate transform)
            pm = depth_to_pointmap(depth[i], intrinsics[i])

        pointmaps.append(pm)
        pointmaps_sam3d.append(pointmap_to_sam3d_format(pm))

    pointmaps = np.stack(pointmaps, axis=0)  # (N, H, W, 3)
    pointmaps_sam3d = np.stack(pointmaps_sam3d, axis=0)  # (N, 3, H, W)

    source_description = "from global scene.glb" if use_global_pointcloud else "from per-view depth maps"
    print(f"\n  Pointmaps shape: {pointmaps.shape} (standard camera space, {source_description})")
    print(f"  Pointmaps SAM3D shape: {pointmaps_sam3d.shape} (channel-first for SAM3D)")
    print(f"  Z range: [{pointmaps[:, :, :, 2].min():.4f}, {pointmaps[:, :, :, 2].max():.4f}] (should be positive)")
    
    # Save comprehensive output
    output_file = output_path / "da3_output.npz"
    np.savez(
        output_file,
        depth=depth,                          # (N, H, W)
        pointmaps=pointmaps,                  # (N, H, W, 3) - from global scene or per-view depth
        pointmaps_sam3d=pointmaps_sam3d,      # (N, 3, H, W) - SAM3D format, ready to use
        extrinsics=extrinsics,                # (N, 3, 4) or (N, 4, 4)
        intrinsics=intrinsics,                # (N, 3, 3)
        image_files=np.array([str(f) for f in image_files]),
        process_res=process_res,
        use_global_pointcloud=use_global_pointcloud,  # Flag to indicate source
    )
    print(f"\nResults saved to: {output_file}")
    print(f"  - depth: {depth.shape}")
    print(f"  - pointmaps: {pointmaps.shape} ({source_description})")
    print(f"  - pointmaps_sam3d: {pointmaps_sam3d.shape} (SAM3D format, ready to use)")
    print(f"  - extrinsics: {extrinsics.shape}")
    print(f"  - intrinsics: {intrinsics.shape}")
    print(f"  - use_global_pointcloud: {use_global_pointcloud}")
    
    # Print summary of camera poses
    print(f"\nCamera poses (first 3 views):")
    for i in range(min(3, N)):
        ext = extrinsics[i]
        # Extract rotation and translation
        if ext.shape == (4, 4):
            R, t = ext[:3, :3], ext[:3, 3]
        else:
            R, t = ext[:, :3], ext[:, 3]
        print(f"  View {i}: t = [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    
    return {
        "depth": depth,
        "pointmaps": pointmaps,
        "pointmaps_sam3d": pointmaps_sam3d,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "image_files": [str(f) for f in image_files],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 on a folder of images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (uses global point cloud from scene.glb - RECOMMENDED)
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example

    # Legacy mode (uses per-view depth maps)
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example --no_global_pointcloud

    # Higher resolution
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example --process_res 756

    # Without visualization (faster)
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example --no_vis
        """
    )
    
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True,
        help="Path to folder containing input images"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Path to output directory"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to DA3 model checkpoint (default: auto-detect)"
    )
    parser.add_argument(
        "--process_res", 
        type=int, 
        default=504,
        help="Processing resolution (default: 504)"
    )
    parser.add_argument(
        "--no_vis", 
        action="store_true",
        help="Disable visualization output (GLB, depth_vis)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--no_global_pointcloud",
        action="store_true",
        help="Disable global point cloud from scene.glb (use per-view depth maps instead)"
    )

    args = parser.parse_args()

    run_da3_inference(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        process_res=args.process_res,
        save_visualization=not args.no_vis,
        device=args.device,
        use_global_pointcloud=not args.no_global_pointcloud,
    )


if __name__ == "__main__":
    main()

