"""
SAM 3D Objects Weighted Inference Script

This script extends the standard inference with attention-based weighted fusion.
Instead of simple averaging across views, it uses attention entropy to determine
per-latent fusion weights.

Key features:
    - Per-latent weighting based on attention entropy
    - Configurable weighting parameters (alpha, layer, step)
    - Optional visualization of weights and entropy
    - Extensible architecture for adding new confidence factors
    - Support for external pointmaps from DA3 (Depth Anything 3)
    - GLB merge visualization (SAM3D output + DA3 scene)

Usage:
    # Basic weighted inference (both Stage 1 and Stage 2 weighted by default)
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3
    
    # Disable all weighting (simple average for both stages)
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 \
        --no_stage1_weighting --no_stage2_weighting
    
    # Custom Stage 1 parameters
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 \
        --stage1_entropy_alpha 80.0 --stage1_entropy_layer 9
    
    # Custom Stage 2 parameters
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 \
        --stage2_entropy_alpha 80.0 --stage2_attention_layer 6
    
    # Use visibility weighting for Stage 2 (requires DA3)
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 \
        --da3_output ./da3_outputs/example/da3_output.npz --stage2_weight_source visibility
"""
import sys
import argparse
from pathlib import Path
import subprocess
import tempfile
import shutil
from typing import List, Optional
from datetime import datetime
import numpy as np
import torch
from loguru import logger

# Import inference code
sys.path.append("notebook")
from inference import Inference
from load_images_and_masks import load_images_and_masks_from_path

from sam3d_objects.utils.cross_attention_logger import CrossAttentionLogger
from sam3d_objects.utils.latent_weighting import WeightingConfig, LatentWeightManager
from sam3d_objects.utils.coordinate_transforms import (
    Z_UP_TO_Y_UP,
    apply_sam3d_pose_to_mesh_vertices,
    apply_sam3d_pose_to_latent_coords,
    canonical_to_pytorch3d,
)
from pytorch3d.transforms import Transform3d, quaternion_to_matrix

# Import DA3 utilities
sys.path.append("scripts")
from run_da3 import extract_object_pointcloud_from_scene, project_pointcloud_to_camera


def merge_glb_with_da3_aligned(
    sam3d_glb_path: Path, 
    da3_output_dir: Path,
    sam3d_pose: dict,
    output_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Merge SAM3D reconstructed object with DA3's full scene GLB (aligned).
    
    DA3's scene.glb contains alignment matrix `hf_alignment` in metadata:
    A = T_center @ M @ w2c0, which includes:
    - w2c0: First frame's world-to-camera transform
    - M: CV -> glTF coordinate system transform
    - T_center: Centering translation
    
    SAM3D object transform chain:
    1. canonical (Z-up) -> Y-up rotation
    2. Apply SAM3D pose -> PyTorch3D camera space
    3. PyTorch3D -> CV camera space: (-x, -y, z) -> (x, y, z)
    4. Apply DA3's alignment matrix A (from scene.glb metadata)
    
    Args:
        sam3d_glb_path: SAM3D output GLB file path (canonical space)
        da3_output_dir: DA3 output directory containing scene.glb
        sam3d_pose: SAM3D pose parameters {'scale', 'rotation', 'translation'}
        output_path: Output path
    
    Returns:
        Aligned GLB file path
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed, cannot merge GLB files")
        return None
    
    if not sam3d_glb_path.exists():
        logger.warning(f"SAM3D GLB not found: {sam3d_glb_path}")
        return None
    
    # Find DA3's scene.glb
    da3_scene_glb = da3_output_dir / "scene.glb"
    da3_npz = da3_output_dir / "da3_output.npz"
    
    if not da3_scene_glb.exists():
        logger.warning(f"DA3 scene.glb not found: {da3_scene_glb}")
        logger.warning("Please run DA3 with visualization enabled")
        return None
    
    if output_path is None:
        output_path = sam3d_glb_path.parent / f"{sam3d_glb_path.stem}_merged_scene.glb"
    
    try:
        # Load SAM3D GLB
        sam3d_scene = trimesh.load(str(sam3d_glb_path))
        
        # Load DA3 scene.glb
        da3_scene = trimesh.load(str(da3_scene_glb))
        
        # Try to read alignment matrix from DA3 scene metadata
        alignment_matrix = None
        if hasattr(da3_scene, 'metadata') and da3_scene.metadata is not None:
            alignment_matrix = da3_scene.metadata.get('hf_alignment', None)
        
        if alignment_matrix is None:
            logger.warning("DA3 scene.glb does not contain alignment matrix (hf_alignment)")
            logger.warning("Falling back to computing alignment from extrinsics")
            
            # Fallback: compute alignment from extrinsics
            if not da3_npz.exists():
                logger.warning(f"DA3 da3_output.npz not found: {da3_npz}")
                return None
            
            da3_data = np.load(da3_npz)
            da3_extrinsics = da3_data["extrinsics"]
            
            # Get first frame w2c
            w2c0 = da3_extrinsics[0]
            if w2c0.shape == (3, 4):
                w2c0_44 = np.eye(4, dtype=np.float64)
                w2c0_44[:3, :4] = w2c0
                w2c0 = w2c0_44
            
            # CV -> glTF coordinate transform
            M_cv_to_gltf = np.eye(4, dtype=np.float64)
            M_cv_to_gltf[1, 1] = -1.0
            M_cv_to_gltf[2, 2] = -1.0
            
            # Compute alignment matrix (without centering)
            A_no_center = M_cv_to_gltf @ w2c0
            
            # Get point cloud center from DA3 scene
            da3_points = []
            if isinstance(da3_scene, trimesh.Scene):
                for geom in da3_scene.geometry.values():
                    if hasattr(geom, 'vertices'):
                        da3_points.append(geom.vertices)
            elif hasattr(da3_scene, 'vertices'):
                da3_points.append(da3_scene.vertices)
            
            if da3_points:
                all_pts = np.vstack(da3_points)
                # DA3 scene is already centered
                # Since it is centered, center should be near 0
                # We need to compute original centering offset
                # This is complex, assume centering offset is 0 for now
                alignment_matrix = A_no_center
                logger.warning("Using alignment without centering (may be slightly off)")
        
        logger.info(f"[Merge Scene] Alignment matrix:\n{alignment_matrix}")
        
        # Extract SAM3D pose parameters
        scale = sam3d_pose.get('scale', np.array([1.0, 1.0, 1.0]))
        rotation_quat = sam3d_pose.get('rotation', np.array([1.0, 0.0, 0.0, 0.0]))  # wxyz
        translation = sam3d_pose.get('translation', np.array([0.0, 0.0, 0.0]))
        
        if len(scale.shape) > 1:
            scale = scale.flatten()
        if len(rotation_quat.shape) > 1:
            rotation_quat = rotation_quat.flatten()
        if len(translation.shape) > 1:
            translation = translation.flatten()
        
        logger.info(f"[Merge Scene] SAM3D pose:")
        logger.info(f"  scale: {scale}")
        logger.info(f"  rotation (wxyz): {rotation_quat}")
        logger.info(f"  translation: {translation}")
        
        # ========================================
        # SAM3D object transform
        # ========================================
        
        # Z-up to Y-up rotation matrix
        z_up_to_y_up_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=np.float32)
        
        # Build pose transform in PyTorch3D space
        quat_tensor = torch.tensor(rotation_quat, dtype=torch.float32).unsqueeze(0)
        R_sam3d = quaternion_to_matrix(quat_tensor)
        scale_tensor = torch.tensor(scale, dtype=torch.float32).reshape(1, -1)
        if scale_tensor.shape[-1] == 1:
            scale_tensor = scale_tensor.repeat(1, 3)
        translation_tensor = torch.tensor(translation, dtype=torch.float32).reshape(1, 3)
        pose_transform = (
            Transform3d(dtype=torch.float32)
            .scale(scale_tensor)
            .rotate(R_sam3d)
            .translate(translation_tensor)
        )
        
        # PyTorch3D to CV camera space transform
        p3d_to_cv = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        
        def transform_sam3d_to_da3_space(vertices):
            """
            Transform SAM3D canonical space vertices to DA3 scene space (glTF)
            """
            # Step 1: Z-up to Y-up
            v_rotated = vertices @ z_up_to_y_up_matrix.T
            
            # Step 2: Apply SAM3D pose -> PyTorch3D space
            pts_local = torch.from_numpy(v_rotated).float().unsqueeze(0)
            pts_p3d = pose_transform.transform_points(pts_local).squeeze(0).numpy()
            
            # Step 3: PyTorch3D -> CV camera space
            pts_cv = pts_p3d @ p3d_to_cv.T
            
            # Step 4: Apply DA3 alignment matrix
            pts_final = trimesh.transform_points(pts_cv, alignment_matrix)
            
            return pts_final
        
        # ========================================
        # Create merged scene
        # ========================================
        
        merged_scene = trimesh.Scene()
        
        # Add DA3 scene (keep as-is, already in correct coordinate system)
        if isinstance(da3_scene, trimesh.Scene):
            for name, geom in da3_scene.geometry.items():
                merged_scene.add_geometry(geom.copy(), node_name=f"da3_{name}")
        else:
            merged_scene.add_geometry(da3_scene.copy(), node_name="da3_scene")
        
        # Transform and add SAM3D object
        sam3d_vertices_final = None
        if isinstance(sam3d_scene, trimesh.Scene):
            for name, geom in sam3d_scene.geometry.items():
                if hasattr(geom, 'vertices'):
                    geom_copy = geom.copy()
                    geom_copy.vertices = transform_sam3d_to_da3_space(geom_copy.vertices)
                    merged_scene.add_geometry(geom_copy, node_name=f"sam3d_{name}")
                    if sam3d_vertices_final is None:
                        sam3d_vertices_final = geom_copy.vertices
                else:
                    merged_scene.add_geometry(geom, node_name=f"sam3d_{name}")
        else:
            if hasattr(sam3d_scene, 'vertices'):
                sam3d_scene_copy = sam3d_scene.copy()
                sam3d_scene_copy.vertices = transform_sam3d_to_da3_space(sam3d_scene_copy.vertices)
                sam3d_vertices_final = sam3d_scene_copy.vertices
                merged_scene.add_geometry(sam3d_scene_copy, node_name="sam3d_object")
            else:
                merged_scene.add_geometry(sam3d_scene.copy(), node_name="sam3d_object")
        
        # Print alignment info
        if sam3d_vertices_final is not None:
            logger.info(f"[Merge Scene] SAM3D object in DA3 space:")
            logger.info(f"  X: [{sam3d_vertices_final[:, 0].min():.4f}, {sam3d_vertices_final[:, 0].max():.4f}]")
            logger.info(f"  Y: [{sam3d_vertices_final[:, 1].min():.4f}, {sam3d_vertices_final[:, 1].max():.4f}]")
            logger.info(f"  Z: [{sam3d_vertices_final[:, 2].min():.4f}, {sam3d_vertices_final[:, 2].max():.4f}]")
        
        # Print DA3 scene bounds
        da3_pts = []
        if isinstance(da3_scene, trimesh.Scene):
            for geom in da3_scene.geometry.values():
                if hasattr(geom, 'vertices'):
                    da3_pts.append(geom.vertices)
        if da3_pts:
            da3_all = np.vstack(da3_pts)
            logger.info(f"[Merge Scene] DA3 scene bounds:")
            logger.info(f"  X: [{da3_all[:, 0].min():.4f}, {da3_all[:, 0].max():.4f}]")
            logger.info(f"  Y: [{da3_all[:, 1].min():.4f}, {da3_all[:, 1].max():.4f}]")
            logger.info(f"  Z: [{da3_all[:, 2].min():.4f}, {da3_all[:, 2].max():.4f}]")
        
        # Export
        merged_scene.export(str(output_path))
        logger.info(f"[Merge Scene] Saved merged GLB: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.warning(f"Failed to merge GLB files: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_multiview_pose_consistency(
    sam3d_glb_path: Path,
    all_view_poses_decoded: list,
    da3_extrinsics: np.ndarray,
    da3_scene_glb_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Visualize multi-view pose consistency: place each view's predicted object in world coordinates.
    
    If all views predict consistently, these objects should overlap.
    If inconsistent, can visually see which views deviate.
    
    Args:
        sam3d_glb_path: SAM3D output GLB file path (canonical space)
        all_view_poses_decoded: List of decoded poses for all views
        da3_extrinsics: DA3 camera extrinsics (N, 3, 4) or (N, 4, 4), world-to-camera
        da3_scene_glb_path: DA3 scene.glb path (optional, for adding scene background)
        output_path: Output path
    
    Returns:
        Visualization GLB file path
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed, cannot create visualization")
        return None
    
    if not sam3d_glb_path.exists():
        logger.warning(f"SAM3D GLB not found: {sam3d_glb_path}")
        return None
    
    if output_path is None:
        output_path = sam3d_glb_path.parent / "multiview_pose_consistency.glb"
    
    try:
        # Load SAM3D GLB (canonical space)
        sam3d_scene = trimesh.load(str(sam3d_glb_path))
        
        # Extract canonical vertices
        canonical_vertices = None
        canonical_faces = None
        if isinstance(sam3d_scene, trimesh.Scene):
            for name, geom in sam3d_scene.geometry.items():
                if hasattr(geom, 'vertices'):
                    canonical_vertices = geom.vertices.copy()
                    if hasattr(geom, 'faces'):
                        canonical_faces = geom.faces.copy()
                    break
        elif hasattr(sam3d_scene, 'vertices'):
            canonical_vertices = sam3d_scene.vertices.copy()
            if hasattr(sam3d_scene, 'faces'):
                canonical_faces = sam3d_scene.faces.copy()
        
        if canonical_vertices is None:
            logger.warning("No vertices found in SAM3D GLB")
            return None
        
        logger.info(f"[MultiView Viz] Canonical vertices: {canonical_vertices.shape}")
        logger.info(f"[MultiView Viz] Number of views: {len(all_view_poses_decoded)}")
        
        # Z-up to Y-up rotation matrix (same as merge_glb_with_da3_aligned)
        z_up_to_y_up_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=np.float32)
        
        # PyTorch3D to CV camera space transform
        p3d_to_cv = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        
        # CV to glTF coordinate transform
        M_cv_to_gltf = np.eye(4, dtype=np.float64)
        M_cv_to_gltf[1, 1] = -1.0
        M_cv_to_gltf[2, 2] = -1.0
        
        # Create scene
        merged_scene = trimesh.Scene()
        
        # If DA3 scene exists, add as background
        alignment_matrix = None
        if da3_scene_glb_path is not None and da3_scene_glb_path.exists():
            da3_scene = trimesh.load(str(da3_scene_glb_path))
            
            # Get alignment matrix
            if hasattr(da3_scene, 'metadata') and da3_scene.metadata is not None:
                alignment_matrix = da3_scene.metadata.get('hf_alignment', None)
            
            # Add DA3 scene (semi-transparent gray)
            if isinstance(da3_scene, trimesh.Scene):
                for name, geom in da3_scene.geometry.items():
                    geom_copy = geom.copy()
                    if hasattr(geom_copy, 'visual'):
                        geom_copy.visual.face_colors = [128, 128, 128, 100]
                    merged_scene.add_geometry(geom_copy, node_name=f"da3_{name}")
        
        # Create transformed object for each view
        colors_per_view = [
            [255, 0, 0, 200],     # View 0: Red
            [0, 255, 0, 200],     # View 1: Green
            [0, 0, 255, 200],     # View 2: Blue
            [255, 255, 0, 200],   # View 3: Yellow
            [255, 0, 255, 200],   # View 4: Magenta
            [0, 255, 255, 200],   # View 5: Cyan
            [255, 128, 0, 200],   # View 6: Orange
            [128, 0, 255, 200],   # View 7: Purple
        ]
        
        for view_idx, pose in enumerate(all_view_poses_decoded):
            # Extract pose parameters
            translation = np.array(pose.get('translation', [[0, 0, 0]])).flatten()[:3]
            rotation_quat = np.array(pose.get('rotation', [[1, 0, 0, 0]])).flatten()[:4]
            scale = np.array(pose.get('scale', [[1, 1, 1]])).flatten()[:3]
            
            # Build transform (same as merge_glb_with_da3_aligned)
            quat_tensor = torch.tensor(rotation_quat, dtype=torch.float32).unsqueeze(0)
            R_sam3d = quaternion_to_matrix(quat_tensor)
            scale_tensor = torch.tensor(scale, dtype=torch.float32).reshape(1, -1)
            if scale_tensor.shape[-1] == 1:
                scale_tensor = scale_tensor.repeat(1, 3)
            translation_tensor = torch.tensor(translation, dtype=torch.float32).reshape(1, 3)
            pose_transform = (
                Transform3d(dtype=torch.float32)
                .scale(scale_tensor)
                .rotate(R_sam3d)
                .translate(translation_tensor)
            )
            
            # Transform vertices
            # Step 1: Z-up to Y-up
            v_rotated = canonical_vertices @ z_up_to_y_up_matrix.T
            
            # Step 2: Apply SAM3D pose -> PyTorch3D space
            pts_local = torch.from_numpy(v_rotated).float().unsqueeze(0)
            pts_p3d = pose_transform.transform_points(pts_local).squeeze(0).numpy()
            
            # Step 3: PyTorch3D -> CV camera space
            pts_cv = pts_p3d @ p3d_to_cv.T
            
            # Step 4: View i camera space -> world coordinates
            w2c_i = da3_extrinsics[view_idx]
            if w2c_i.shape == (3, 4):
                w2c_i_44 = np.eye(4, dtype=np.float64)
                w2c_i_44[:3, :4] = w2c_i
                w2c_i = w2c_i_44
            c2w_i = np.linalg.inv(w2c_i)
            pts_world = trimesh.transform_points(pts_cv, c2w_i)
            
            # Step 5: World coordinates -> glTF coordinates
            pts_gltf = trimesh.transform_points(pts_world, M_cv_to_gltf)
            
            # Step 6: Apply centering offset if alignment matrix exists
            if alignment_matrix is not None and view_idx == 0:
                # Use View 0 to compute centering offset
                # Apply alignment_matrix to View 0 CV space points
                pts_aligned_v0 = trimesh.transform_points(pts_cv, alignment_matrix)
                center_offset = pts_aligned_v0.mean(axis=0) - pts_gltf.mean(axis=0)
            
            if alignment_matrix is not None:
                pts_final = pts_gltf + center_offset
            else:
                pts_final = pts_gltf
            
            # Filter invalid points
            valid = np.isfinite(pts_final).all(axis=1)
            pts_final = pts_final[valid]
            
            # Create mesh
            color = colors_per_view[view_idx % len(colors_per_view)]
            if canonical_faces is not None and valid.sum() == len(canonical_vertices):
                mesh = trimesh.Trimesh(
                    vertices=pts_final,
                    faces=canonical_faces,
                    process=False
                )
                mesh.visual.face_colors = color
            else:
                mesh = trimesh.PointCloud(pts_final, colors=np.tile(color, (len(pts_final), 1)))
            
            merged_scene.add_geometry(mesh, node_name=f"view{view_idx}_object")
            
            logger.info(f"  View {view_idx}: center = {pts_final.mean(axis=0)}, scale = {scale[0]:.4f}")
        
        # Export
        merged_scene.export(str(output_path))
        logger.info(f"[MultiView Viz] Saved: {output_path}")
        logger.info(f"  Colors: View0=Red, View1=Green, View2=Blue, View3=Yellow, View4=Magenta, View5=Cyan, View6=Orange, View7=Purple")
        
        return output_path
        
    except Exception as e:
        logger.warning(f"Failed to create multiview visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import SSIPointmapNormalizer
from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer


def convert_da3_extrinsics_to_camera_poses(
    da3_extrinsics: np.ndarray,
) -> List[dict]:
    """
    Convert DA3 extrinsics (world-to-camera) to camera_poses format.
    
    DA3 extrinsics are (N, 3, 4) or (N, 4, 4) w2c matrices.
    
    Args:
        da3_extrinsics: DA3 camera extrinsics, shape (N, 3, 4) or (N, 4, 4)
    
    Returns:
        List of camera pose dicts, each containing:
            - 'view_idx': int
            - 'c2w': (4, 4) camera-to-world matrix
            - 'w2c': (4, 4) world-to-camera matrix
            - 'R_c2w': (3, 3) rotation matrix
            - 't_c2w': (3,) translation vector
            - 'camera_position': (3,) camera position in world coordinates
    """
    num_views = da3_extrinsics.shape[0]
    camera_poses = []
    
    for view_idx in range(num_views):
        w2c_raw = da3_extrinsics[view_idx]  # (3, 4) or (4, 4)
        
        # Convert to (4, 4) format
        if w2c_raw.shape == (3, 4):
            w2c = np.eye(4)
            w2c[:3, :] = w2c_raw
        else:
            w2c = w2c_raw
        
        # Compute c2w = inv(w2c)
        c2w = np.linalg.inv(w2c)
        
        # Extract rotation and translation
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        camera_position = t_c2w  # Camera position is the translation part of c2w
        
        camera_poses.append({
            'view_idx': view_idx,
            'c2w': c2w,
            'w2c': w2c,
            'R_c2w': R_c2w,
            't_c2w': t_c2w,
            'camera_position': camera_position,
        })
    
    logger.info(f"[DA3 Extrinsics] Converted {num_views} extrinsics to camera poses")
    return camera_poses


def overlay_sam3d_on_pointmap(
    sam3d_glb_path: Path,
    input_pointmap,
    sam3d_pose: dict,
    input_image = None,
    output_path: Optional[Path] = None,
    pointmap_scale: Optional[np.ndarray] = None,
    pointmap_shift: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """
    Overlay SAM3D reconstructed object onto input pointmap.
    
    SAM3D pose parameters (scale, rotation, translation) are in real-world scale,
    and in PyTorch3D camera space.
    Input pointmap should also be in PyTorch3D camera space.
    
    Transform pipeline:
    SAM3D canonical (±0.5)
        ↓ scale * rotation + translation (SAM3D pose, real-world scale, PyTorch3D space)
    PyTorch3D camera space (real-world scale)
    
    Args:
        sam3d_glb_path: SAM3D output GLB file path (canonical space)
        input_pointmap: Input pointmap, shape (3, H, W), in PyTorch3D camera space
        sam3d_pose: SAM3D pose parameters {'scale', 'rotation', 'translation'}
        input_image: Original image for point cloud coloring
        output_path: Output path
    
    Returns:
        Overlaid GLB file path
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh or scipy not installed, cannot create overlay GLB")
        return None
    
    if not sam3d_glb_path.exists():
        logger.warning(f"SAM3D GLB not found: {sam3d_glb_path}")
        return None
    
    if output_path is None:
        output_path = sam3d_glb_path.parent / f"{sam3d_glb_path.stem}_overlay.glb"
    
    try:
        # Load SAM3D GLB
        sam3d_scene = trimesh.load(str(sam3d_glb_path))
        
        # Extract SAM3D pose parameters (already in PyTorch3D camera space, real-world scale)
        scale = sam3d_pose.get('scale', np.array([1.0, 1.0, 1.0]))
        rotation_quat = sam3d_pose.get('rotation', np.array([1.0, 0.0, 0.0, 0.0]))  # wxyz
        translation = sam3d_pose.get('translation', np.array([0.0, 0.0, 0.0]))
        
        if len(scale.shape) > 1:
            scale = scale.flatten()
        if len(rotation_quat.shape) > 1:
            rotation_quat = rotation_quat.flatten()
        if len(translation.shape) > 1:
            translation = translation.flatten()
        
        logger.info(f"[Overlay] SAM3D pose (PyTorch3D camera space):")
        logger.info(f"  scale: {scale} (object size, unit: meters)")
        logger.info(f"  rotation (wxyz): {rotation_quat}")
        logger.info(f"  translation: {translation} (object position, unit: meters)")
        
        # SAM3D internally applies z-up -> y-up rotation to GLB vertices
        # Must be consistent with layout_post_optimization_utils.get_mesh
        # Transform matrix: X = X, Y = -Z, Z = Y
        z_up_to_y_up_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=np.float32)
        quat_tensor = torch.tensor(rotation_quat, dtype=torch.float32).unsqueeze(0)
        R_sam3d = quaternion_to_matrix(quat_tensor)
        scale_tensor = torch.tensor(scale, dtype=torch.float32).reshape(1, -1)
        if scale_tensor.shape[-1] == 1:
            scale_tensor = scale_tensor.repeat(1, 3)
        translation_tensor = torch.tensor(translation, dtype=torch.float32).reshape(1, 3)
        pose_transform = (
            Transform3d(dtype=torch.float32)
            .scale(scale_tensor)
            .rotate(R_sam3d)
            .translate(translation_tensor)
        )
        
        def transform_to_pytorch3d_camera(vertices):
            """
            Transform SAM3D canonical space vertices to PyTorch3D camera space.
            
            Steps:
            1. Rotate canonical vertices from Z-up to Y-up (handled internally by SAM3D)
            2. Apply SAM3D pose (scale, rotation, translation)
            """
            # 1. Z-up to Y-up rotation
            v_rotated = vertices @ z_up_to_y_up_matrix.T
            pts_local = torch.from_numpy(v_rotated).float().unsqueeze(0)
            pts_world = pose_transform.transform_points(pts_local).squeeze(0).numpy()
            return pts_world
        
        # Create merged scene
        merged_scene = trimesh.Scene()
        
        # Transform and add SAM3D object
        if isinstance(sam3d_scene, trimesh.Scene):
            for name, geom in sam3d_scene.geometry.items():
                if hasattr(geom, 'vertices'):
                    geom_copy = geom.copy()
                    geom_copy.vertices = transform_to_pytorch3d_camera(geom_copy.vertices)
                    merged_scene.add_geometry(geom_copy, node_name=f"sam3d_{name}")
                else:
                    merged_scene.add_geometry(geom, node_name=f"sam3d_{name}")
        else:
            if hasattr(sam3d_scene, 'vertices'):
                sam3d_scene.vertices = transform_to_pytorch3d_camera(sam3d_scene.vertices)
            merged_scene.add_geometry(sam3d_scene, node_name="sam3d_object")
        
        # Create point cloud from input pointmap (already in PyTorch3D camera space)
        # input_pointmap shape: (3, H, W) or (1, 3, H, W)
        pm_np = input_pointmap
        if torch.is_tensor(pm_np):
            pm_tensor = pm_np.detach().cpu()
        else:
            pm_tensor = torch.from_numpy(pm_np).float()
            
        # Remove batch dimension
        while pm_tensor.ndim > 3:
            pm_tensor = pm_tensor[0]
        
        # Convert to (3, H, W)
        if pm_tensor.ndim == 3 and pm_tensor.shape[0] != 3:
            pm_tensor = pm_tensor.permute(2, 0, 1)
        
        # De-normalize (if needed)
        if pointmap_scale is not None and pointmap_shift is not None:
            normalizer = SSIPointmapNormalizer()
            scale_t = torch.as_tensor(pointmap_scale).float().view(-1)
            shift_t = torch.as_tensor(pointmap_shift).float().view(-1)
            pm_tensor = normalizer.denormalize(pm_tensor, scale_t, shift_t)
        
        pm_np = pm_tensor.permute(1, 2, 0).numpy()
        H, W = pm_np.shape[:2]
        
        # Get colors (from original image)
        colors = None
        if input_image is not None:
            from PIL import Image as PILImage
            if hasattr(input_image, 'convert'):
                # PIL Image
                img_np = np.array(input_image.convert("RGB"))
            else:
                # numpy array
                img_np = input_image
                if img_np.shape[-1] == 4:
                    img_np = img_np[..., :3]
            # Resize image to match pointmap resolution if needed
            if img_np.shape[:2] != (H, W):
                img_pil = PILImage.fromarray(img_np.astype(np.uint8))
                img_pil_resized = img_pil.resize((W, H), PILImage.BILINEAR)
                img_np = np.array(img_pil_resized)
            colors = img_np.reshape(-1, 3)
        
        # Filter invalid points (NaN, Inf)
        valid_mask = np.all(np.isfinite(pm_np), axis=-1)
        pm_points = pm_np[valid_mask].reshape(-1, 3)
        
        if colors is not None:
            colors = colors.reshape(H, W, 3)[valid_mask].reshape(-1, 3)
        else:
            # Default gray
            colors = np.full((len(pm_points), 3), 128, dtype=np.uint8)
        
        # Downsample
        if len(pm_points) > 100000:
            step = len(pm_points) // 100000
            pm_points = pm_points[::step]
            colors = colors[::step]
        
        # Create point cloud
        point_cloud = trimesh.points.PointCloud(vertices=pm_points, colors=colors)
        merged_scene.add_geometry(point_cloud, node_name="input_pointcloud")
        
        logger.info(f"[Overlay] Points in pointcloud: {len(pm_points)}")
        
        # Export
        merged_scene.export(str(output_path))
        logger.info(f"✓ Overlay GLB saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.warning(f"Failed to create overlay GLB: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Self-Occlusion Detection using Voxel Ray Casting
# ============================================================================

def ray_box_intersection(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> tuple:
    """
    Compute ray-AABB box intersection.
    
    Returns:
        (t_enter, t_exit) or (None, None) if no intersection
    """
    t_min = -np.inf
    t_max = np.inf
    
    for i in range(3):
        if abs(ray_dir[i]) < 1e-10:
            # Ray parallel to this axis
            if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                return None, None
        else:
            t1 = (box_min[i] - ray_origin[i]) / ray_dir[i]
            t2 = (box_max[i] - ray_origin[i]) / ray_dir[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None, None
    
    return t_min, t_max


def trace_ray_3d_dda(
    start: np.ndarray,  # (3,) Ray start (can be outside voxel grid)
    end: np.ndarray,    # (3,) Ray end (voxel index)
    grid_size: int = 64,
) -> List[tuple]:
    """
    3D DDA (Digital Differential Analyzer) algorithm.
    Trace ray from start to end, return all traversed voxels.
    
    Optimization: if start is outside grid, compute intersection with grid boundary first.
    
    Args:
        start: Ray start (float coordinates)
        end: Ray end (float coordinates)
        grid_size: Voxel grid size
    
    Returns:
        List of (dim0, dim1, dim2) voxel indices, in order from start to end
    """
    # Ray direction
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return []
    direction = direction / length
    
    # Check if start is outside grid, if so, find entry point
    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([grid_size, grid_size, grid_size])
    
    actual_start = start.copy()
    
    # If start is outside grid, compute entry point
    if not np.all((start >= 0) & (start < grid_size)):
        t_enter, t_exit = ray_box_intersection(start, direction, box_min, box_max)
        if t_enter is None or t_enter > length:
            # Ray does not pass through grid, or entry point is after end
            return []
        if t_enter > 0:
            # Start from entry point
            actual_start = start + direction * (t_enter + 0.001)
    
    # Current position
    current = actual_start.copy()
    
    # Current voxel
    voxel = np.floor(current).astype(int)
    # Ensure points on boundary are handled correctly
    voxel = np.clip(voxel, 0, grid_size - 1)
    
    end_voxel = np.floor(end).astype(int)
    end_voxel = np.clip(end_voxel, 0, grid_size - 1)
    
    # Step direction
    step = np.sign(direction).astype(int)
    step[step == 0] = 1  # Avoid division by zero
    
    # Compute distance to next voxel boundary
    tmax = np.zeros(3)
    tdelta = np.zeros(3)
    
    for i in range(3):
        if abs(direction[i]) < 1e-10:
            tmax[i] = float('inf')
            tdelta[i] = float('inf')
        else:
            if direction[i] > 0:
                tmax[i] = (voxel[i] + 1 - current[i]) / direction[i]
            else:
                tmax[i] = (voxel[i] - current[i]) / direction[i]
            tdelta[i] = abs(1.0 / direction[i])
    
    # Collect traversed voxels
    voxels = []
    max_steps = grid_size * 3  # Max steps in grid
    
    for _ in range(max_steps):
        # Check if inside grid
        if np.all((voxel >= 0) & (voxel < grid_size)):
            voxels.append(tuple(voxel))
        else:
            # Already outside grid, stop
            break
        
        # Check if reached end
        if np.all(voxel == end_voxel):
            break
        
        # Find minimum tmax, decide next step direction
        min_axis = np.argmin(tmax)
        
        # Step to next voxel
        voxel[min_axis] += step[min_axis]
        tmax[min_axis] += tdelta[min_axis]
    
    return voxels


def compute_self_occlusion(
    latent_coords: np.ndarray,  # (N, 4) or (N, 3) - voxel coordinates
    camera_position_voxel: np.ndarray,  # (3,) Camera position in voxel space
    grid_size: int = 64,
    neighbor_tolerance: float = 4.0,  # Ignore occluding voxels within this distance (4.0 handles grazing angles)
) -> np.ndarray:
    """
    Detect self-occlusion using 3D DDA ray tracing with neighbor tolerance.
    
    Core idea:
    - Build 64x64x64 occupancy grid
    - For each voxel, cast ray from camera to that voxel
    - If ray passes through other occupied voxels (far enough from target), the voxel is occluded
    
    Improvements:
    - neighbor_tolerance: ignore occluding voxels within this distance of target
    - This avoids false positives from adjacent voxels
    
    Args:
        latent_coords: (N, 4) or (N, 3), voxel coordinates
        camera_position_voxel: Camera position in voxel space
        grid_size: Voxel grid size (default 64)
        neighbor_tolerance: Ignore occluding voxels within this distance (default 1.5, ~sqrt(3), diagonal neighbors)
    
    Returns:
        self_visible: (N,) bool array, True = not self-occluded
    """
    # Handle coordinate format
    if latent_coords.shape[1] == 4:
        voxel_coords = latent_coords[:, 1:4].astype(int)
    else:
        voxel_coords = latent_coords.astype(int)
    
    N = len(voxel_coords)
    
    # Debug info
    logger.info(f"[Self-Occlusion DDA] Voxel coords range: "
               f"dim0=[{voxel_coords[:, 0].min()}, {voxel_coords[:, 0].max()}], "
               f"dim1=[{voxel_coords[:, 1].min()}, {voxel_coords[:, 1].max()}], "
               f"dim2=[{voxel_coords[:, 2].min()}, {voxel_coords[:, 2].max()}]")
    logger.info(f"[Self-Occlusion DDA] Camera position in voxel space: {camera_position_voxel}")
    logger.info(f"[Self-Occlusion DDA] Neighbor tolerance: {neighbor_tolerance}")
    
    # Step 1: Build occupancy grid
    occupancy = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    for coord in voxel_coords:
        d0, d1, d2 = coord[0], coord[1], coord[2]
        if 0 <= d0 < grid_size and 0 <= d1 < grid_size and 0 <= d2 < grid_size:
            occupancy[d0, d1, d2] = True
    
    logger.info(f"[Self-Occlusion DDA] Built occupancy grid: {occupancy.sum()} occupied voxels")
    
    # Step 2: DDA ray tracing for each voxel
    self_visible = np.ones(N, dtype=bool)
    occluded_count = 0
    
    # Pre-compute target voxel coordinates (float, for distance calculation)
    target_coords_float = voxel_coords.astype(float)
    
    tolerance_sq = neighbor_tolerance ** 2
    
    for i in range(N):
        target = target_coords_float[i] + 0.5  # Voxel center
        target_int = voxel_coords[i]
        
        # DDA ray tracing
        ray_voxels = trace_ray_3d_dda(
            camera_position_voxel,
            target,
            grid_size
        )
        
        # Check if there are occupied voxels along the ray (excluding target and neighbors)
        for voxel in ray_voxels:
            d0, d1, d2 = voxel
            
            # Skip voxels outside grid
            if not (0 <= d0 < grid_size and 0 <= d1 < grid_size and 0 <= d2 < grid_size):
                continue
            
            # Skip unoccupied voxels
            if not occupancy[d0, d1, d2]:
                continue
            
            # Compute distance to target (voxel units)
            dist_sq = (d0 - target_int[0])**2 + (d1 - target_int[1])**2 + (d2 - target_int[2])**2
            
            # If too close to target (including target itself), skip
            if dist_sq <= tolerance_sq:
                continue
            
            # Found real occluding voxel
            self_visible[i] = False
            occluded_count += 1
            break
    
    visible_count = N - occluded_count
    logger.info(f"[Self-Occlusion DDA] Results: {visible_count} visible, {occluded_count} occluded "
               f"({100 * visible_count / N:.1f}% visible)")
    
    return self_visible


def canonical_to_voxel(pos_canonical: np.ndarray, scale: float) -> np.ndarray:
    """
    Convert canonical space coordinates to voxel space coordinates.
    
    Transform chain (extracted from compute_latent_visibility):
    voxel [0, 64) → normalized [-0.5, 0.5] → Z_UP_TO_Y_UP → scale → canonical
    
    Inverse transform:
    canonical → /scale → Y_UP_TO_Z_UP → (x+0.5)*64 → voxel
    
    Args:
        pos_canonical: (..., 3) canonical space coordinates [x, y, z] (Y-up)
        scale: Object scale factor
    
    Returns:
        pos_voxel: (..., 3) voxel space coordinates, keeping [x, y, z] order for ray tracing
    """
    # 1. Remove scale
    pos_normalized = pos_canonical / scale
    
    # 2. Y-up -> Z-up inverse transform
    # Z_UP_TO_Y_UP: (x, y, z)_zup → (x, -z, y)_yup
    # Inverse transform: (x, y, z)_yup -> (x, z, -y)_zup
    # 
    # Note: voxel coords from argwhere are in [z, y, x] order
    # After Z_UP_TO_Y_UP transform, canonical = [dim0, dim2, -dim1]
    # x = a, y = c, z = -b
    # → a = x, b = -z, c = y
    # → normalized = [x, -z, y]
    
    x, y, z = pos_normalized[..., 0], pos_normalized[..., 1], pos_normalized[..., 2]
    
    # normalized in voxel order [a, b, c] where canonical = [a, c, -b]
    # so a = x, b = -z, c = y
    voxel_normalized = np.stack([x, -z, y], axis=-1)
    
    # 3. normalized [-0.5, 0.5] → voxel [0, 64)
    pos_voxel = (voxel_normalized + 0.5) * 64
    
    return pos_voxel


def compute_self_occlusion_for_all_views(
    latent_coords: np.ndarray,  # (N, 4) or (N, 3) - voxel coordinates
    camera_positions_canonical: List[np.ndarray],  # List of camera positions in canonical space
    scale: float,
    grid_size: int = 64,
    neighbor_tolerance: float = 4.0,  # Ignore occluding voxels within this distance
) -> np.ndarray:
    """
    Compute self-occlusion for all views.
    
    Args:
        latent_coords: Voxel coordinates
        camera_positions_canonical: Camera position for each view (canonical space)
        scale: Object scale factor
        grid_size: Voxel grid size
        neighbor_tolerance: Ignore occluding voxels within this distance (handles grazing angles)
    
    Returns:
        self_occlusion_matrix: (N, num_views) matrix, 1.0 = visible, 0.0 = self-occluded
    """
    num_views = len(camera_positions_canonical)
    N = len(latent_coords)
    
    self_occlusion_matrix = np.zeros((N, num_views), dtype=np.float32)
    
    for view_idx, camera_pos_canonical in enumerate(camera_positions_canonical):
        # Convert camera position to voxel space
        camera_pos_voxel = canonical_to_voxel(camera_pos_canonical, scale)
        
        logger.info(f"[Self-Occlusion] View {view_idx}: "
                   f"camera canonical={camera_pos_canonical}, voxel={camera_pos_voxel}")
        
        # Compute self-occlusion for this view
        self_visible = compute_self_occlusion(
            latent_coords, 
            camera_pos_voxel,
            grid_size,
            neighbor_tolerance
        )
        
        self_occlusion_matrix[:, view_idx] = self_visible.astype(np.float32)
    
    return self_occlusion_matrix


def visualize_self_occlusion_per_view(
    self_occlusion_matrix: np.ndarray,  # (N, num_views) - self-occlusion result
    visibility_result: dict,  # Result from compute_latent_visibility (contains canonical coords and camera poses)
    output_dir: Path,
) -> List[Path]:
    """
    Visualize self-occlusion results for each view.
    
    Directly use canonical coordinates and camera poses from visibility_result (verified),
    ensuring alignment with latent_visibility_per_view.
    
    Green: visible (not self-occluded)
    Red: self-occluded
    
    Returns:
        List of output file paths
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed, cannot visualize")
        return []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Directly use canonical coords from visibility_result (verified correct)
    canonical_coords = visibility_result['canonical_coords']
    canonical_camera_poses = visibility_result['canonical_camera_poses']
    scale = visibility_result['scale']
    
    num_views = len(canonical_camera_poses)
    output_paths = []
    
    # Define view colors (same as latent_visibility_per_view)
    view_colors = [
        [255, 100, 100, 255],  # Red
        [100, 255, 100, 255],  # Green
        [100, 100, 255, 255],  # Blue
        [255, 255, 100, 255],  # Yellow
        [255, 100, 255, 255],  # Purple
        [100, 255, 255, 255],  # Cyan
        [255, 180, 100, 255],  # Orange
        [180, 100, 255, 255],  # Purple-blue
    ]
    
    for view_idx in range(num_views):
        scene = trimesh.Scene()
        
        # Self-occlusion status
        self_visible = self_occlusion_matrix[:, view_idx] > 0.5
        
        # Colors: green=visible, red=occluded
        colors_visible = np.zeros((len(canonical_coords), 4), dtype=np.uint8)
        colors_visible[self_visible] = [0, 255, 0, 255]   # Green: visible
        colors_visible[~self_visible] = [255, 0, 0, 255]  # Red: occluded
        
        # Use spheres to display latent points (larger and clearer)
        # Sample display (if too many points)
        max_spheres = 10000
        if len(canonical_coords) > max_spheres:
            indices = np.random.choice(len(canonical_coords), max_spheres, replace=False)
        else:
            indices = np.arange(len(canonical_coords))
        
        sphere_radius = scale * 0.008  # Sphere radius
        for idx in indices:
            sphere = trimesh.creation.icosphere(radius=sphere_radius, subdivisions=1)
            sphere.apply_translation(canonical_coords[idx])
            color = colors_visible[idx]
            sphere.visual.vertex_colors = color
            scene.add_geometry(sphere, node_name=f"latent_{idx}")
        
        # Add all cameras, current view larger, others smaller
        for cam_idx, cam_pose in enumerate(canonical_camera_poses):
            camera_pos = cam_pose['camera_position']
            
            # Current view camera larger, others smaller
            if cam_idx == view_idx:
                radius = scale * 0.08  # Large
                subdivisions = 2
            else:
                radius = scale * 0.03  # Small
                subdivisions = 1
            
            camera_sphere = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
            camera_sphere.apply_translation(camera_pos)
            
            # Color
            color = view_colors[cam_idx % len(view_colors)]
            camera_sphere.visual.vertex_colors = color
            
            scene.add_geometry(camera_sphere, node_name=f"camera_{cam_idx}")
        
        # Save
        output_path = output_dir / f"self_occlusion_view_{view_idx:02d}.glb"
        scene.export(str(output_path))
        output_paths.append(output_path)
        
        visible_count = self_visible.sum()
        logger.info(f"[Self-Occlusion Viz] View {view_idx}: "
                   f"{visible_count}/{len(canonical_coords)} visible ({100*visible_count/len(canonical_coords):.1f}%), "
                   f"saved to {output_path.name}")
    
    return output_paths


def compute_latent_visibility(
    latent_coords: np.ndarray,  # (N, 4) or (N, 3) - Stage 2 latent coordinates (voxel space)
    object_pose: dict,  # Object pose from Stage 1 {'scale', 'rotation', 'translation'} (in view0 coordinates)
    camera_poses: List[dict],  # List of camera poses, each containing {'c2w', 'w2c', 'camera_position', 'R_c2w'}
    self_occlusion_tolerance: float = 4.0,  # Self-occlusion detection tolerance (voxel units)
) -> dict:
    """
    Compute visibility of each latent point from each view in CANONICAL space.
    
    **Core idea**:
    - Object in canonical space, only apply scale (not rotation/translation)
    - Transform camera poses to canonical space
    - Use self-occlusion (DDA ray tracing) for visibility
    
    Args:
        latent_coords: Stage 2 latent coordinates (N, 4) or (N, 3)
        object_pose: Object pose {'scale', 'rotation' (wxyz), 'translation'}
        camera_poses: List of camera poses
        self_occlusion_tolerance: Self-occlusion detection tolerance (voxel units)
    
    Returns:
        dict: 
            - visibility_matrix: (N_latents, N_views) visibility matrix (0=occluded, 1=visible)
            - canonical_coords: (N, 3) latent points in canonical space
            - canonical_camera_poses: Camera poses in canonical space
            - scale: Object scale
    """
    from scipy.spatial.transform import Rotation as R_scipy
    
    num_views = len(camera_poses)
    
    # === Helper function to convert tensor to numpy ===
    def to_numpy(x):
        if x is None:
            return None
        if hasattr(x, 'cpu'):
            return x.cpu().numpy()
        return np.array(x)
    
    # === Step 1: Object pose parameters ===
    obj_scale = np.atleast_1d(to_numpy(object_pose.get('scale', [1, 1, 1]))).flatten()
    if len(obj_scale) == 1:
        obj_scale = np.array([obj_scale[0], obj_scale[0], obj_scale[0]])
    obj_rotation_quat = np.atleast_1d(to_numpy(object_pose.get('rotation', [1, 0, 0, 0]))).flatten()
    obj_translation = np.atleast_1d(to_numpy(object_pose.get('translation', [0, 0, 0]))).flatten()
    
    # Object rotation matrix (wxyz -> scipy xyzw)
    obj_R = R_scipy.from_quat([obj_rotation_quat[1], obj_rotation_quat[2], 
                               obj_rotation_quat[3], obj_rotation_quat[0]]).as_matrix()
    
    # Z-up to Y-up transform (consistent with GLB standard)
    Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    
    logger.info(f"[Visibility Canonical] Computing visibility in CANONICAL space")
    logger.info(f"[Visibility Canonical] Object scale: {obj_scale}")
    logger.info(f"[Visibility Canonical] Object rotation (wxyz): {obj_rotation_quat}")
    logger.info(f"[Visibility Canonical] Object translation: {obj_translation}")
    
    # === Step 2: Transform latent points to canonical space (apply scale, Y-up only) ===
    # Convert to numpy if tensor
    if hasattr(latent_coords, 'cpu'):
        latent_coords = latent_coords.cpu().numpy()
    
    # Handle (N, 4) format
    if latent_coords.shape[1] == 4:
        coords = latent_coords[:, 1:4].copy()
    else:
        coords = latent_coords.copy()
    
    # Convert voxel indices to canonical [-0.5, 0.5]
    if coords.max() > 1.0:
        coords = (coords / 64.0) - 0.5
    coords = np.clip(coords, -0.5, 0.5)
    
    # Apply Z-up to Y-up and scale
    canonical_coords = (coords @ Z_UP_TO_Y_UP) * obj_scale[0]
    
    num_latents = canonical_coords.shape[0]
    
    logger.info(f"[Visibility Canonical] Latent points in canonical space:")
    logger.info(f"  Count: {num_latents}")
    logger.info(f"  X: [{canonical_coords[:, 0].min():.4f}, {canonical_coords[:, 0].max():.4f}]")
    logger.info(f"  Y: [{canonical_coords[:, 1].min():.4f}, {canonical_coords[:, 1].max():.4f}]")
    logger.info(f"  Z: [{canonical_coords[:, 2].min():.4f}, {canonical_coords[:, 2].max():.4f}]")
    
    # === Step 3: Transform camera poses to canonical space ===
    # Use the same method as visualize_in_canonical_space
    canonical_camera_poses = []
    camera_positions_for_occlusion = []  # For self-occlusion calculation
    for view_idx, cam_pose in enumerate(camera_poses):
        # Get camera pose in world coordinates
        camera_pos_world = np.array(cam_pose.get('camera_position', [0, 0, 0])).flatten()
        cam_R_c2w = np.array(cam_pose.get('R_c2w', np.eye(3)))
        
        # Transform camera position to canonical space:
        # 1. Subtract object translation
        # 2. Apply inverse of object rotation
        # 3. Apply Z-up to Y-up transform
        camera_pos_obj = obj_R.T @ (camera_pos_world - obj_translation)
        camera_pos_canonical = camera_pos_obj @ Z_UP_TO_Y_UP
        
        # Transform camera's three axis vectors separately (not rotation matrix directly)
        # This ensures consistency with visualize_in_canonical_space
        camera_forward_world = cam_R_c2w @ np.array([0, 0, 1])  # Z axis (forward)
        camera_up_world = cam_R_c2w @ np.array([0, 1, 0])        # Y axis (up)
        camera_right_world = cam_R_c2w @ np.array([1, 0, 0])     # X axis (right)
        
        # Transform each axis to canonical space
        camera_forward_obj = obj_R.T @ camera_forward_world
        camera_forward_canonical = camera_forward_obj @ Z_UP_TO_Y_UP
        camera_forward_canonical = camera_forward_canonical / (np.linalg.norm(camera_forward_canonical) + 1e-8)
        
        camera_up_obj = obj_R.T @ camera_up_world
        camera_up_canonical = camera_up_obj @ Z_UP_TO_Y_UP
        camera_up_canonical = camera_up_canonical / (np.linalg.norm(camera_up_canonical) + 1e-8)
        
        camera_right_obj = obj_R.T @ camera_right_world
        camera_right_canonical = camera_right_obj @ Z_UP_TO_Y_UP
        camera_right_canonical = camera_right_canonical / (np.linalg.norm(camera_right_canonical) + 1e-8)
        
        # Rebuild c2w and w2c matrices
        cam_R_canonical = np.column_stack([camera_right_canonical, camera_up_canonical, camera_forward_canonical])
        
        # Compute w2c matrix in canonical space
        w2c_canonical = np.eye(4)
        w2c_canonical[:3, :3] = cam_R_canonical.T  # R_w2c = R_c2w.T
        w2c_canonical[:3, 3] = -cam_R_canonical.T @ camera_pos_canonical
        
        canonical_camera_poses.append({
            'camera_position': camera_pos_canonical,
            'R_c2w': cam_R_canonical,
            'w2c': w2c_canonical,
            'camera_forward': camera_forward_canonical,  # Also save forward direction for visualization
            'view_idx': view_idx,
        })
        
        # Save camera position for self-occlusion calculation
        camera_positions_for_occlusion.append(camera_pos_canonical)
        
        if view_idx == 0:
            logger.info(f"[Visibility Canonical] Camera 0 in canonical space:")
            logger.info(f"  Position: {camera_pos_canonical}")
            logger.info(f"  Forward: {camera_forward_canonical}")
    
    # === Step 4: Use self-occlusion (DDA) for visibility ===
    logger.info(f"[Visibility] Computing self-occlusion with tolerance={self_occlusion_tolerance}")
    
    # Call self-occlusion calculation function
    visibility_matrix = compute_self_occlusion_for_all_views(
        latent_coords=latent_coords,  # Original voxel coordinates
        camera_positions_canonical=camera_positions_for_occlusion,
        scale=obj_scale[0],
        grid_size=64,
        neighbor_tolerance=self_occlusion_tolerance,
    )
    
    # Statistics
    logger.info(f"[Visibility Canonical] Visibility matrix computed: shape={visibility_matrix.shape}")
    logger.info(f"[Visibility Canonical] Stats: mean={visibility_matrix.mean():.3f}, "
               f"min={visibility_matrix.min():.3f}, max={visibility_matrix.max():.3f}")
    
    for view_idx in range(num_views):
        view_vis = visibility_matrix[:, view_idx]
        visible_count = (view_vis > 0.5).sum()
        logger.info(f"  View {view_idx}: visible={visible_count}/{num_latents} ({visible_count/num_latents:.1%})")
    
    # Return visibility matrix and canonical space data (for visualization)
    return {
        'visibility_matrix': visibility_matrix,
        'canonical_coords': canonical_coords,
        'canonical_camera_poses': canonical_camera_poses,
        'scale': obj_scale[0],
    }


def visualize_in_canonical_space(
    latent_coords: np.ndarray,
    visibility_matrix: np.ndarray,
    scale: float,
    reference_glb = None,
    camera_poses: Optional[List[dict]] = None,
    object_pose: Optional[dict] = None,  # Need object pose to transform cameras to canonical space
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Display Latent and Mesh in object Canonical space (Y-up, consistent with GLB standard).
    
    **Key finding**:
    - GLB export applies Z-up -> Y-up transform to mesh: (x,y,z) -> (x,-z,y)
    - Latent coords don't have this transform
    - So we need to apply the same transform to latent for alignment
    
    **Camera pose handling**:
    - Input camera_poses are in world coordinates
    - Need to use inverse of object_pose to transform cameras to object canonical space
    
    Args:
        latent_coords: Latent coordinates (N, 4) or (N, 3)
        visibility_matrix: Visibility matrix (N, N_views)
        scale: Object scale factor
        reference_glb: Reference mesh (trimesh.Scene), already in Y-up space
        camera_poses: List of camera poses (optional, in world coordinates)
        object_pose: Object pose dict (scale, rotation, translation), for transforming cameras to canonical space
        output_path: Output path
    
    Returns:
        Output file path
    """
    try:
        import trimesh
    except ImportError:
        logger.error("trimesh not installed")
        return None
    
    # Convert scale to numpy float if it's a tensor
    if hasattr(scale, 'cpu'):  # It's a tensor
        scale = float(scale.cpu().numpy().flatten()[0])
    elif hasattr(scale, '__len__'):  # It's an array
        scale = float(np.atleast_1d(scale).flatten()[0])
    else:
        scale = float(scale)
    
    if output_path is None:
        output_path = Path("visualization") / "canonical_view.glb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[Canonical Viz] Creating visualization in Y-up canonical space (GLB standard)")
    logger.info(f"[Canonical Viz] Scale = {scale:.4f}")
    
    # === Process Latent coords ===
    # Convert to numpy if tensor
    if hasattr(latent_coords, 'cpu'):
        latent_coords = latent_coords.cpu().numpy()
    
    # Handle (N, 4) format
    if latent_coords.shape[1] == 4:
        coords = latent_coords[:, 1:4].copy()
    else:
        coords = latent_coords.copy()
    
    # Convert voxel indices to canonical [-0.5, 0.5]
    if coords.max() > 1.0:
        coords = (coords / 64.0) - 0.5
    coords = np.clip(coords, -0.5, 0.5)
    
    logger.info(f"[Canonical Viz] Latent in Z-up canonical space:")
    logger.info(f"  dim0: [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}], range={coords[:, 0].max()-coords[:, 0].min():.4f}")
    logger.info(f"  dim1: [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}], range={coords[:, 1].max()-coords[:, 1].min():.4f}")
    logger.info(f"  dim2: [{coords[:, 2].min():.4f}, {coords[:, 2].max():.4f}], range={coords[:, 2].max()-coords[:, 2].min():.4f}")
    
    # Apply Z-up to Y-up transformation (same as GLB export does for mesh)
    # This transforms (x, y, z) -> (x, -z, y)
    Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    latent_yup = coords @ Z_UP_TO_Y_UP
    
    # Apply scale
    latent_world = latent_yup * scale
    
    logger.info(f"[Canonical Viz] Latent after Z-up->Y-up transform and scale:")
    logger.info(f"  X: [{latent_world[:, 0].min():.4f}, {latent_world[:, 0].max():.4f}], range={latent_world[:, 0].max()-latent_world[:, 0].min():.4f}")
    logger.info(f"  Y: [{latent_world[:, 1].min():.4f}, {latent_world[:, 1].max():.4f}], range={latent_world[:, 1].max()-latent_world[:, 1].min():.4f}")
    logger.info(f"  Z: [{latent_world[:, 2].min():.4f}, {latent_world[:, 2].max():.4f}], range={latent_world[:, 2].max()-latent_world[:, 2].min():.4f}")
    
    # === Process Mesh ===
    # GLB mesh is already in Y-up space, just apply scale
    mesh_for_scene = None
    if reference_glb is not None:
        try:
            mesh_vertices = None
            mesh_faces = None
            if isinstance(reference_glb, trimesh.Scene):
                for name, geom in reference_glb.geometry.items():
                    if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                        mesh_vertices = geom.vertices.copy()
                        mesh_faces = geom.faces.copy()
                        break
            elif hasattr(reference_glb, 'vertices'):
                mesh_vertices = reference_glb.vertices.copy()
                mesh_faces = reference_glb.faces.copy() if hasattr(reference_glb, 'faces') else None
            
            if mesh_vertices is not None:
                # GLB mesh is already in Y-up, already scaled (vertices are in [-0.5, 0.5])
                # Just apply scale to match latent
                mesh_world = mesh_vertices * scale
                
                logger.info(f"[Canonical Viz] Mesh (Y-up, from GLB) after scale:")
                logger.info(f"  X: [{mesh_world[:, 0].min():.4f}, {mesh_world[:, 0].max():.4f}], range={mesh_world[:, 0].max()-mesh_world[:, 0].min():.4f}")
                logger.info(f"  Y: [{mesh_world[:, 1].min():.4f}, {mesh_world[:, 1].max():.4f}], range={mesh_world[:, 1].max()-mesh_world[:, 1].min():.4f}")
                logger.info(f"  Z: [{mesh_world[:, 2].min():.4f}, {mesh_world[:, 2].max():.4f}], range={mesh_world[:, 2].max()-mesh_world[:, 2].min():.4f}")
                
                if mesh_faces is not None:
                    mesh_for_scene = trimesh.Trimesh(vertices=mesh_world, faces=mesh_faces)
                    mesh_for_scene.visual.face_colors = [0, 255, 255, 100]  # Cyan, semi-transparent
        except Exception as e:
            logger.warning(f"[Canonical Viz] Could not process mesh: {e}")
    
    # === Create scene ===
    scene = trimesh.Scene()
    
    # Visibility coloring
    visibility_scores = visibility_matrix.mean(axis=1)
    colors = np.zeros((len(latent_world), 4), dtype=np.uint8)
    colors[:, 0] = (255 * (1.0 - visibility_scores)).astype(np.uint8)  # Red for invisible
    colors[:, 1] = (255 * visibility_scores).astype(np.uint8)  # Green for visible
    colors[:, 3] = 200  # Alpha
    
    # Add latent point cloud
    point_cloud = trimesh.PointCloud(vertices=latent_world, colors=colors)
    scene.add_geometry(point_cloud, node_name="latent_points")
    
    # Add mesh
    if mesh_for_scene is not None:
        scene.add_geometry(mesh_for_scene, node_name="mesh_reference")
        logger.info(f"[Canonical Viz] Added mesh to scene")
    
    # === Add camera pose visualization (if provided) ===
    if camera_poses is not None and len(camera_poses) > 0 and object_pose is not None:
        logger.info(f"[Canonical Viz] Adding {len(camera_poses)} camera poses (transformed to canonical space)")
        
        # Get object pose for camera transform
        from scipy.spatial.transform import Rotation as R_scipy
        
        obj_scale = np.atleast_1d(object_pose.get('scale', [1, 1, 1])).flatten()
        if len(obj_scale) == 1:
            obj_scale = np.array([obj_scale[0], obj_scale[0], obj_scale[0]])
        obj_rotation_quat = np.atleast_1d(object_pose.get('rotation', [1, 0, 0, 0])).flatten()
        obj_translation = np.atleast_1d(object_pose.get('translation', [0, 0, 0])).flatten()
        
        # Object rotation matrix (wxyz -> scipy needs xyzw)
        obj_R = R_scipy.from_quat([obj_rotation_quat[1], obj_rotation_quat[2], 
                                    obj_rotation_quat[3], obj_rotation_quat[0]]).as_matrix()
        
        # Z-up to Y-up transform (consistent with GLB)
        Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        
        # Camera colors (consistent with other visualizations)
        camera_colors = [
            [255, 0, 0],    # View 0: Red
            [0, 255, 0],    # View 1: Green
            [0, 0, 255],    # View 2: Blue
            [255, 255, 0],  # View 3: Yellow
            [255, 0, 255],  # View 4: Magenta
            [0, 255, 255],  # View 5: Cyan
            [255, 128, 0],  # View 6: Orange
            [128, 0, 255],  # View 7: Purple
        ]
        
        for i, cam_pose in enumerate(camera_poses):
            try:
                # Get camera pose in world coordinates
                # camera_poses format: R_c2w, t_c2w, camera_position, c2w, w2c
                cam_R_c2w = np.array(cam_pose.get('R_c2w', np.eye(3)))
                camera_pos_world = np.array(cam_pose.get('camera_position', [0, 0, 0])).flatten()
                
                # Transform camera position to object canonical space (SCALED):
                # Note: Object in canonical_view is canonical * scale
                # So camera should also be in the same scaled space
                # 1. Subtract object translation (world coordinates)
                # 2. Apply inverse of object rotation (get position in object frame)
                # 3. Do not divide by scale (displayed object is also scaled)
                # 4. Apply Z-up to Y-up transform
                camera_pos_obj = obj_R.T @ (camera_pos_world - obj_translation)
                # Do not divide by scale! Displayed object is in scaled canonical space
                camera_pos_canonical = camera_pos_obj @ Z_UP_TO_Y_UP
                
                # Camera direction also needs transform (Z axis direction in c2w frame)
                camera_forward_world = cam_R_c2w @ np.array([0, 0, 1])  # Z axis direction of c2w
                camera_forward_obj = obj_R.T @ camera_forward_world
                camera_forward_canonical = camera_forward_obj @ Z_UP_TO_Y_UP
                # Normalize
                camera_forward_canonical = camera_forward_canonical / (np.linalg.norm(camera_forward_canonical) + 1e-8)
                
                color = camera_colors[i % len(camera_colors)]
                
                # Create camera sphere
                cam_marker = trimesh.creation.icosphere(radius=0.03, subdivisions=1)
                cam_marker.apply_translation(camera_pos_canonical)
                cam_marker.visual.face_colors = color + [255]
                scene.add_geometry(cam_marker, node_name=f"camera_{i}")
                
                # Add direction indicator line
                line_end = camera_pos_canonical + camera_forward_canonical * 0.15
                line_verts = np.array([camera_pos_canonical, line_end])
                line_colors = np.array([color + [255], color + [255]])
                line_pc = trimesh.PointCloud(vertices=line_verts, colors=line_colors)
                scene.add_geometry(line_pc, node_name=f"camera_{i}_dir")
                
                if i == 0:
                    logger.info(f"[Canonical Viz] Camera 0 world pos: {camera_pos_world}")
                    logger.info(f"[Canonical Viz] Camera 0 canonical pos: {camera_pos_canonical}")
                
            except Exception as e:
                logger.warning(f"[Canonical Viz] Could not add camera {i}: {e}")
    
    # Add coordinate axes (at origin)
    axis_length = scale * 0.3
    axis_verts = np.array([
        [0, 0, 0], [axis_length, 0, 0],
        [0, 0, 0], [0, axis_length, 0],
        [0, 0, 0], [0, 0, axis_length],
    ])
    axis_colors = np.array([
        [255, 0, 0, 255], [255, 0, 0, 255],  # X - Red
        [0, 255, 0, 255], [0, 255, 0, 255],  # Y - Green (up in Y-up space)
        [0, 0, 255, 255], [0, 0, 255, 255],  # Z - Blue
    ])
    axis_pc = trimesh.PointCloud(vertices=axis_verts, colors=axis_colors)
    scene.add_geometry(axis_pc, node_name="axes")
    
    # Save
    scene.export(str(output_path))
    logger.info(f"[Canonical Viz] Saved to: {output_path}")
    
    return output_path


def visualize_latent_visibility(
    visibility_result: dict,  # Result from compute_latent_visibility
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Visualize latent point visibility in CANONICAL space.
    
    Generate a GLB file containing:
    1. Latent points (colored by visibility: green=visible, red=invisible)
    2. Camera positions and orientations
    3. Coordinate axes
    
    Note: Does not show mesh, only latent points
    
    Args:
        visibility_result: Dictionary returned by compute_latent_visibility, containing:
            - visibility_matrix: Visibility matrix (N, N_views)
            - canonical_coords: Latent points in canonical space
            - canonical_camera_poses: Camera poses in canonical space
            - scale: Object scale
        output_path: Output path
    
    Returns:
        Output GLB file path
    """
    try:
        import trimesh
    except ImportError:
        logger.error("trimesh not installed, cannot create visualization")
        return None
    
    if output_path is None:
        output_path = Path("visualization") / "latent_visibility.glb"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    visibility_matrix = visibility_result['visibility_matrix']
    canonical_coords = visibility_result['canonical_coords']
    canonical_camera_poses = visibility_result['canonical_camera_poses']
    scale = visibility_result.get('scale', 1.0)
    
    logger.info(f"[Visibility Viz Canonical] Creating visualization in CANONICAL space...")
    logger.info(f"[Visibility Viz Canonical] {len(canonical_coords)} latent points, {len(canonical_camera_poses)} cameras")
    logger.info(f"[Visibility Viz Canonical] Latent coords range: "
                f"X=[{canonical_coords[:, 0].min():.4f}, {canonical_coords[:, 0].max():.4f}], "
                f"Y=[{canonical_coords[:, 1].min():.4f}, {canonical_coords[:, 1].max():.4f}], "
                f"Z=[{canonical_coords[:, 2].min():.4f}, {canonical_coords[:, 2].max():.4f}]")
    
    # Compute visibility score for each point (average across all views)
    visibility_scores = visibility_matrix.mean(axis=1)  # (N,)
    
    # Create scene
    scene = trimesh.Scene()
    
    # Color latent points by visibility score
    # 0.0 = red (invisible), 1.0 = green (fully visible)
    colors = np.zeros((len(canonical_coords), 3), dtype=np.uint8)
    colors[:, 0] = (255 * (1.0 - visibility_scores)).astype(np.uint8)  # Red component
    colors[:, 1] = (255 * visibility_scores).astype(np.uint8)  # Green component
    colors[:, 2] = 0  # Blue component
    
    # Create point cloud
    point_cloud = trimesh.PointCloud(vertices=canonical_coords, colors=colors)
    scene.add_geometry(point_cloud, node_name="latent_points")
    
    # Different color for each camera
    view_colors = [
        [255, 0, 0, 200],    # View 0: Red
        [0, 255, 0, 200],    # View 1: Green
        [0, 0, 255, 200],    # View 2: Blue
        [255, 255, 0, 200],  # View 3: Yellow
        [255, 0, 255, 200],  # View 4: Magenta
        [0, 255, 255, 200],  # View 5: Cyan
        [255, 128, 0, 200],  # View 6: Orange
        [128, 0, 255, 200],  # View 7: Purple
    ]
    
    # Add cameras (in canonical space) - simple sphere+direction line visualization
    for cam_idx, cam_pose in enumerate(canonical_camera_poses):
        camera_pos = cam_pose['camera_position']
        camera_forward = cam_pose.get('camera_forward', np.array([0, 0, 1]))
        
        color = view_colors[cam_idx % len(view_colors)]
        
        # Camera position marker (small sphere)
        camera_sphere = trimesh.creation.icosphere(subdivisions=1, radius=scale * 0.02)
        camera_sphere.apply_translation(camera_pos)
        camera_sphere.visual.face_colors = color
        scene.add_geometry(camera_sphere, node_name=f"camera_{cam_idx}")
        
        # Camera direction line (from camera position towards object)
        line_length = scale * 0.15
        line_end = camera_pos + camera_forward * line_length
        line_verts = np.array([camera_pos, line_end])
        line_colors = np.array([color, color])
        line_pc = trimesh.PointCloud(vertices=line_verts, colors=line_colors)
        scene.add_geometry(line_pc, node_name=f"camera_{cam_idx}_dir")
    
    # Add coordinate axes (length scaled accordingly)
    axis_length = scale * 0.3
    axis_vertices = np.array([
        [0, 0, 0], [axis_length, 0, 0],  # X
        [0, 0, 0], [0, axis_length, 0],  # Y
        [0, 0, 0], [0, 0, axis_length],  # Z
    ])
    axis_colors = np.array([
        [255, 0, 0, 255], [255, 0, 0, 255],  # X - Red
        [0, 255, 0, 255], [0, 255, 0, 255],  # Y - Green (up)
        [0, 0, 255, 255], [0, 0, 255, 255],  # Z - Blue
    ])
    axis_pc = trimesh.PointCloud(axis_vertices, colors=axis_colors)
    scene.add_geometry(axis_pc, node_name="canonical_axes")
    
    # Save
    scene.export(str(output_path))
    logger.info(f"[Visibility Viz Canonical] Saved to: {output_path}")
    
    return output_path


def visualize_visibility_per_view(
    latent_coords: np.ndarray,
    visibility_matrix: np.ndarray,
    object_pose: dict,
    camera_poses: List[dict],
    output_dir: Path,
    num_views_per_image: int = 6,
) -> None:
    """
    Generate one image per view showing visible latent points from that view.
    Each image contains multiple view renderings (grid layout) to show visible parts.
    
    Args:
        latent_coords: Latent coordinates (N, 4) or (N, 3)
        visibility_matrix: Visibility matrix (N, N_views)
        object_pose: Object pose
        camera_poses: List of camera poses
        output_dir: Output directory
        num_views_per_image: Number of views per image (grid layout)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.error("matplotlib not installed, cannot create visibility images")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_views = visibility_matrix.shape[1]
    num_latents = visibility_matrix.shape[0]
    
    logger.info(f"[Visibility Images] Creating {num_views} visibility images...")
    
    # Use unified coordinate transform function
    # Transform chain: canonical (Z-up) -> Y-up -> scale -> rotate -> translate
    world_coords = apply_sam3d_pose_to_latent_coords(latent_coords, object_pose)
    
    # Compute grid layout (2 columns x 3 rows = 6 views)
    n_cols = 2
    n_rows = (num_views_per_image + n_cols - 1) // n_cols
    
    # Generate one image for each view
    for view_idx in range(num_views):
        # Visibility for this view
        view_visibility = visibility_matrix[:, view_idx]  # (N,)
        
        # Create figure with multiple subplots (multiple view renderings)
        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        fig.suptitle(f'Visibility from View {view_idx}\n(Red=Invisible, Green=Visible)', 
                     fontsize=14, fontweight='bold')
        
        # Show num_views_per_image view renderings
        views_to_show = min(num_views_per_image, num_views)
        
        for subplot_idx in range(views_to_show):
            ax = fig.add_subplot(n_rows, n_cols, subplot_idx + 1, projection='3d')
            
            # Use different view visibility to filter points
            if subplot_idx < num_views:
                subplot_view_visibility = visibility_matrix[:, subplot_idx]
            else:
                subplot_view_visibility = view_visibility
            
            # Only show visible points (visibility > 0)
            visible_mask = subplot_view_visibility > 0.5
            visible_coords = world_coords[visible_mask]
            visible_scores = subplot_view_visibility[visible_mask]
            
            if len(visible_coords) > 0:
                # Color by visibility score
                colors_visible = np.zeros((len(visible_coords), 3))
                colors_visible[:, 0] = (1.0 - visible_scores)  # Red (invisible)
                colors_visible[:, 1] = visible_scores  # Green (visible)
                colors_visible[:, 2] = 0  # Blue
                
                # Draw point cloud
                ax.scatter(visible_coords[:, 0], visible_coords[:, 1], visible_coords[:, 2],
                          c=colors_visible, s=1, alpha=0.6)
            
            # Draw camera position and direction (if available)
            if subplot_idx < len(camera_poses):
                cam_pose = camera_poses[subplot_idx]
                camera_pos = cam_pose.get('camera_position')
                c2w = cam_pose.get('c2w')
                
                if camera_pos is not None:
                    # Camera position
                    ax.scatter([camera_pos[0]], [camera_pos[1]], [camera_pos[2]], 
                              c='magenta', s=50, marker='o', label='Camera')
                    
                    # Camera direction (Z axis)
                    if c2w is not None:
                        camera_z = c2w[:3, 2]
                        camera_end = camera_pos + camera_z * 0.1
                        ax.plot([camera_pos[0], camera_end[0]], 
                               [camera_pos[1], camera_end[1]], 
                               [camera_pos[2], camera_end[2]], 
                               'c-', linewidth=2, label='View Dir')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'View {subplot_idx}' + 
                        (f' (Visible: {visible_mask.sum()}/{num_latents})' if len(visible_coords) > 0 else ' (No visible points)'))
            ax.legend()
        
        plt.tight_layout()
        
        # Save image
        output_file = output_dir / f"visibility_view_{view_idx:02d}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[Visibility Images] Saved view {view_idx} visibility image: {output_file}")
    
    logger.info(f"[Visibility Images] Created {num_views} visibility images in {output_dir}")


def parse_image_names(image_names_str: Optional[str]) -> Optional[List[str]]:
    """Parse image names string."""
    if image_names_str is None or image_names_str == "":
        return None
    names = [x.strip() for x in image_names_str.split(",") if x.strip()]
    return names if names else None


def parse_attention_layers(layers_str: Optional[str]) -> Optional[List[int]]:
    """Parse attention layer indices from CLI string."""
    if layers_str is None:
        return None
    tokens = [token.strip() for token in layers_str.split(",") if token.strip()]
    if not tokens:
        return None
    indices: List[int] = []
    for token in tokens:
        try:
            indices.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid attention layer index: {token}") from exc
    return indices


def get_output_dir(
    input_path: Path, 
    mask_prompt: Optional[str] = None, 
    image_names: Optional[List[str]] = None,
    is_single_view: bool = False,
    # Stage 1 parameters
    stage1_weighting: bool = True,
    stage1_entropy_alpha: float = 30.0,
    # Stage 2 parameters
    stage2_weighting: bool = True,
    stage2_weight_source: str = "entropy",
    stage2_entropy_alpha: float = 30.0,
    stage2_visibility_alpha: float = 30.0,
    self_occlusion_tolerance: float = 4.0,
) -> Path:
    """
    Create output directory based on input path and parameters.
    
    Directory structure:
        visualization/{dataset_name}/{mask_prompt}/{detailed_name}/
    
    Example:
        visualization/quike/box/quike_box_multiview_s1ea60_s2entropy_a60_20231205_123456/
    """
    visualization_dir = Path("visualization")
    
    # Level 1: Dataset name (last component of input_path)
    dataset_name = input_path.name if input_path.is_dir() else input_path.parent.name
    
    # Level 2: Mask prompt (or "default" if not specified)
    mask_name = mask_prompt if mask_prompt else "default"
    
    # Level 3: Detailed folder name
    # Start with dataset_mask prefix
    dir_name = f"{dataset_name}_{mask_name}"
    
    # Add view info
    if is_single_view:
        if image_names and len(image_names) == 1:
            safe_name = image_names[0].replace("/", "_").replace("\\", "_")
            dir_name = f"{dir_name}_{safe_name}"
        else:
            dir_name = f"{dir_name}_single"
    elif image_names:
        num_views = len(image_names)
        dir_name = f"{dir_name}_{num_views}v"
    else:
        dir_name = f"{dir_name}_mv"
    
    # Add Stage 1 weighting parameters
    if stage1_weighting:
        s1_alpha_str = f"{stage1_entropy_alpha:g}"
        dir_name = f"{dir_name}_s1a{s1_alpha_str}"
    else:
        dir_name = f"{dir_name}_s1off"
    
    # Add Stage 2 weighting parameters
    if stage2_weighting:
        if stage2_weight_source == "entropy":
            s2_alpha_str = f"{stage2_entropy_alpha:g}"
            dir_name = f"{dir_name}_s2e{s2_alpha_str}"
        elif stage2_weight_source == "visibility":
            s2_alpha_str = f"{stage2_visibility_alpha:g}"
            tol_str = f"{self_occlusion_tolerance:g}"
            dir_name = f"{dir_name}_s2v{s2_alpha_str}t{tol_str}"
        elif stage2_weight_source == "mixed":
            e_str = f"{stage2_entropy_alpha:g}"
            v_str = f"{stage2_visibility_alpha:g}"
            dir_name = f"{dir_name}_s2m_e{e_str}v{v_str}"
    else:
        dir_name = f"{dir_name}_s2off"
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{dir_name}_{timestamp}"
    
    # Create hierarchical directory structure
    output_dir = visualization_dir / dataset_name / mask_name / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    return output_dir


def run_weighted_inference(
    input_path: Path,
    mask_prompt: Optional[str] = None,
    image_names: Optional[List[str]] = None,
    seed: int = 42,
    stage1_steps: int = 50,
    stage2_steps: int = 25,
    decode_formats: List[str] = None,
    model_tag: str = "hf",
    # Stage 1 (Shape) Weighting parameters
    stage1_weighting: bool = True,
    stage1_entropy_layer: int = 9,
    stage1_entropy_alpha: float = 30.0,
    # Stage 2 (Texture) Weighting parameters
    stage2_weighting: bool = True,
    stage2_weight_source: str = "entropy",  # "entropy", "visibility", "mixed"
    stage2_entropy_alpha: float = 30.0,
    stage2_visibility_alpha: float = 30.0,
    stage2_attention_layer: int = 6,
    stage2_attention_step: int = 0,
    stage2_min_weight: float = 0.001,
    stage2_weight_combine_mode: str = "average",  # "average" or "multiply"
    stage2_visibility_weight_ratio: float = 0.5,  # Ratio for averaging in mixed mode
    # Visualization
    visualize_weights: bool = False,
    save_attention: bool = False,
    attention_layers_to_save: Optional[List[int]] = None,
    save_coords: bool = True,  # Default True for weighted inference
    save_stage2_init: bool = False,
    # DA3 integration
    da3_output_path: Optional[str] = None,
    merge_da3_glb: bool = False,
    overlay_pointmap: bool = False,
    enable_latent_visibility: bool = False,
    self_occlusion_tolerance: float = 4.0,
):
    """
    Run weighted inference with adaptive multi-view fusion.
    
    Two-stage weighting:
    - Stage 1 (Shape): Uses entropy-based weighting for shape generation
    - Stage 2 (Texture): Uses entropy/visibility/mixed weighting for texture generation
    
    Args:
        input_path: Input data path
        mask_prompt: Mask folder name
        image_names: List of image names to use
        seed: Random seed for reproducibility
        stage1_steps: Stage 1 inference steps
        stage2_steps: Stage 2 inference steps
        decode_formats: Output formats (gaussian, mesh)
        model_tag: Model checkpoint tag
        
        Stage 1 Weighting:
            stage1_weighting: Enable Stage 1 entropy weighting (default: True)
            stage1_entropy_layer: Attention layer for Stage 1 (default: 9)
            stage1_entropy_alpha: Gibbs temperature for Stage 1 (default: 60.0)
        
        Stage 2 Weighting:
            stage2_weighting: Enable Stage 2 weighting (default: True)
            stage2_weight_source: "entropy", "visibility", or "mixed" (default: "entropy")
            stage2_entropy_alpha: Gibbs temperature for entropy (default: 60.0)
            stage2_visibility_alpha: Gibbs temperature for visibility (default: 60.0)
            stage2_attention_layer: Attention layer for Stage 2 (default: 6)
            stage2_attention_step: Diffusion step for attention (default: 0)
            stage2_min_weight: Minimum weight (default: 0.001)
        
        DA3 Integration:
            da3_output_path: Path to DA3 output (required for visibility weighting)
            merge_da3_glb: Merge SAM3D output with DA3 scene
            overlay_pointmap: Overlay SAM3D on View 0 pointmap
    """
    config_path = f"checkpoints/{model_tag}/pipeline.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    
    logger.info(f"Loading model: {config_path}")
    inference = Inference(config_path, compile=False)
    
    if hasattr(inference._pipeline, 'rendering_engine'):
        if inference._pipeline.rendering_engine != "pytorch3d":
            logger.warning(f"Rendering engine is set to {inference._pipeline.rendering_engine}, changing to pytorch3d")
            inference._pipeline.rendering_engine = "pytorch3d"
    
    logger.info(f"Loading data: {input_path}")
    if mask_prompt:
        logger.info(f"Mask prompt: {mask_prompt}")
    
    view_images, view_masks = load_images_and_masks_from_path(
        input_path=input_path,
        mask_prompt=mask_prompt,
        image_names=image_names,
    )
    
    num_views = len(view_images)
    logger.info(f"Successfully loaded {num_views} views")
    
    # Load external pointmaps from DA3 if provided
    view_pointmaps = None
    da3_dir = None  # DA3 output directory (for GLB merge)
    da3_extrinsics = None  # Camera extrinsics for alignment
    da3_intrinsics = None  # Camera intrinsics (if available)
    da3_pointmaps = None  # Raw pointmaps for alignment visualization
    if da3_output_path is not None:
        da3_path = Path(da3_output_path)
        da3_dir = da3_path.parent  # Store the directory for potential GLB merge
        
        # Strict mode: if da3_output is specified, it MUST be used successfully
        # Otherwise, raise an error to help debug issues
        
        if not da3_path.exists():
            raise FileNotFoundError(
                f"DA3 output file not found: {da3_path}\n"
                f"Please run: python scripts/run_da3.py --image_dir <your_image_dir> --output_dir <output_dir>"
            )
        
        logger.info(f"Loading external pointmaps from DA3: {da3_path}")
        da3_data = np.load(da3_path)
        
        # Check if pointmaps_sam3d exists
        if "pointmaps_sam3d" not in da3_data:
            raise ValueError(
                f"No 'pointmaps_sam3d' found in DA3 output: {da3_path}\n"
                f"Available keys: {list(da3_data.keys())}\n"
                f"Please regenerate DA3 output with the latest run_da3.py script."
            )
        
        da3_pointmaps = da3_data["pointmaps_sam3d"]
        logger.info(f"  DA3 pointmaps shape: {da3_pointmaps.shape}")
        
        # Load extrinsics for alignment
        if "extrinsics" in da3_data:
            da3_extrinsics = da3_data["extrinsics"]
            logger.info(f"  DA3 extrinsics shape: {da3_extrinsics.shape}")
        
        # Load intrinsics if available
        if "intrinsics" in da3_data:
            da3_intrinsics = da3_data["intrinsics"]
            logger.info(f"  DA3 intrinsics shape: {da3_intrinsics.shape}")
        
        # Get inference image file names for matching
        # Re-detect the image names using the same logic as load_images_and_masks
        if mask_prompt:
            images_dir = input_path / "images"
        else:
            images_dir = input_path
        
        # Get inference image order (using natural sort)
        def natural_sort_key(path):
            stem = path.stem
            try:
                return (0, int(stem), stem)
            except ValueError:
                return (1, 0, stem)
        
        if image_names:
            # User specified image names
            inference_image_names = list(image_names)
        else:
            # Auto-detect
            inf_image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            inf_image_files = [f for f in inf_image_files if "_mask" not in f.name]
            inf_image_files = sorted(inf_image_files, key=natural_sort_key)
            inference_image_names = [f.stem for f in inf_image_files]
        
        logger.info(f"  Inference image order: {inference_image_names}")
        
        # Build DA3 filename -> index mapping
        da3_file_mapping = {}
        if "image_files" in da3_data:
            da3_image_files = da3_data["image_files"]
            for idx, filepath in enumerate(da3_image_files):
                filename = Path(str(filepath)).stem  # Extract filename without extension
                da3_file_mapping[filename] = idx
            logger.info(f"  DA3 image order: {[Path(str(f)).stem for f in da3_image_files]}")
        
        # Match pointmaps by filename
        view_pointmaps = []
        matched_da3_extrinsics = []
        matched_da3_intrinsics = []
        
        for inf_name in inference_image_names:
            if inf_name in da3_file_mapping:
                da3_idx = da3_file_mapping[inf_name]
                view_pointmaps.append(da3_pointmaps[da3_idx])
                if da3_extrinsics is not None:
                    matched_da3_extrinsics.append(da3_extrinsics[da3_idx])
                if da3_intrinsics is not None:
                    matched_da3_intrinsics.append(da3_intrinsics[da3_idx])
                logger.info(f"    Matched: inference '{inf_name}' -> DA3 index {da3_idx}")
            else:
                # Fallback: use index-based matching if filename not found
                # This handles the case where DA3 doesn't have image_files or names don't match
                fallback_idx = inference_image_names.index(inf_name)
                if fallback_idx < da3_pointmaps.shape[0]:
                    view_pointmaps.append(da3_pointmaps[fallback_idx])
                    if da3_extrinsics is not None:
                        matched_da3_extrinsics.append(da3_extrinsics[fallback_idx])
                    if da3_intrinsics is not None:
                        matched_da3_intrinsics.append(da3_intrinsics[fallback_idx])
                    logger.warning(f"    Fallback: inference '{inf_name}' -> DA3 index {fallback_idx} (filename not found in DA3)")
                else:
                    raise ValueError(
                        f"Cannot match pointmap for image '{inf_name}'!\n"
                        f"  DA3 has {da3_pointmaps.shape[0]} pointmaps\n"
                        f"  Inference needs {num_views} views\n"
                        f"Please ensure DA3 was run on the SAME images."
                    )
        
        # Update extrinsics and intrinsics to matched order
        if matched_da3_extrinsics:
            da3_extrinsics = np.array(matched_da3_extrinsics)
        if matched_da3_intrinsics:
            da3_intrinsics = np.array(matched_da3_intrinsics)

        logger.info(f"  Successfully loaded and matched {len(view_pointmaps)} external pointmaps from DA3")

        # Extract global object point cloud from scene.glb using all view masks
        # This provides a consistent geometric reference for alignment
        object_pointcloud_global = None
        if da3_extrinsics is not None and da3_intrinsics is not None:
            scene_glb_path = da3_dir / "scene.glb"
            if scene_glb_path.exists():
                logger.info(f"\n{'='*60}")
                logger.info(f"Extracting object point cloud from scene.glb")
                logger.info(f"{'='*60}")

                # Prepare masks (convert from PIL to numpy if needed)
                view_masks_np = []
                for mask in view_masks:
                    if hasattr(mask, 'convert'):  # PIL Image
                        mask_np = np.array(mask.convert('L')) / 255.0
                    else:
                        mask_np = np.array(mask)
                    view_masks_np.append(mask_np)

                # Get image shape from first mask
                H, W = view_masks_np[0].shape

                # Extract object points from global scene
                object_pointcloud_global = extract_object_pointcloud_from_scene(
                    scene_glb_path,
                    view_masks_np,
                    da3_extrinsics,
                    da3_intrinsics,
                    (H, W),
                )

                if object_pointcloud_global is not None:
                    logger.info(f"Successfully extracted {len(object_pointcloud_global)} object points")

                    # Project global object pointcloud to view 0 camera space for alignment
                    # This provides a consistent geometric reference for all object alignments
                    logger.info(f"Projecting global object pointcloud to view 0 camera space...")
                    object_pointcloud_view0 = project_pointcloud_to_camera(
                        object_pointcloud_global,
                        da3_extrinsics[0],  # View 0 extrinsic
                        da3_intrinsics[0],  # View 0 intrinsic
                        (H, W),
                    )

                    # Convert to torch (M, 3) format
                    # Extract non-zero points (those that were successfully projected)
                    nonzero_mask = np.any(object_pointcloud_view0 != 0, axis=-1)  # (H, W)
                    object_points_camera = object_pointcloud_view0[nonzero_mask]  # (M, 3)
                    object_pointcloud_camera_tensor = torch.from_numpy(object_points_camera).float()

                    logger.info(f"  Projected {len(object_points_camera)} object points to view 0 camera space")
                    logger.info(f"  Object point cloud shape: {object_pointcloud_camera_tensor.shape}")
                else:
                    logger.warning("Failed to extract object point cloud from scene.glb")
                    object_pointcloud_camera_tensor = None
            else:
                logger.warning(f"scene.glb not found at {scene_glb_path}, cannot extract object point cloud")
                object_pointcloud_camera_tensor = None
        else:
            object_pointcloud_camera_tensor = None
    else:
        object_pointcloud_camera_tensor = None
    
    is_single_view = num_views == 1
    
    if is_single_view:
        logger.warning("Single view detected - weighting is not applicable, using standard inference")
        stage1_weighting = False
        stage2_weighting = False
    
    # Check parameter conflicts
    # 1. --merge_da3_glb requires --da3_output
    if merge_da3_glb and da3_output_path is None:
        raise ValueError(
            "Parameter conflict: --merge_da3_glb requires --da3_output.\n"
            "  --merge_da3_glb needs DA3's scene.glb to merge with SAM3D output.\n"
            "  Please provide: --da3_output <path_to_da3_output.npz>\n"
            "  Or remove --merge_da3_glb."
        )
    
    # 2. visibility/mixed weight source requires --da3_output
    if stage2_weight_source in ["visibility", "mixed"] and da3_output_path is None:
        raise ValueError(
            f"Parameter conflict: --stage2_weight_source {stage2_weight_source} requires --da3_output.\n"
            "  Visibility weighting needs DA3's camera extrinsics.\n"
            "  Please provide: --da3_output <path_to_da3_output.npz>\n"
            "  Or use --stage2_weight_source entropy."
        )
    
    output_dir = get_output_dir(
        input_path=input_path, 
        mask_prompt=mask_prompt, 
        image_names=image_names, 
        is_single_view=is_single_view,
        # Stage 1 parameters
        stage1_weighting=stage1_weighting,
        stage1_entropy_alpha=stage1_entropy_alpha,
        # Stage 2 parameters
        stage2_weighting=stage2_weighting,
        stage2_weight_source=stage2_weight_source,
        stage2_entropy_alpha=stage2_entropy_alpha,
        stage2_visibility_alpha=stage2_visibility_alpha,
        self_occlusion_tolerance=self_occlusion_tolerance,
    )
    
    # Setup logging
    log_file = output_dir / "inference.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
    )
    
    decode_formats = decode_formats or ["gaussian", "mesh"]
    
    # ========================================
    # Print experiment configuration
    # ========================================
    logger.info("=" * 70)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Input: {input_path}")
    logger.info(f"Mask prompt: {mask_prompt}")
    logger.info(f"Number of views: {num_views}")
    logger.info(f"Seed: {seed}")
    logger.info("-" * 70)
    logger.info("Stage 1 (Shape) Weighting:")
    logger.info(f"  Enabled: {stage1_weighting}")
    if stage1_weighting:
        logger.info(f"  Entropy Layer: {stage1_entropy_layer}")
        logger.info(f"  Entropy Alpha: {stage1_entropy_alpha}")
    logger.info("-" * 70)
    logger.info("Stage 2 (Texture) Weighting:")
    logger.info(f"  Enabled: {stage2_weighting}")
    if stage2_weighting:
        logger.info(f"  Weight Source: {stage2_weight_source}")
        logger.info(f"  Entropy Alpha: {stage2_entropy_alpha}")
        logger.info(f"  Attention Layer: {stage2_attention_layer}")
        logger.info(f"  Attention Step: {stage2_attention_step}")
        if stage2_weight_source in ["visibility", "mixed"]:
            logger.info(f"  Visibility Alpha: {stage2_visibility_alpha}")
            logger.info(f"  Self-occlusion Tolerance: {self_occlusion_tolerance}")
        if stage2_weight_source == "mixed":
            logger.info(f"  Combine Mode: {stage2_weight_combine_mode}")
            logger.info(f"  Visibility Weight Ratio: {stage2_visibility_weight_ratio}")
    logger.info("-" * 70)
    logger.info(f"DA3 Output: {da3_output_path}")
    logger.info(f"Merge DA3 GLB: {merge_da3_glb}")
    logger.info(f"Overlay Pointmap: {overlay_pointmap}")
    logger.info("=" * 70)
    
    # Create visibility callback if needed (uses self-occlusion / DDA ray tracing)
    visibility_callback = None
    if stage2_weight_source in ["visibility", "mixed"] and stage2_weighting:
        if da3_extrinsics is None:
            raise ValueError(
                f"stage2_weight_source='{stage2_weight_source}' requires DA3 output with camera extrinsics.\n"
                f"Please provide --da3_output with valid extrinsics."
            )
        
        # Create callback for visibility based on self-occlusion (DDA ray tracing)
        def create_visibility_callback(camera_poses_data, tolerance):
            """Create a visibility callback using self-occlusion (DDA ray tracing)."""
            def visibility_callback_impl(downsampled_coords, num_views, object_pose):
                """
                Compute visibility matrix for downsampled coords using self-occlusion.
                
                Args:
                    downsampled_coords: (N, 4) coords in voxel space [batch, z, y, x]
                    num_views: Number of views
                    object_pose: Dict with 'scale', 'rotation', 'translation'
                
                Returns:
                    visibility_matrix: (num_views, N) matrix where 1=visible, 0=self-occluded
                """
                from scipy.spatial.transform import Rotation
                
                # Helper to convert tensors to numpy
                def _to_np(x):
                    if hasattr(x, 'cpu'):
                        return x.cpu().numpy()
                    return np.array(x)
                
                # Convert object_pose tensors to numpy
                obj_scale = _to_np(object_pose.get('scale', [1, 1, 1])).flatten()
                obj_rotation = _to_np(object_pose.get('rotation', [1, 0, 0, 0])).flatten()
                obj_translation = _to_np(object_pose.get('translation', [0, 0, 0])).flatten()
                
                # Extract camera positions in canonical space
                camera_positions_canonical = []
                for cam_pose in camera_poses_data:
                    # Transform camera from world to canonical space
                    cam_pos_world = np.array(cam_pose.get('camera_position', [0, 0, 0])).flatten()
                    
                    # World to canonical: inverse of object pose
                    # canonical_pos = R_obj^T @ (world_pos - t_obj)
                    R_obj = Rotation.from_quat([obj_rotation[1], obj_rotation[2], 
                                                obj_rotation[3], obj_rotation[0]]).as_matrix()
                    cam_pos_canonical = R_obj.T @ (cam_pos_world - obj_translation)
                    camera_positions_canonical.append(cam_pos_canonical)
                
                # Compute self-occlusion for all views
                # visibility_matrix: (N, num_views) where 1=visible, 0=occluded
                visibility_matrix = compute_self_occlusion_for_all_views(
                    latent_coords=downsampled_coords,
                    camera_positions_canonical=camera_positions_canonical,
                    scale=float(obj_scale[0]),
                    grid_size=64,
                    neighbor_tolerance=tolerance,
                )
                
                # Transpose to (num_views, N) to match expected format
                vis_matrix = visibility_matrix.T
                return vis_matrix
            
            return visibility_callback_impl
        
        # Convert da3_extrinsics to camera_poses format
        camera_poses_for_visibility = convert_da3_extrinsics_to_camera_poses(da3_extrinsics)
        logger.info(f"Converted {len(camera_poses_for_visibility)} camera poses for visibility callback")
        
        visibility_callback = create_visibility_callback(
            camera_poses_data=camera_poses_for_visibility,
            tolerance=self_occlusion_tolerance,
        )
        logger.info(f"Visibility callback created (self-occlusion based), tolerance={self_occlusion_tolerance}")
    
    # Setup weighting config for Stage 2
    weighting_config = WeightingConfig(
        use_entropy=stage2_weighting,
        entropy_alpha=stage2_entropy_alpha,
        attention_layer=stage2_attention_layer,
        attention_step=stage2_attention_step,
        min_weight=stage2_min_weight,
        # Visibility-related parameters
        weight_source=stage2_weight_source,
        visibility_alpha=stage2_visibility_alpha,
        weight_combine_mode=stage2_weight_combine_mode,
        visibility_weight_ratio=stage2_visibility_weight_ratio,
        visibility_callback=visibility_callback,
    )
    
    # Setup attention logger (only if explicitly requested for analysis)
    attention_logger: Optional[CrossAttentionLogger] = None
    if save_attention:
        # Only save attention when explicitly requested (for analysis purposes)
        layers_to_hook = attention_layers_to_save or [stage2_attention_layer]
        if stage2_attention_layer not in layers_to_hook:
            layers_to_hook.append(stage2_attention_layer)
        
        attention_dir = output_dir / "attention"
        attention_logger = CrossAttentionLogger(
            attention_dir,
            enabled_stages=["ss", "slat"],  # Enable both Stage 1 (SS) and Stage 2 (SLAT)
            layer_indices=layers_to_hook,
            save_coords=save_coords,
        )
        attention_logger.attach_to_pipeline(inference._pipeline)
        logger.info(f"Cross-attention logging enabled → layers={layers_to_hook}, save_coords={save_coords}")
    
    # Note: Weighting uses in-memory AttentionCollector, not CrossAttentionLogger
    # The attention for weight computation is collected directly during warmup pass
    
    # Prepare pointmap with embedded object point cloud if available
    pointmap_for_inference = None
    if view_pointmaps is not None and len(view_pointmaps) > 0:
        pointmap_for_inference = view_pointmaps[0]  # Use view 0 pointmap

    # Inject object point cloud into pipeline via a custom mechanism
    # We'll store it in the pipeline temporarily for compute_pointmap to access
    if object_pointcloud_camera_tensor is not None:
        logger.info("Passing object point cloud to pipeline for alignment")
        # Temporarily store in pipeline instance
        inference._pipeline._temp_object_pointcloud_camera = object_pointcloud_camera_tensor

    # Run inference
    if is_single_view:
        logger.info("Single-view inference mode")
        image = view_images[0]
        mask = view_masks[0] if view_masks else None
        result = inference._pipeline.run(
            image,
            mask,
            seed=seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
            stage1_inference_steps=stage1_steps,
            stage2_inference_steps=stage2_steps,
            decode_formats=decode_formats,
            attention_logger=attention_logger,
            pointmap=pointmap_for_inference,
        )
        weight_manager = None
    else:
        s1_mode = "weighted" if stage1_weighting else "average"
        s2_mode = f"weighted ({stage2_weight_source})" if stage2_weighting else "average"
        logger.info(f"Multi-view inference mode: Stage1={s1_mode}, Stage2={s2_mode}")
        if view_pointmaps is not None:
            logger.info(f"Using external pointmaps from DA3")
        result = inference._pipeline.run_multi_view(
            view_images=view_images,
            view_masks=view_masks,
            view_pointmaps=view_pointmaps,  # External pointmaps from DA3
            seed=seed,
            mode="multidiffusion",
            stage1_inference_steps=stage1_steps,
            stage2_inference_steps=stage2_steps,
            decode_formats=decode_formats,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
            attention_logger=attention_logger,
            # Pass weighting config for weighted fusion (Stage 2)
            weighting_config=weighting_config if stage2_weighting else None,
            # Save Stage 2 init for stability analysis
            save_stage2_init=save_stage2_init,
            save_stage2_init_path=output_dir / "stage2_init.pt" if save_stage2_init else None,
            # Stage 1 weighting parameters
            ss_weighting=stage1_weighting,
            ss_entropy_layer=stage1_entropy_layer,
            ss_entropy_alpha=stage1_entropy_alpha,
            ss_warmup_steps=1,  # Fixed at 1 for stability
        )
        weight_manager = result.get("weight_manager")
        
        # Log if stage2_init was saved
        if save_stage2_init and (output_dir / "stage2_init.pt").exists():
            logger.info(f"Stage 2 initial latent saved to: {output_dir / 'stage2_init.pt'}")
        
        # Latent visibility computation
        # Note: visibility analysis only uses DA3 GT camera poses, not estimated poses
        if enable_latent_visibility:
            logger.info("=" * 60)
            logger.info("Computing Latent Visibility (using DA3 GT camera poses)")
            logger.info("=" * 60)
            
            # Check required data
            if 'coords' not in result:
                logger.warning("[Visibility] No 'coords' found in result, cannot compute visibility")
            elif da3_extrinsics is None:
                logger.warning("[Visibility] Cannot compute visibility: requires --da3_output with extrinsics. "
                             "Visibility analysis only uses GT camera poses, not estimated poses.")
            else:
                latent_coords = result['coords']  # (N, 4) or (N, 3)
                
                # Get object pose (use Stage 1 output, this is the true object pose)
                # Stage 2 refined_poses is for camera pose estimation, not the true object pose
                object_pose = {}
                if 'scale' in result:
                    object_pose['scale'] = result['scale'].cpu().numpy() if torch.is_tensor(result['scale']) else result['scale']
                if 'rotation' in result:
                    object_pose['rotation'] = result['rotation'].cpu().numpy() if torch.is_tensor(result['rotation']) else result['rotation']
                if 'translation' in result:
                    object_pose['translation'] = result['translation'].cpu().numpy() if torch.is_tensor(result['translation']) else result['translation']
                logger.info("[Visibility] Using Stage 1 pose for object (true object pose)")
                
                if not object_pose:
                    logger.warning("[Visibility] No object pose found in result, using default")
                    object_pose = {
                        'scale': np.array([1.0, 1.0, 1.0]),
                        'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                        'translation': np.array([0.0, 0.0, 0.0]),
                    }
                
                # Only use DA3 GT camera poses (not estimated poses)
                # Note: need to transform DA3 GT camera poses to View 0 coordinate system (consistent with object pose)
                logger.info("[Visibility] Using DA3 GT camera poses (extrinsics)")
                camera_poses_da3 = convert_da3_extrinsics_to_camera_poses(da3_extrinsics)
                
                # Transform DA3 camera poses to View 0 coordinate system
                # DA3 View 0 w2c represents: DA3 world coordinates -> View 0 camera coordinates
                # So w2c_view0 @ P_world = P_view0
                # For camera i, its c2w in DA3 world coordinates is c2w_i_world
                # To transform to View 0 coordinate system: c2w_i_view0 = w2c_view0 @ c2w_i_world
                w2c_view0_da3 = da3_extrinsics[0]  # DA3 View 0 w2c (3, 4) or (4, 4)
                if w2c_view0_da3.shape == (3, 4):
                    w2c_view0_da3_44 = np.eye(4)
                    w2c_view0_da3_44[:3, :] = w2c_view0_da3
                    w2c_view0_da3 = w2c_view0_da3_44
                
                # Transform all DA3 camera poses to View 0 coordinate system, from OpenCV to PyTorch3D space
                # 
                # Coordinate system notes：
                # - DA3 extrinsics in OpenCV space: X-right, Y-down, Z-forward
                # - SAM3D pose in PyTorch3D space: X-left, Y-up, Z-forward
                # - Need OpenCV -> PyTorch3D transform: (-x, -y, z)
                #
                # For c2w matrix, need to transform both rotation and translation:
                # M_p3d = M_cv_to_p3d @ M_cv @ M_p3d_to_cv
                # where M_cv_to_p3d = diag(-1, -1, 1)
                opencv_to_pytorch3d = np.diag([-1.0, -1.0, 1.0])
                
                camera_poses = []
                for da3_cam_pose in camera_poses_da3:
                    # DA3 camera c2w in DA3 world coordinates
                    c2w_da3_world = da3_cam_pose['c2w']
                    
                    # Transform to View 0 coordinate system (still in OpenCV space)
                    # P_view0 = w2c_view0 @ P_world = w2c_view0 @ c2w_i_world @ P_cam_i
                    # So c2w_i_view0 = w2c_view0 @ c2w_i_world
                    c2w_view0_cv = w2c_view0_da3 @ c2w_da3_world
                    
                    # Transform from OpenCV space to PyTorch3D space
                    # Rotation: R_p3d = M @ R_cv @ M^T
                    # Translation: t_p3d = M @ t_cv
                    R_cv = c2w_view0_cv[:3, :3]
                    t_cv = c2w_view0_cv[:3, 3]
                    
                    R_p3d = opencv_to_pytorch3d @ R_cv @ opencv_to_pytorch3d.T
                    t_p3d = opencv_to_pytorch3d @ t_cv
                    
                    c2w_view0 = np.eye(4)
                    c2w_view0[:3, :3] = R_p3d
                    c2w_view0[:3, 3] = t_p3d
                    
                    camera_poses.append({
                        'view_idx': da3_cam_pose['view_idx'],
                        'c2w': c2w_view0,
                        'w2c': np.linalg.inv(c2w_view0),
                        'R_c2w': c2w_view0[:3, :3],
                        't_c2w': c2w_view0[:3, 3],
                        'camera_position': c2w_view0[:3, 3],
                    })
                
                logger.info("[Visibility] Converted DA3 GT camera poses to View 0 coordinate system (PyTorch3D space)")
                
                # Compute visibility (using self-occlusion / DDA ray tracing)
                visibility_result = compute_latent_visibility(
                    latent_coords=latent_coords.cpu().numpy() if torch.is_tensor(latent_coords) else latent_coords,
                    object_pose=object_pose,
                    camera_poses=camera_poses,
                    self_occlusion_tolerance=self_occlusion_tolerance,
                )
                
                if visibility_result is not None:
                    result['latent_visibility'] = visibility_result['visibility_matrix']
                    result['visibility_canonical_data'] = visibility_result  # Save full data for subsequent analysis
                    
                    # GLB visualization (in canonical space) - latent_visibility.glb
                    viz_path = visualize_latent_visibility(
                        visibility_result=visibility_result,
                        output_path=output_dir / "latent_visibility.glb",
                    )
                    
                    if viz_path:
                        logger.info(f"✓ Latent visibility GLB (canonical) saved to: {viz_path}")
                    
                    # Per-view visibility GLB visualization (one GLB file per view)
                    viz_dir = output_dir / "latent_visibility_per_view"
                    visualize_self_occlusion_per_view(
                        self_occlusion_matrix=visibility_result['visibility_matrix'],
                        visibility_result=visibility_result,
                        output_dir=viz_dir,
                    )
                    
                    if viz_dir.exists():
                        logger.info(f"✓ Latent visibility per-view GLB files saved to: {viz_dir}")
                    
                    # Statistics
                    visibility_matrix = visibility_result['visibility_matrix']
                    for view_idx in range(visibility_matrix.shape[1]):
                        visible_ratio = visibility_matrix[:, view_idx].mean()
                        logger.info(f"  View {view_idx}: {visible_ratio*100:.1f}% visible")
                    
                else:
                    logger.warning("[Visibility] Visibility computation returned None")
    
    # Save results
    saved_files = []
    
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    s1_str = "weighted" if stage1_weighting else "average"
    s2_str = f"weighted ({stage2_weight_source})" if stage2_weighting else "average"
    print(f"Mode: Stage1={s1_str}, Stage2={s2_str}")
    print(f"Generated coordinates: {result['coords'].shape[0] if 'coords' in result else 'N/A'}")
    print(f"{'='*60}")
    
    glb_path = None
    if 'glb' in result and result['glb'] is not None:
        glb_path = output_dir / "result.glb"
        result['glb'].export(str(glb_path))
        saved_files.append("result.glb")
        print(f"✓ GLB file saved to: {glb_path}")
        
        # Merge with DA3 scene.glb if requested (with alignment)
        if merge_da3_glb and da3_dir is not None:
            # Prepare pose parameters for alignment
            # Note: SAM3D pose parameters are already in real-world scale
            sam3d_pose = {}
            if 'scale' in result:
                sam3d_pose['scale'] = result['scale'].cpu().numpy() if torch.is_tensor(result['scale']) else result['scale']
            if 'rotation' in result:
                sam3d_pose['rotation'] = result['rotation'].cpu().numpy() if torch.is_tensor(result['rotation']) else result['rotation']
            if 'translation' in result:
                sam3d_pose['translation'] = result['translation'].cpu().numpy() if torch.is_tensor(result['translation']) else result['translation']
            
            if sam3d_pose:
                # Merge with DA3's complete scene.glb
                merged_path = merge_glb_with_da3_aligned(
                    glb_path, da3_dir, sam3d_pose
                )
                if merged_path:
                    saved_files.append(merged_path.name)
                    print(f"✓ Merged GLB with DA3 scene saved to: {merged_path}")
            else:
                logger.warning("Cannot align: missing SAM3D pose parameters")
        elif merge_da3_glb and da3_dir is None:
            logger.warning("--merge_da3_glb specified but no DA3 output directory available (need --da3_output)")
        
        # Overlay SAM3D result on input pointmap for pose visualization
        # Only overlay on actually used pointmaps
        if overlay_pointmap:
            sam3d_pose = {}
            if 'scale' in result:
                sam3d_pose['scale'] = result['scale'].cpu().numpy() if torch.is_tensor(result['scale']) else result['scale']
            if 'rotation' in result:
                sam3d_pose['rotation'] = result['rotation'].cpu().numpy() if torch.is_tensor(result['rotation']) else result['rotation']
            if 'translation' in result:
                sam3d_pose['translation'] = result['translation'].cpu().numpy() if torch.is_tensor(result['translation']) else result['translation']
            
            if sam3d_pose:
                pointmap_data = None
                pm_scale_np = None
                pm_shift_np = None
                
                if 'raw_view_pointmaps' in result and result['raw_view_pointmaps']:
                    pointmap_data = result['raw_view_pointmaps'][0]
                    logger.info("[Overlay] Using raw_view_pointmaps[0] (metric)")
                elif 'pointmap' in result:
                    pointmap_data = result['pointmap']
                    logger.info("[Overlay] Using result['pointmap'] (metric)")
                elif 'view_ss_input_dicts' in result and result['view_ss_input_dicts']:
                    internal_pm = result['view_ss_input_dicts'][0].get('pointmap')
                    if internal_pm is not None:
                        pointmap_data = internal_pm
                        logger.info("[Overlay] Using normalized pointmap from view_ss_input_dicts")
                    # Try to read scale/shift from per-view input
                    pm_scale = result['view_ss_input_dicts'][0].get('pointmap_scale')
                    pm_shift = result['view_ss_input_dicts'][0].get('pointmap_shift')
                    if pm_scale is not None:
                        pm_scale_np = pm_scale.detach().cpu().numpy() if torch.is_tensor(pm_scale) else np.array(pm_scale)
                    if pm_shift is not None:
                        pm_shift_np = pm_shift.detach().cpu().numpy() if torch.is_tensor(pm_shift) else np.array(pm_shift)
                else:
                    logger.warning("Overlay: no pointmap source found")
                
                if pointmap_data is not None:
                    overlay_path = overlay_sam3d_on_pointmap(
                        glb_path,
                        pointmap_data,
                        sam3d_pose,
                        input_image=view_images[0] if view_images else None,
                        output_path=None,
                        pointmap_scale=pm_scale_np,
                        pointmap_shift=pm_shift_np,
                    )
                    if overlay_path:
                        saved_files.append(overlay_path.name)
                        print(f"✓ Overlay saved to: {overlay_path}")
                else:
                    logger.warning("Cannot create overlay: missing input pointmap")
    
    if 'gs' in result:
        output_path = output_dir / "result.ply"
        result['gs'].save_ply(str(output_path))
        saved_files.append("result.ply")
        print(f"✓ Gaussian Splatting (PLY) saved to: {output_path}")
    elif 'gaussian' in result:
        if isinstance(result['gaussian'], list) and len(result['gaussian']) > 0:
            output_path = output_dir / "result.ply"
            result['gaussian'][0].save_ply(str(output_path))
            saved_files.append("result.ply")
            print(f"✓ Gaussian Splatting (PLY) saved to: {output_path}")
    
    # Save pose and geometry parameters
    # These are important for converting from canonical space to metric/camera space
    # Reference: https://github.com/Stability-AI/stable-point-aware-3d/issues/XXX
    # - translation, rotation, scale: transform from canonical ([-0.5, 0.5]) to camera/metric space
    # - pointmap_scale: the scale factor used to normalize the pointmap (needed for real-world alignment)
    params = {}
    
    # Pose parameters
    if 'translation' in result:
        params['translation'] = result['translation'].cpu().numpy() if torch.is_tensor(result['translation']) else result['translation']
    if 'rotation' in result:
        params['rotation'] = result['rotation'].cpu().numpy() if torch.is_tensor(result['rotation']) else result['rotation']
    if 'scale' in result:
        params['scale'] = result['scale'].cpu().numpy() if torch.is_tensor(result['scale']) else result['scale']
    if 'downsample_factor' in result:
        params['downsample_factor'] = float(result['downsample_factor']) if torch.is_tensor(result['downsample_factor']) else result['downsample_factor']
    
    # Pointmap normalization parameters (for real-world alignment)
    if 'pointmap_scale' in result and result['pointmap_scale'] is not None:
        params['pointmap_scale'] = result['pointmap_scale'].cpu().numpy() if torch.is_tensor(result['pointmap_scale']) else result['pointmap_scale']
    if 'pointmap_shift' in result and result['pointmap_shift'] is not None:
        params['pointmap_shift'] = result['pointmap_shift'].cpu().numpy() if torch.is_tensor(result['pointmap_shift']) else result['pointmap_shift']
    
    # Geometry parameters
    if 'coords' in result:
        params['coords'] = result['coords'].cpu().numpy() if torch.is_tensor(result['coords']) else result['coords']
    
    if params:
        params_path = output_dir / "params.npz"
        np.savez(params_path, **params)
        saved_files.append("params.npz")
        print(f"✓ Parameters saved to: {params_path}")
    
    print(f"\n{'='*60}")
    print(f"All output files saved to: {output_dir}")
    print(f"Saved files: {', '.join(saved_files)}")
    print(f"{'='*60}")
    
    if attention_logger is not None:
        attention_logger.close()
    
    # Save weighting analysis if enabled
    if weight_manager is not None and visualize_weights:
        logger.info("Saving weight visualizations...")
        
        # Save weights and visualizations
        weights_dir = output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_data = weight_manager.get_analysis_data()
        weights_downsampled = analysis_data.get("weights", {})  # Weights in downsampled dimension
        weights_expanded = analysis_data.get("expanded_weights", {})  # Expanded weights
        entropy_per_view = analysis_data.get("entropy_per_view", {})
        original_coords = analysis_data.get("original_coords")  # Original coords
        downsampled_coords = analysis_data.get("downsampled_coords")  # Downsampled coords
        downsample_idx = analysis_data.get("downsample_idx")  # Index mapping
        
        # Log dimension info
        if weights_downsampled:
            sample_w = list(weights_downsampled.values())[0]
            logger.info(f"Downsampled weights dimension: {sample_w.shape[0]}")
        if weights_expanded:
            sample_w = list(weights_expanded.values())[0]
            logger.info(f"Expanded weights dimension: {sample_w.shape[0]}")
        if original_coords is not None:
            logger.info(f"Original coords shape: {original_coords.shape}")
        if downsampled_coords is not None:
            logger.info(f"Downsampled coords shape: {downsampled_coords.shape}")
        
        # Save weights as .pt file
        torch.save({
            "weights_downsampled": {k: v.cpu() for k, v in weights_downsampled.items()} if weights_downsampled else {},
            "weights_expanded": {k: v.cpu() for k, v in weights_expanded.items()} if weights_expanded else {},
            "entropy": {k: v.cpu() for k, v in entropy_per_view.items()} if entropy_per_view else {},
            "config": {
                "entropy_alpha": weighting_config.entropy_alpha,
                "attention_layer": weighting_config.attention_layer,
                "attention_step": weighting_config.attention_step,
            },
            "original_coords": original_coords.cpu() if original_coords is not None else None,
            "downsampled_coords": downsampled_coords.cpu() if downsampled_coords is not None else None,
            "downsample_idx": downsample_idx.cpu() if downsample_idx is not None else None,
        }, weights_dir / "fusion_weights.pt")
        
        logger.info(f"Saved fusion weights to {weights_dir / 'fusion_weights.pt'}")
        
        # ============ Weight Analysis ============
        analysis_log = weights_dir / "weight_analysis.log"
        with open(analysis_log, "w") as f:
            def log_analysis(msg):
                f.write(msg + "\n")
                logger.info(msg)
            
            log_analysis("=" * 60)
            log_analysis("Weight Analysis Report")
            log_analysis("=" * 60)
            log_analysis(f"Number of views: {len(weights_downsampled)}")
            log_analysis(f"Entropy alpha: {weighting_config.entropy_alpha}")
            log_analysis(f"Attention layer: {weighting_config.attention_layer}")
            log_analysis(f"Attention step: {weighting_config.attention_step}")
            
            # Entropy analysis
            if entropy_per_view:
                log_analysis("\n--- Entropy Analysis ---")
                entropy_values = []
                for view_idx, e in sorted(entropy_per_view.items()):
                    log_analysis(
                        f"  View {view_idx}: min={e.min():.4f}, max={e.max():.4f}, "
                        f"mean={e.mean():.4f}, std={e.std():.4f}"
                    )
                    entropy_values.append(e)
                
                # Cross-view entropy difference
                if len(entropy_values) > 1:
                    entropy_stack = torch.stack(entropy_values, dim=0)
                    view_std = entropy_stack.std(dim=0)
                    log_analysis(f"\n  Cross-view entropy std (per latent):")
                    log_analysis(f"    min={view_std.min():.4f}, max={view_std.max():.4f}, mean={view_std.mean():.4f}")
            
            # Weight analysis
            log_analysis("\n--- Weight Analysis (Downsampled) ---")
            for view_idx, w in sorted(weights_downsampled.items()):
                log_analysis(
                    f"  View {view_idx}: min={w.min():.6f}, max={w.max():.6f}, "
                    f"mean={w.mean():.6f}, std={w.std():.6f}"
                )
            
            # Check weight sum
            views = sorted(weights_downsampled.keys())
            weight_sum = sum(weights_downsampled[v] for v in views)
            log_analysis(f"\n  Weight sum: min={weight_sum.min():.4f}, max={weight_sum.max():.4f}")
            
            # Cross-view weight difference
            weight_stack = torch.stack([weights_downsampled[v] for v in views], dim=0)
            view_std = weight_stack.std(dim=0)
            log_analysis(f"\n  Cross-view weight std (per latent):")
            log_analysis(f"    min={view_std.min():.6f}, max={view_std.max():.6f}, mean={view_std.mean():.6f}")
            
            # Find latents with most weight variation
            top_k = 5
            top_indices = torch.argsort(view_std, descending=True)[:top_k]
            log_analysis(f"\n  Top {top_k} latents with most weight variation:")
            for idx in top_indices:
                log_analysis(f"    Latent {idx.item()}: std={view_std[idx]:.4f}")
                for v in views:
                    log_analysis(f"      View {v}: {weights_downsampled[v][idx]:.4f}")
            
            log_analysis("\n" + "=" * 60)
        
        logger.info(f"Saved weight analysis to {analysis_log}")
        
        # Generate visualizations
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # numpy is already imported at module level as np
            
            # Weight distribution histogram (downsampled)
            if weights_downsampled:
                fig, axes = plt.subplots(1, len(weights_downsampled), figsize=(4 * len(weights_downsampled), 4))
                if len(weights_downsampled) == 1:
                    axes = [axes]
                
                for ax, (view_idx, w) in zip(axes, sorted(weights_downsampled.items())):
                    w_np = w.cpu().numpy()
                    ax.hist(w_np, bins=50, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Weight')
                    ax.set_ylabel('Count')
                    ax.set_title(f'View {view_idx} (downsampled)\nmean={w_np.mean():.4f}, std={w_np.std():.4f}')
                
                plt.tight_layout()
                plt.savefig(weights_dir / 'weight_distribution_downsampled.png', dpi=150)
                plt.close()
                logger.info("Saved downsampled weight distribution plot")
            
            # Weight distribution histogram (expanded)
            if weights_expanded:
                fig, axes = plt.subplots(1, len(weights_expanded), figsize=(4 * len(weights_expanded), 4))
                if len(weights_expanded) == 1:
                    axes = [axes]
                
                for ax, (view_idx, w) in zip(axes, sorted(weights_expanded.items())):
                    w_np = w.cpu().numpy()
                    ax.hist(w_np, bins=50, alpha=0.7, edgecolor='black', color='green')
                    ax.set_xlabel('Weight')
                    ax.set_ylabel('Count')
                    ax.set_title(f'View {view_idx} (expanded)\nmean={w_np.mean():.4f}, std={w_np.std():.4f}')
                
                plt.tight_layout()
                plt.savefig(weights_dir / 'weight_distribution_expanded.png', dpi=150)
                plt.close()
                logger.info("Saved expanded weight distribution plot")
            
            # Entropy distribution histogram
            if entropy_per_view:
                fig, axes = plt.subplots(1, len(entropy_per_view), figsize=(4 * len(entropy_per_view), 4))
                if len(entropy_per_view) == 1:
                    axes = [axes]
                
                for ax, (view_idx, e) in zip(axes, sorted(entropy_per_view.items())):
                    e_np = e.cpu().numpy()
                    ax.hist(e_np, bins=50, alpha=0.7, edgecolor='black', color='orange')
                    ax.set_xlabel('Entropy')
                    ax.set_ylabel('Count')
                    ax.set_title(f'View {view_idx}\nmean={e_np.mean():.4f}, std={e_np.std():.4f}')
                
                plt.tight_layout()
                plt.savefig(weights_dir / 'entropy_distribution.png', dpi=150)
                plt.close()
                logger.info("Saved entropy distribution plot")
            
            # 3D visualization with DOWNSAMPLED coords (where attention is computed)
            if downsampled_coords is not None and weights_downsampled:
                coords_np = downsampled_coords.cpu().numpy()
                x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
                
                # Normalize coordinates
                x = (x - x.min()) / (x.max() - x.min() + 1e-6)
                y = (y - y.min()) / (y.max() - y.min() + 1e-6)
                z = (z - z.min()) / (z.max() - z.min() + 1e-6)
                
                for view_idx, w in sorted(weights_downsampled.items()):
                    w_np = w.cpu().numpy()
                    
                    # Robust normalization
                    vmin, vmax = np.percentile(w_np, [2, 98])
                    w_norm = np.clip((w_np - vmin) / (vmax - vmin + 1e-6), 0, 1)
                    
                    order = np.argsort(z)
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(
                        x[order], y[order], z[order],
                        c=w_norm[order],
                        cmap='viridis',
                        s=2,
                        alpha=0.6,
                    )
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'View {view_idx} Weight (Downsampled, {len(w_np)} points)')
                    
                    plt.colorbar(scatter, ax=ax, shrink=0.6, label='Weight')
                    plt.savefig(weights_dir / f'weight_3d_downsampled_view{view_idx:02d}.png', dpi=150)
                    plt.close()
                
                logger.info("Saved 3D weight visualizations (downsampled)")
            
            # 3D visualization with ORIGINAL coords (expanded weights)
            if original_coords is not None and weights_expanded:
                coords_np = original_coords.cpu().numpy()
                x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
                
                # Normalize coordinates
                x = (x - x.min()) / (x.max() - x.min() + 1e-6)
                y = (y - y.min()) / (y.max() - y.min() + 1e-6)
                z = (z - z.min()) / (z.max() - z.min() + 1e-6)
                
                for view_idx, w in sorted(weights_expanded.items()):
                    w_np = w.cpu().numpy()
                    
                    # Robust normalization
                    vmin, vmax = np.percentile(w_np, [2, 98])
                    w_norm = np.clip((w_np - vmin) / (vmax - vmin + 1e-6), 0, 1)
                    
                    order = np.argsort(z)
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(
                        x[order], y[order], z[order],
                        c=w_norm[order],
                        cmap='viridis',
                        s=0.5,  # Smaller points because more points
                        alpha=0.4,
                    )
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'View {view_idx} Weight (Expanded, {len(w_np)} points)')
                    
                    plt.colorbar(scatter, ax=ax, shrink=0.6, label='Weight')
                    plt.savefig(weights_dir / f'weight_3d_expanded_view{view_idx:02d}.png', dpi=150)
                    plt.close()
                
                logger.info("Saved 3D weight visualizations (expanded)")
                
        except ImportError as e:
            logger.warning(f"Could not generate visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D Objects Weighted Inference - Per-latent weighted multi-view fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic weighted inference (both stages weighted by default)
  python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3
  
  # Disable all weighting (simple average for both stages)
  python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 \
      --no_stage1_weighting --no_stage2_weighting
  
  # Only Stage 2 weighting (disable Stage 1)
  python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 \
      --no_stage1_weighting
  
  # With visibility weighting (requires DA3)
  python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 \
      --da3_output ./da3_outputs/example/da3_output.npz --stage2_weight_source visibility
        """
    )
    
    # Input/Output
    parser.add_argument("--input_path", type=str, required=True, help="Input path")
    parser.add_argument("--mask_prompt", type=str, default=None, help="Mask folder name")
    parser.add_argument("--image_names", type=str, default=None, help="Image names (comma-separated)")
    parser.add_argument("--model_tag", type=str, default="hf", help="Model tag")
    
    # Inference parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stage1_steps", type=int, default=50, help="Stage 1 (shape) inference steps")
    parser.add_argument("--stage2_steps", type=int, default=25, help="Stage 2 (texture) inference steps")
    parser.add_argument("--decode_formats", type=str, default="gaussian,mesh", help="Decode formats")
    
    # ========================================
    # Stage 1 (Shape) Weighting Parameters
    # ========================================
    parser.add_argument("--no_stage1_weighting", action="store_true",
                        help="Disable entropy-based weighting for Stage 1 (shape generation). "
                             "Uses simple average instead. By default, Stage 1 weighting is ENABLED.")
    parser.add_argument("--stage1_entropy_layer", type=int, default=9,
                        help="Stage 1: Which attention layer to use for weight computation (default: 9)")
    parser.add_argument("--stage1_entropy_alpha", type=float, default=30.0,
                        help="Stage 1: Gibbs temperature for entropy weighting (default: 30.0). "
                             "Higher = more contrast/aggressive, lower = more conservative.")
    
    # ========================================
    # Stage 2 (Texture) Weighting Parameters
    # ========================================
    parser.add_argument("--no_stage2_weighting", action="store_true",
                        help="Disable weighting for Stage 2 (texture generation). "
                             "Uses simple average instead. By default, Stage 2 weighting is ENABLED.")
    parser.add_argument("--stage2_weight_source", type=str, default="entropy",
                        choices=["entropy", "visibility", "mixed"],
                        help="Stage 2: Source for fusion weights. "
                             "'entropy' (default): Use attention entropy only. "
                             "'visibility': Use self-occlusion visibility (requires --da3_output). "
                             "'mixed': Combine entropy and visibility.")
    parser.add_argument("--stage2_entropy_alpha", type=float, default=30.0,
                        help="Stage 2: Gibbs temperature for entropy weighting (default: 30.0). "
                             "Higher = more contrast/aggressive, lower = more conservative.")
    parser.add_argument("--stage2_visibility_alpha", type=float, default=30.0,
                        help="Stage 2: Gibbs temperature for visibility weighting (default: 30.0). "
                             "Higher = more contrast/aggressive, lower = more conservative.")
    parser.add_argument("--stage2_attention_layer", type=int, default=6,
                        help="Stage 2: Which attention layer to use for weight computation (default: 6)")
    parser.add_argument("--stage2_attention_step", type=int, default=0,
                        help="Stage 2: Which diffusion step to use for weight computation (default: 0)")
    parser.add_argument("--stage2_min_weight", type=float, default=0.001,
                        help="Stage 2: Minimum weight to prevent complete zeroing (default: 0.001)")
    
    # Stage 2 mixed mode parameters
    parser.add_argument("--stage2_weight_combine_mode", type=str, default="average",
                        choices=["average", "multiply"],
                        help="Stage 2: How to combine entropy and visibility in 'mixed' mode. "
                             "'average': weighted average. 'multiply': multiply then normalize.")
    parser.add_argument("--stage2_visibility_weight_ratio", type=float, default=0.5,
                        help="Stage 2: Visibility ratio in 'average' mode (0.0-1.0). "
                             "0.0 = entropy only, 1.0 = visibility only.")
    
    # ========================================
    # Visualization Parameters
    # ========================================
    parser.add_argument("--visualize_weights", action="store_true",
                        help="Save weight and entropy visualizations")
    parser.add_argument("--save_attention", action="store_true",
                        help="Save all attention weights (for analysis)")
    parser.add_argument("--attention_layers", type=str, default=None,
                        help="Which layers to save attention for (comma-separated)")
    parser.add_argument("--save_stage2_init", action="store_true",
                        help="Save Stage 2 initial latent for iteration stability analysis")
    
    # ========================================
    # DA3 Integration Parameters
    # ========================================
    parser.add_argument("--da3_output", type=str, default=None,
                        help="Path to DA3 output npz file (from run_da3.py). "
                             "Required for visibility weighting and GLB merge.")
    parser.add_argument("--merge_da3_glb", action="store_true",
                        help="Merge SAM3D output GLB with DA3 scene.glb (requires --da3_output)")
    parser.add_argument("--overlay_pointmap", action="store_true",
                        help="Overlay SAM3D result on View 0 pointmap for pose verification")
    parser.add_argument("--compute_latent_visibility", action="store_true",
                        help="Compute and visualize latent visibility per view (requires --da3_output)")
    parser.add_argument("--self_occlusion_tolerance", type=float, default=4.0,
                        help="Tolerance for self-occlusion detection in voxel units (default: 4.0)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    image_names = parse_image_names(args.image_names)
    decode_formats = [fmt.strip() for fmt in args.decode_formats.split(",") if fmt.strip()]
    
    try:
        run_weighted_inference(
            input_path=input_path,
            mask_prompt=args.mask_prompt,
            image_names=image_names,
            seed=args.seed,
            stage1_steps=args.stage1_steps,
            stage2_steps=args.stage2_steps,
            decode_formats=decode_formats,
            model_tag=args.model_tag,
            # Stage 1 (Shape) weighting
            stage1_weighting=not args.no_stage1_weighting,
            stage1_entropy_layer=args.stage1_entropy_layer,
            stage1_entropy_alpha=args.stage1_entropy_alpha,
            # Stage 2 (Texture) weighting
            stage2_weighting=not args.no_stage2_weighting,
            stage2_weight_source=args.stage2_weight_source,
            stage2_entropy_alpha=args.stage2_entropy_alpha,
            stage2_visibility_alpha=args.stage2_visibility_alpha,
            stage2_attention_layer=args.stage2_attention_layer,
            stage2_attention_step=args.stage2_attention_step,
            stage2_min_weight=args.stage2_min_weight,
            stage2_weight_combine_mode=args.stage2_weight_combine_mode,
            stage2_visibility_weight_ratio=args.stage2_visibility_weight_ratio,
            # Visualization
            visualize_weights=args.visualize_weights,
            save_attention=args.save_attention,
            attention_layers_to_save=parse_attention_layers(args.attention_layers),
            save_stage2_init=args.save_stage2_init,
            # DA3 integration
            da3_output_path=args.da3_output,
            merge_da3_glb=args.merge_da3_glb,
            overlay_pointmap=args.overlay_pointmap,
            enable_latent_visibility=args.compute_latent_visibility,
            self_occlusion_tolerance=args.self_occlusion_tolerance,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

