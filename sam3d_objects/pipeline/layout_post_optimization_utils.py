# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams,
    TexturesVertex,
)
from pytorch3d.transforms import quaternion_to_matrix, Transform3d
import random
import open3d as o3d
from scipy.ndimage import label, binary_dilation, binary_fill_holes, binary_erosion, minimum_filter

def remove_small_regions(mask, min_area=100):
    """
    Remove small disconnected regions (floating points) from the mask.
    Keeps all regions with area >= min_area.
    """
    labeled_mask, num_labels = label(mask)
    cleaned = np.zeros_like(mask, dtype=bool)
    for i in range(1, num_labels + 1):
        region = (labeled_mask == i)
        if region.sum() >= min_area:
            cleaned |= region
    return cleaned

def is_near_image_border(mask, border_thickness=10):
    """
    Check if the mask touches the image border within a given thickness.
    """
    border_mask = np.zeros_like(mask, dtype=bool)
    border_mask[:border_thickness, :] = True
    border_mask[-border_thickness:, :] = True
    border_mask[:, :border_thickness] = True
    border_mask[:, -border_thickness:] = True
    return np.any(mask & border_mask)    

def is_occluded_by_others(mask, point_map, dilation_iter=2, z_thresh=0.05, filter_size=3):
    """
    Efficient occlusion detection using depth map and internal/external edges.
    """
    z_map = point_map[..., 2]
    if not np.any(mask):
        return False

    # Create internal and external edge masks
    eroded = binary_erosion(mask, iterations=dilation_iter)
    dilated = binary_dilation(mask, iterations=dilation_iter)

    internal_edge = mask & (~eroded)
    external_edge = dilated & (~mask)

    # Set invalid areas to +inf so they don't affect min-pooling
    z_ext = np.where(external_edge, z_map, np.inf)

    # Apply minimum filter to get local min depth around internal edges
    z_ext_min = minimum_filter(z_ext, size=filter_size, mode='constant', cval=np.inf)

    # Depth values at internal edge
    z_int = np.where(internal_edge, z_map, np.nan)

    # Compare depth difference
    diff = z_int - z_ext_min
    occlusion_mask = (diff > z_thresh) & (~np.isnan(diff))

    # return np.any(occlusion_mask)
    return np.sum(occlusion_mask) > 10

def has_internal_occlusion(mask, min_hole_area=20):
    """
    Check if the mask has internal holes or has been split into fragments.
    This may indicate internal occlusion.
    """
    # Check number of connected components
    labeled, num_features = label(mask)
    if num_features > 1:
        return True  # Mask is fragmented

    # Check for internal holes
    filled = binary_fill_holes(mask)
    holes = filled & (~mask)
    return np.sum(holes) >= min_hole_area

def check_occlusion(mask, point_map,
                    min_region_area=25,
                    border_thickness=5,
                    z_thresh=0.3,
                    min_hole_area=100):
    """
    Main function to check different types of occlusion for a given mask and 3D point map.
    """
    # clean mask by removing floating points
    cleaned_mask = remove_small_regions(mask, min_area=min_region_area)
    dilation_iter = 2
    filter_size = 2 * dilation_iter + 1

    # run occlusion checks
    return (
        is_near_image_border(cleaned_mask, border_thickness)
        or is_occluded_by_others(cleaned_mask, point_map, dilation_iter, z_thresh, filter_size)
        or has_internal_occlusion(cleaned_mask, min_hole_area)
    )

def get_mesh(Mesh, tfm_ori, device):
    mesh_vertices = Mesh.vertices.copy()
    # rotate mesh (from z-up to y-up)
    mesh_vertices = mesh_vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
    mesh_vertices = torch.from_numpy(mesh_vertices).float().cuda()
    points_world = tfm_ori.transform_points(mesh_vertices.unsqueeze(0))
    Mesh.vertices = points_world[0].cpu().numpy()  # pytorch3d, y-up, x left, z inwards.
    verts, faces_idx = load_and_simplify_mesh(Mesh, device)
    # === Add dummy white texture ===
    textures = TexturesVertex(verts_features=torch.ones_like(verts)[None])  # (1, V, 3)
    mesh = Meshes(verts=[verts], faces=[faces_idx], textures=textures)

    return mesh, faces_idx, textures


def get_mask_renderer(Mask, min_size, Intrinsics, device):
    orig_h, orig_w = Mask.shape[-2:]
    min_orig_size = min(orig_w, orig_h)
    scale_factor = min_size / min_orig_size
    mask = F.interpolate(
        Mask[None, None],
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
    )
    H, W = mask.shape[-2:]

    intrinsics = denormalize_f(Intrinsics.cpu().numpy(), H, W)
    cameras = PerspectiveCameras(
        focal_length=torch.tensor(
            [[intrinsics[0, 0], intrinsics[1, 1]]], device=device, dtype=torch.float32
        ),
        principal_point=torch.tensor(
            [[intrinsics[0, 2], intrinsics[1, 2]]], device=device, dtype=torch.float32
        ),
        image_size=torch.tensor([[H, W]], device=device, dtype=torch.float32),
        in_ndc=False,
        device=device,
    )
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=1e-6,
        faces_per_pixel=50,
        max_faces_per_bin=50000,
    )
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    return mask, renderer


def run_alignment(
    Point_Map,
    mask,
    mesh,
    center,
    faces_idx,
    textures,
    renderer,
    device,
    align_pm_coordinate=False,
):

    # from point map coordinate to pytorch3d
    target_object_points = Point_Map[mask[0, 0].bool()]
    if align_pm_coordinate:
        target_object_points[:, 0] *= -1
        target_object_points[:, 1] *= -1
    # Get rid of flying points
    thresh = 0.9
    depth_quantile = torch.quantile(target_object_points[:, 2], thresh)
    target_object_points = target_object_points[
        target_object_points[:, 2] <= depth_quantile
    ]
    flag_notgt = False

    if target_object_points.shape[0] == 0:
        flag_notgt = True
        return None, None, None, None, None, None, None, flag_notgt    

    source_points, target_points = mesh.verts_packed(), target_object_points
    # align to moge object points.
    height_src = torch.max(source_points[:, 1]) - torch.min(source_points[:, 1])
    height_tgt = torch.max(target_points[:, 1]) - torch.min(target_points[:, 1])
    scale_1 = height_tgt / height_src
    source_points *= scale_1
    center *= scale_1

    center_src = torch.mean(source_points, dim=0)
    center_tgt = torch.mean(target_points, dim=0)
    translation_1 = center_tgt - center_src

    source_points += translation_1
    center += translation_1

    # manually align based on moge point cloud.
    tfm1 = (
        Transform3d(device=device)
        .scale(scale_1.expand(3)[None])
        .translate(translation_1[None])
    )
    mesh = Meshes(verts=[source_points], faces=[faces_idx], textures=textures)
    rendered = renderer(mesh)
    ori_iou = compute_iou(rendered[..., 3][0][None, None], mask, threshold=0.5)
    final_iou = ori_iou.cpu().item()

    return source_points, target_points, center, tfm1, mesh, ori_iou, final_iou, flag_notgt


def apply_transform(mesh, center, quat, translation, scale):
    quat_normalized = quat / quat.norm()
    R = quaternion_to_matrix(quat_normalized)
    # transform to the world coordinate system center.
    verts = mesh.verts_packed() - center
    # perform operation
    verts = verts * scale
    verts = verts @ R.transpose(0, 1)
    # transform back to the original position after rotation.
    verts += center
    verts = verts + translation

    transformed_mesh = Meshes(
        verts=[verts], faces=[mesh.faces_packed()], textures=mesh.textures
    )
    return transformed_mesh


def compute_loss(rendered, mask_gt, loss_weights, quat, translation, scale):

    pred_mask = rendered[..., 3][0]
    # === 1. MSE Loss on mask ===
    loss_mask = F.mse_loss(pred_mask, mask_gt[0, 0])

    # === 2. Reg Loss on quaternion ===
    quat_normalized = quat / quat.norm()
    loss_reg_q = F.mse_loss(
        quat_normalized, torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device)
    )
    loss_reg_t = torch.norm(translation) ** 2
    loss_reg_s = (scale - 1.0) ** 2

    # === Total weighted loss ===
    total_loss = (
        loss_weights["mask"] * loss_mask
        + loss_weights["reg_q"] * loss_reg_q
        + loss_weights["reg_t"] * loss_reg_t
        + loss_weights["reg_s"] * loss_reg_s
    )

    return total_loss


def export_transformed_mesh_glb(
    verts, mesh_obj, center, quat, translation, scale, output_path
):
    quat_normalized = quat / quat.norm()

    R = quaternion_to_matrix(quat_normalized)
    # transform to the world coordinate system center.
    verts -= center
    # perform operations.
    verts = verts * scale
    verts = verts @ R.transpose(0, 1)
    # transform back to the original position after rotation.
    verts += center
    verts = verts + translation

    mesh_obj.vertices = verts.cpu().numpy()
    output_path = os.path.join(output_path, "result.glb")
    # import pdb
    # pdb.set_trace()
    mesh_obj.export(output_path)
    return


def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def layout_post_optimization_with_pointcloud(
    Mesh,
    Quaternion,
    Translation,
    Scale,
    Mask,
    object_pointcloud_camera: torch.Tensor,  # (M, 3) object points in view 0 camera space
    Intrinsics,
    Enable_shape_ICP=True,
    Enable_rendering_optimization=True,
    min_size=512,
    device=None,
):
    """
    Layout post-optimization using pre-extracted object point cloud.

    This function bypasses pointmap generation and directly uses the global
    object point cloud for alignment. This ensures all objects share the same
    geometric reference.

    Args:
        Mesh: Mesh object to align
        Quaternion: Initial rotation quaternion
        Translation: Initial translation
        Scale: Initial scale
        Mask: Object mask
        object_pointcloud_camera: (M, 3) object point cloud in view 0 camera space
        Intrinsics: Camera intrinsics
        Enable_shape_ICP: Whether to run ICP alignment
        Enable_rendering_optimization: Whether to run rendering optimization
        min_size: Minimum size for rendering
        device: Device to use

    Returns:
        Tuple of (quaternion, translation, scale, iou, flag_icp, flag_optim)
    """
    from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, Transform3d
    from pytorch3d.structures import Meshes
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    set_seed(100)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Ensure object_pointcloud_camera is on the correct device
    if object_pointcloud_camera is not None:
        object_pointcloud_camera = object_pointcloud_camera.to(device)

    # init transform and process mesh
    Rotation = quaternion_to_matrix(Quaternion.squeeze(1))
    center = Translation[0].clone()
    tfm_ori = compose_transform(scale=Scale, rotation=Rotation, translation=Translation)
    mesh, faces_idx, textures = get_mesh(Mesh, tfm_ori, device)

    # get mask and renderer
    mask, renderer = get_mask_renderer(Mask, min_size, Intrinsics, device)

    # Check if we have valid object points
    if object_pointcloud_camera is None or len(object_pointcloud_camera) == 0:
        logger.warning("No object point cloud provided, returning original pose")
        return (
            Quaternion,
            Translation,
            Scale,
            -1.0,
            False,
            False,
        )

    # Use object point cloud directly as target points (already in camera space)
    target_points = object_pointcloud_camera

    # Filter outliers using quantile
    thresh = 0.9
    if len(target_points) > 10:
        depth_quantile = torch.quantile(target_points[:, 2], thresh)
        target_points = target_points[target_points[:, 2] <= depth_quantile]

    if len(target_points) < 10:
        logger.warning(f"Too few target points ({len(target_points)}), returning original pose")
        return (
            Quaternion,
            Translation,
            Scale,
            -1.0,
            False,
            False,
        )

    # Get source points from mesh
    source_points = mesh.verts_packed()

    # Step 1: Manual alignment based on height and center
    height_src = torch.max(source_points[:, 1]) - torch.min(source_points[:, 1])
    height_tgt = torch.max(target_points[:, 1]) - torch.min(target_points[:, 1])
    scale_1 = height_tgt / height_src
    source_points *= scale_1
    center *= scale_1

    center_src = torch.mean(source_points, dim=0)
    center_tgt = torch.mean(target_points, dim=0)
    translation_1 = center_tgt - center_src

    source_points += translation_1
    center += translation_1

    # Apply manual alignment
    tfm1 = (
        Transform3d(device=device)
        .scale(scale_1.expand(3)[None])
        .translate(translation_1[None])
    )
    mesh = Meshes(verts=[source_points], faces=[faces_idx], textures=textures)
    rendered = renderer(mesh)
    ori_iou = compute_iou(rendered[..., 3][0][None, None], mask, threshold=0.5)
    final_iou = ori_iou.cpu().item()

    logger.info(f"Manual alignment IOU: {final_iou:.4f}")

    # Step 2: Shape ICP
    Flag_ICP = False
    if Enable_shape_ICP and len(target_points) >= 10:
        try:
            points_aligned_icp, transformation = run_ICP(
                mesh, source_points, target_points, threshold=0.05
            )
            mesh_ICP = Meshes(
                verts=[points_aligned_icp], faces=[faces_idx], textures=textures
            )
            rendered = renderer(mesh_ICP)
            ori_iou_shapeICP = compute_iou(
                rendered[..., 3][0][None, None], mask, threshold=0.5
            )
            # determine whether accept ICP
            if ori_iou_shapeICP > ori_iou:
                Flag_ICP = True
                mesh = mesh_ICP
                final_iou = ori_iou_shapeICP.cpu().item()
                T_o3d = torch.tensor(transformation, dtype=torch.float32, device=device)
                T_o3d = T_o3d.T
                A = T_o3d[:3, :3]
                t = T_o3d[3, :3]
                scale = A.norm(dim=1)
                R = A / scale[:, None]
                center = ((center[None] * scale) @ R + t)[0]  # transform center
                tfm2 = (
                    Transform3d(device=device)
                    .scale(scale[None])
                    .rotate(R[None])
                    .translate(t[None])
                )
                logger.info(f"ICP alignment IOU: {final_iou:.4f} (accepted)")
            else:
                scale_2 = torch.tensor(1.0, device=device)
                translation_2 = torch.zeros(3, device=device)
                tfm2 = (
                    Transform3d(device=device)
                    .scale(scale_2.expand(3)[None])
                    .translate(translation_2[None])
                )
                logger.info(f"ICP alignment IOU: {ori_iou_shapeICP.cpu().item():.4f} (rejected)")
        except Exception as e:
            logger.warning(f"ICP failed: {e}")
            scale_2 = torch.tensor(1.0, device=device)
            translation_2 = torch.zeros(3, device=device)
            tfm2 = (
                Transform3d(device=device)
                .scale(scale_2.expand(3)[None])
                .translate(translation_2[None])
            )
    else:
        scale_2 = torch.tensor(1.0, device=device)
        translation_2 = torch.zeros(3, device=device)
        tfm2 = (
            Transform3d(device=device)
            .scale(scale_2.expand(3)[None])
            .translate(translation_2[None])
        )

    # Step 3: Render-and-Compare optimization (optional)
    Flag_optim = False
    if Enable_rendering_optimization:
        try:
            quat, translation, scale, R = run_render_compare(
                mesh, center, renderer, mask, device
            )
            with torch.no_grad():
                transformed = apply_transform(mesh, center, quat, translation, scale)
                rendered = renderer(transformed)
            optimized_iou = compute_iou(
                rendered[..., 3][0][None, None], mask, threshold=0.5
            )
            # Accept if better IOU
            if optimized_iou > ori_iou:
                Flag_optim = True
                final_iou = optimized_iou.cpu().item()
                tfm3 = (
                    Transform3d(device=device)
                    .scale(scale[None])
                    .rotate(R[None])
                    .translate(translation[None])
                )
                logger.info(f"Render optimization IOU: {final_iou:.4f} (accepted)")
            else:
                tfm3 = Transform3d(device=device)
                logger.info(f"Render optimization IOU: {optimized_iou.cpu().item():.4f} (rejected)")
        except Exception as e:
            logger.warning(f"Render optimization failed: {e}")
            tfm3 = Transform3d(device=device)
    else:
        tfm3 = Transform3d(device=device)

    # Compose all transformations
    tfm = tfm_ori.compose(tfm1).compose(tfm2).compose(tfm3)

    # Extract final pose
    M = tfm.get_matrix()[0]
    A = M[:3, :3]
    t = M[3, :3]
    scale_final = A.norm(dim=1)
    R_final = A / scale_final[:, None]
    quat_final = matrix_to_quaternion(R_final[None])

    return (
        quat_final,
        t[None],
        scale_final,
        final_iou,
        Flag_ICP,
        Flag_optim,
    )


def load_and_simplify_mesh(Mesh, device, target_triangles=5000):

    vertices = np.asarray(Mesh.vertices)
    faces = np.asarray(Mesh.faces)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_non_manifold_edges()

    if len(mesh_o3d.triangles) > target_triangles:
        mesh_simplified = mesh_o3d.simplify_quadric_decimation(target_triangles)
    else:
        mesh_simplified = mesh_o3d

    verts = torch.tensor(
        np.asarray(mesh_simplified.vertices), dtype=torch.float32, device=device
    )
    faces = torch.tensor(
        np.asarray(mesh_simplified.triangles), dtype=torch.int64, device=device
    )

    return verts, faces


def compute_iou(render_mask_obj, mask_obj_gt, threshold=0.5):

    # Binarize masks
    pred = (render_mask_obj > threshold).float()
    gt_obj = (mask_obj_gt > threshold).float()

    # mask = pred[0, 0].cpu().numpy() * 255
    # mask_uint8 = mask.astype(np.uint8)
    # cv2.imwrite(path, mask_uint8)

    # Compute intersection and union
    intersection = (pred * gt_obj).sum()
    union = ((pred + gt_obj) > 0).float().sum()

    if union == 0:
        return torch.tensor(1.0 if intersection == 0 else 0.0)  # avoid division by zero

    iou = intersection / union
    return iou


def denormalize_f(norm_K, height, width):
    # Extract cx and cy from the normalized K matrix
    cx_norm = norm_K[0][2]  # c_x is at K[0][2]
    cy_norm = norm_K[1][2]  # c_y is at K[1][2]

    fx_norm = norm_K[0][0]  # Normalized fx
    fy_norm = norm_K[1][1]  # Normalized fy
    s_norm = norm_K[0][1]  # Skew (usually 0)

    # Scale to absolute values
    fx_abs = fx_norm * width
    fy_abs = fy_norm * height
    cx_abs = cx_norm * width
    cy_abs = cy_norm * height
    s_abs = s_norm * width

    # Construct absolute K matrix
    abs_K = np.array([[fx_abs, s_abs, cx_abs], [0.0, fy_abs, cy_abs], [0.0, 0.0, 1.0]])
    return abs_K


# Convert torch tensors to Open3D point clouds
def tensor_to_o3d_pcd(tensor):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tensor.cpu().numpy())
    return pcd


# Convert Open3D back to torch tensor
def o3d_to_tensor(pcd):
    return torch.tensor(np.asarray(pcd.points), dtype=torch.float32)


def run_ICP(source_points_mesh, source_points, target_points, threshold):
    # Convert your point clouds
    mesh_src_pcd = tensor_to_o3d_pcd(source_points_mesh.verts_padded().squeeze(0))
    src_pcd = tensor_to_o3d_pcd(source_points)
    tgt_pcd = tensor_to_o3d_pcd(target_points)

    # Run ICP
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    # Apply transformation
    mesh_src_pcd.transform(reg_p2p.transformation)
    points_aligned_icp = o3d_to_tensor(mesh_src_pcd).to(source_points.device)

    return points_aligned_icp, reg_p2p.transformation


def run_render_compare(mesh, center, renderer, mask, device):

    quat = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, requires_grad=True)
    )
    translation = torch.nn.Parameter(
        torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
    )
    scale = torch.nn.Parameter(torch.tensor(1.0, device=device, requires_grad=True))

    def get_optimizer(stage):
        if stage == 1:
            return torch.optim.Adam([translation, scale], lr=1e-2)
        elif stage == 2:
            return torch.optim.Adam([quat, translation, scale], lr=5e-3)

    loss_weights = {"mask": 200, "reg_q": 0.1, "reg_t": 0.05, "reg_s": 0.05}
    prev_loss = None

    global_step = 0
    for stage in [1, 2]:
        optimizer = get_optimizer(stage)
        iters = [5, 25]
        for i in range(iters[stage - 1]):
            optimizer.zero_grad()
            transformed = apply_transform(mesh, center, quat, translation, scale)
            rendered = renderer(transformed)
            loss = compute_loss(rendered, mask, loss_weights, quat, translation, scale)
            loss.backward()
            optimizer.step()
            global_step += 1
            if prev_loss is not None and abs(loss.item() - prev_loss) < 1e-5:
                break
            prev_loss = loss.item()

    quat, translation, scale = quat.detach(), translation.detach(), scale.detach()
    quat_normalized = quat / quat.norm()
    R = quaternion_to_matrix(quat_normalized)

    return quat, translation, scale, R
