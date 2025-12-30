# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Union, Optional, Any
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from loguru import logger
from PIL import Image

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import Transform3d

from sam3d_objects.model.backbone.dit.embedder.pointmap import PointPatchEmbed
from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.data.dataset.tdfy.transforms_3d import (
    DecomposedTransform,
)
from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
from sam3d_objects.pipeline.inference_utils import o3d_plane_estimation, estimate_plane_area


def camera_to_pytorch3d_camera(device="cpu") -> DecomposedTransform:
    """
    R3 camera space --> PyTorch3D camera space
    Also needed for pointmaps
    """
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    return DecomposedTransform(
        rotation=r3_to_p3d_R,
        translation=r3_to_p3d_T,
        scale=torch.tensor(1.0, dtype=r3_to_p3d_R.dtype, device=device),
    )


def recursive_fn_factory(fn):
    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, torch.Tensor):
            return fn(b)
        # Yes, writing out an explicit white list of
        # trivial types is tedious, but so are bugs that
        # come from not applying fn, when expected to have
        # applied it.
        if b is None:
            return b
        trivial_types = [bool, int, float]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None
):
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            f"compiled {fn}" if name is None else name
        ):
            cont_args = recursive_contiguous(args)
            cont_kwargs = recursive_contiguous(kwargs)
            result = compiled_fn(*cont_args, **cont_kwargs)
            cloned_result = recursive_clone(result)
            return cloned_result

    return compiled_fn_wrapper


class InferencePipelinePointMap(InferencePipeline):

    def __init__(
        self, *args, depth_model, layout_post_optimization_method=None, clip_pointmap_beyond_scale=None, **kwargs
    ):
        self.depth_model = depth_model
        self.layout_post_optimization_method = layout_post_optimization_method
        self.clip_pointmap_beyond_scale = clip_pointmap_beyond_scale
        super().__init__(*args, **kwargs)

    def _compile(self):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        compile_mode = "max-autotune"

        for embedder, _ in self.condition_embedders[
            "ss_condition_embedder"
        ].embedder_list:
            if isinstance(embedder, PointPatchEmbed):
                logger.info("Found PointPatchEmbed")
                embedder.inner_forward = compile_wrapper(
                    embedder.inner_forward,
                    mode=compile_mode,
                    fullgraph=True,
                )
            else:
                embedder.forward = compile_wrapper(
                    embedder.forward,
                    mode=compile_mode,
                    fullgraph=True,
                )

        self.models["ss_generator"].reverse_fn.inner_forward = compile_wrapper(
            self.models["ss_generator"].reverse_fn.inner_forward,
            mode=compile_mode,
            fullgraph=True,
        )

        self.models["ss_decoder"].forward = compile_wrapper(
            self.models["ss_decoder"].forward,
            mode=compile_mode,
            fullgraph=True,
        )

        self._warmup()

    def _warmup(self, num_warmup_iters=3):
        test_image = np.ones((512, 512, 4), dtype=np.uint8) * 255
        test_image[:, :, :3] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(test_image)
        mask = None
        image = self.merge_image_and_mask(image, mask)
        with torch.inference_mode(False):
            with torch.no_grad():
                for _ in tqdm(range(num_warmup_iters)):
                    pointmap_dict = recursive_clone(self.compute_pointmap(image))
                    pointmap = pointmap_dict["pointmap"]

                    ss_input_dict = self.preprocess_image(
                        image, self.ss_preprocessor, pointmap=pointmap
                    )
                    ss_return_dict = self.sample_sparse_structure(
                        ss_input_dict, inference_steps=None
                    )

                    _ = self.run_layout_model(
                        ss_input_dict,
                        ss_return_dict,
                        inference_steps=None,
                    )

    def _preprocess_image_and_mask_pointmap(
        self, rgb_image, mask_image, pointmap, img_mask_pointmap_joint_transform
    ):
        for trans in img_mask_pointmap_joint_transform:
            rgb_image, mask_image, pointmap = trans(
                rgb_image, mask_image, pointmap=pointmap
            )
        return rgb_image, mask_image, pointmap

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
        preprocessor,
        pointmap=None,
    ) -> torch.Tensor:
        # canonical type is numpy
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        assert image.ndim == 3  # no batch dimension as of now
        assert image.shape[-1] == 4  # rgba format
        assert image.dtype == np.uint8  # [0,255] range

        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
            rgb_image, rgb_image_mask, pointmap
        )
        
        # Put in a for loop?
        _item = preprocessor_return_dict
        item = {
            "mask": _item["mask"][None].to(self.device),
            "image": _item["image"][None].to(self.device),
            "rgb_image": _item["rgb_image"][None].to(self.device),
            "rgb_image_mask": _item["rgb_image_mask"][None].to(self.device),
        }

        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = _item["pointmap"][None].to(self.device)
            item["rgb_pointmap"] = _item["rgb_pointmap"][None].to(self.device)
            item["pointmap_scale"] = _item["pointmap_scale"][None].to(self.device)
            item["pointmap_shift"] = _item["pointmap_shift"][None].to(self.device)
            item["rgb_pointmap_scale"] = _item["rgb_pointmap_scale"][None].to(self.device)
            item["rgb_pointmap_shift"] = _item["rgb_pointmap_shift"][None].to(self.device)

        return item

    def _clip_pointmap(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.clip_pointmap_beyond_scale is None:
            return pointmap

        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask_resized = torchvision.transforms.functional.resize(
            mask, pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        ).squeeze(0)

        pointmap_flat = pointmap.reshape(3, -1)
        # Get valid points from the mask
        mask_bool = mask_resized.reshape(-1) > 0.5
        mask_points = pointmap_flat[:, mask_bool]
        mask_distance = mask_points.nanmedian(dim=-1).values[-1]
        logger.info(f"mask_distance: {mask_distance}")
        pointmap_clipped_flat = torch.where(
            pointmap_flat[2, ...].abs() > self.clip_pointmap_beyond_scale * mask_distance,
            torch.full_like(pointmap_flat, float('nan')),
            pointmap_flat
        )
        pointmap_clipped = pointmap_clipped_flat.reshape(pointmap.shape)
        return pointmap_clipped



    def compute_pointmap(self, image, pointmap=None):
        loaded_image = self.image_to_float(image)
        loaded_image = torch.from_numpy(loaded_image)
        loaded_mask = loaded_image[..., -1]
        loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]

        if pointmap is None:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    output = self.depth_model(loaded_image)
            pointmaps = output["pointmaps"]
            camera_convention_transform = (
                Transform3d()
                .rotate(camera_to_pytorch3d_camera(device=self.device).rotation)
                .to(self.device)
            )
            points_tensor = camera_convention_transform.transform_points(pointmaps)
            intrinsics = output.get("intrinsics", None)
        else:
            # External pointmap provided (e.g., from DA3)
            # Input format: (3, H, W)
            # 
            # 根据实际日志分析，DA3 和 MoGe 的原始 pointmap 坐标系是一致的：
            # 
            # DA3 原始输出 (run_da3.py 的 depth_to_pointmap):
            #   - X: 右 (正)
            #   - Y: 下 (正，但物体在图像中心偏上时为负)
            #   - Z: 前 (正，远离相机)
            # 
            # MoGe 原始输出:
            #   - X: 右 (正)
            #   - Y: 下 (正，但物体在图像中心偏上时为负)
            #   - Z: 前 (正，远离相机)
            # 
            # 两者都是标准相机坐标系，不需要额外的 Y/Z 翻转！
            # 只需要应用 camera_to_pytorch3d: (x, y, z) -> (-x, -y, z)
            
            output = {}
            points_tensor = pointmap.to(self.device)
            
            # Convert from (3, H, W) to (H, W, 3) for processing
            if points_tensor.dim() == 3 and points_tensor.shape[0] == 3:
                points_tensor = points_tensor.permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
            
            # Resize if needed (before coordinate transform)
            # loaded_image is (3, H', W'), points_tensor is now (H, W, 3)
            if (points_tensor.shape[0], points_tensor.shape[1]) != (loaded_image.shape[1], loaded_image.shape[2]):
                points_tensor = torch.nn.functional.interpolate(
                    points_tensor.permute(2, 0, 1).unsqueeze(0),  # (H, W, 3) -> (1, 3, H, W)
                    size=(loaded_image.shape[1], loaded_image.shape[2]),
                    mode="nearest",
                ).squeeze(0).permute(1, 2, 0)  # back to (H', W', 3)
            
            # 调试输出：DA3 原始 pointmap（标准相机坐标系）
            logger.info(f"[External Pointmap] DA3 raw input (standard camera space):")
            logger.info(f"  X: [{points_tensor[..., 0].min():.4f}, {points_tensor[..., 0].max():.4f}], mean={points_tensor[..., 0].mean():.4f}")
            logger.info(f"  Y: [{points_tensor[..., 1].min():.4f}, {points_tensor[..., 1].max():.4f}], mean={points_tensor[..., 1].mean():.4f}")
            logger.info(f"  Z: [{points_tensor[..., 2].min():.4f}, {points_tensor[..., 2].max():.4f}], mean={points_tensor[..., 2].mean():.4f}")
            
            # 应用与 MoGe 相同的变换: (x, y, z) -> (-x, -y, z)
            # 这是从标准相机坐标系到 PyTorch3D 相机坐标系的变换
            camera_convention_transform = (
                Transform3d()
                .rotate(camera_to_pytorch3d_camera(device=self.device).rotation)
                .to(self.device)
            )
            points_tensor = camera_convention_transform.transform_points(points_tensor)
            
            logger.info(f"[External Pointmap] After camera_to_pytorch3d (-x, -y, z):")
            logger.info(f"  X: [{points_tensor[..., 0].min():.4f}, {points_tensor[..., 0].max():.4f}], mean={points_tensor[..., 0].mean():.4f}")
            logger.info(f"  Y: [{points_tensor[..., 1].min():.4f}, {points_tensor[..., 1].max():.4f}], mean={points_tensor[..., 1].mean():.4f}")
            logger.info(f"  Z: [{points_tensor[..., 2].min():.4f}, {points_tensor[..., 2].max():.4f}], mean={points_tensor[..., 2].mean():.4f}")
            
            intrinsics = None

        points_tensor = points_tensor.permute(2, 0, 1)
        points_tensor = self._clip_pointmap(points_tensor, loaded_mask)
        
        # Prepare the point map tensor
        point_map_tensor = {
            "pointmap": points_tensor,
            "pts_color": loaded_image,
        }

        # If depth model doesn't provide intrinsics, infer them
        if intrinsics is None:
            intrinsics_result = infer_intrinsics_from_pointmap(
                points_tensor.permute(1, 2, 0), device=self.device
            )
            point_map_tensor["intrinsics"] = intrinsics_result["intrinsics"]

        # Check if object point cloud is available (passed via temporary attribute)
        if hasattr(self, '_temp_object_pointcloud_camera'):
            point_map_tensor["object_pointcloud_camera"] = self._temp_object_pointcloud_camera
            logger.info(f"Embedded object point cloud in pointmap_dict: {self._temp_object_pointcloud_camera.shape}")

        return point_map_tensor

    def run_post_optimization(self, mesh_glb, intrinsics, pose_dict, layout_input_dict, object_pointcloud_camera=None):
        """
        Run post-optimization alignment.

        Args:
            mesh_glb: Mesh to align
            intrinsics: Camera intrinsics
            pose_dict: Initial pose dictionary
            layout_input_dict: Input data dictionary (may contain rgb_pointmap)
            object_pointcloud_camera: (Optional) Pre-extracted object point cloud in camera space.
                                     If provided, uses point cloud-based alignment instead of pointmap.

        Returns:
            Dictionary with refined pose and IOU
        """
        intrinsics = intrinsics.clone()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        re_focal = min(fx, fy)
        intrinsics[0, 0], intrinsics[1, 1] = re_focal, re_focal

        # Use point cloud-based alignment if object point cloud is provided
        if object_pointcloud_camera is not None:
            from sam3d_objects.pipeline.layout_post_optimization_utils import layout_post_optimization_with_pointcloud

            logger.info("Using point cloud-based alignment (bypassing pointmap)")
            revised_quat, revised_t, revised_scale, final_iou, _, _ = (
                layout_post_optimization_with_pointcloud(
                    mesh_glb,
                    pose_dict["rotation"],
                    pose_dict["translation"],
                    pose_dict["scale"],
                    layout_input_dict["rgb_image_mask"][0, 0],
                    object_pointcloud_camera,  # Use pre-extracted point cloud
                    intrinsics,
                    min_size=518,
                )
            )
        else:
            # Legacy: use pointmap-based alignment
            logger.info("Using pointmap-based alignment (legacy)")
            revised_quat, revised_t, revised_scale, final_iou, _, _ = (
                self.layout_post_optimization_method(
                    mesh_glb,
                    pose_dict["rotation"],
                    pose_dict["translation"],
                    pose_dict["scale"],
                    layout_input_dict["rgb_image_mask"][0, 0],
                    layout_input_dict["rgb_pointmap"][0].permute(1, 2, 0),
                    intrinsics,
                    min_size=518,
                )
            )

        return {
            "rotation": revised_quat,
            "translation": revised_t,
            "scale": revised_scale,
            "iou": final_iou,
        }


    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed: Optional[int] = None,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=True,
        use_vertex_color=False,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        pointmap=None,
        decode_formats=None,
        estimate_plane=False,
        attention_logger: Optional[Any] = None,
    ) -> dict:
        image = self.merge_image_and_mask(image, mask)
        with self.device: 
            pointmap_dict = self.compute_pointmap(image, pointmap)
            pointmap = pointmap_dict["pointmap"]
            pts = type(self)._down_sample_img(pointmap)
            pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])

            if estimate_plane:
                return self.estimate_plane(pointmap_dict, image)

            ss_input_dict = self.preprocess_image(
                image, self.ss_preprocessor, pointmap=pointmap
            )

            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            if seed is not None:
                torch.manual_seed(seed)
            if attention_logger is not None:
                attention_logger.start_stage("ss")
                attention_logger.set_num_views(1)
                attention_logger.set_view(0)
            ss_return_dict = self.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
                attention_logger=attention_logger,
            )

            # We could probably use the decoder from the models themselves
            pointmap_scale = ss_input_dict.get("pointmap_scale", None)
            pointmap_shift = ss_input_dict.get("pointmap_shift", None)
            ss_return_dict.update(
                self.pose_decoder(
                    ss_return_dict,
                    scene_scale=pointmap_scale,
                    scene_shift=pointmap_shift,
                )
            )

            logger.info(f"Rescaling scale by {ss_return_dict['downsample_factor']} after downsampling")
            ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]

            if stage1_only:
                logger.info("Finished!")
                ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                return {
                    **ss_return_dict,
                    "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                    "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),  # HxWx3
                }
                # return ss_return_dict

            coords = ss_return_dict["coords"]
            if attention_logger is not None:
                attention_logger.start_stage("slat")
                attention_logger.set_num_views(1)
                attention_logger.set_view(0)
            slat = self.sample_slat(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
                attention_logger=attention_logger,
            )
            outputs = self.decode_slat(
                slat, self.decode_formats if decode_formats is None else decode_formats
            )
            outputs = self.postprocess_slat_output(
                outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
            )
            glb = outputs.get("glb", None)

            try:
                if (
                    with_layout_postprocess
                    and self.layout_post_optimization_method is not None
                ):
                    assert glb is not None, "require mesh to run postprocessing"
                    logger.info("Running layout post optimization method...")

                    # Check if object point cloud is available (passed via pointmap_dict)
                    object_pointcloud_camera = pointmap_dict.get("object_pointcloud_camera", None)

                    postprocessed_pose = self.run_post_optimization(
                        deepcopy(glb),
                        pointmap_dict["intrinsics"],
                        ss_return_dict,
                        ss_input_dict,
                        object_pointcloud_camera=object_pointcloud_camera,
                    )
                    ss_return_dict.update(postprocessed_pose)
            except Exception as e:
                logger.error(
                    f"Error during layout post optimization: {e}", exc_info=True
                )

            # glb.export("sample.glb")
            logger.info("Finished!")

            return {
                **ss_return_dict,
                **outputs,
                "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),  # HxWx3
            }

    @staticmethod
    def _down_sample_img(img_3chw: torch.Tensor):
        # img_3chw: (3, H, W)
        x = img_3chw.unsqueeze(0)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        max_side = max(x.shape[2], x.shape[3])
        scale_factor = 1.0

        # heuristics
        if max_side > 3800:
            scale_factor = 0.125
        if max_side > 1900:
            scale_factor = 0.25
        elif max_side > 1200:
            scale_factor = 0.5

        x = torch.nn.functional.interpolate(
            x,
            scale_factor=(scale_factor, scale_factor),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )  # -> (1, 3, H/4, W/4)
        return x.squeeze(0)

    def estimate_plane(self, pointmap_dict, image, ground_area_threshold=0.25, min_points=100):
        assert image.shape[-1] == 4  # rgba format
        # Extract mask from alpha channel
        floor_mask = type(self)._down_sample_img(torch.from_numpy(image[..., -1]).float().unsqueeze(0))[0] > 0.5
        pts = type(self)._down_sample_img(pointmap_dict["pointmap"])

        # Get all points in 3D space (H, W, 3)
        pts_hwc = pts.cpu().permute((1, 2, 0))

        valid_mask_points = floor_mask.cpu().numpy()
        # Extract points that fall within the mask
        if valid_mask_points.any():
            # Get points within mask
            masked_points = pts_hwc[valid_mask_points]
            # Filter out invalid points (zero points from depth estimation failures)
            valid_points_mask = torch.norm(masked_points, dim=-1) > 1e-6
            valid_points = masked_points[valid_points_mask]
            points = valid_points.numpy()
        else:
            points = np.array([]).reshape(0, 3)
     
        # Calculate area coverage and check num of points
        overlap_area = estimate_plane_area(floor_mask)
        has_enough_points = len(points) >= min_points

        logger.info(f"Plane estimation: {len(points)} points, {overlap_area:.3f} area coverage")
        if overlap_area > ground_area_threshold and has_enough_points:
            try:
                mesh = o3d_plane_estimation(points)
                logger.info("Successfully estimated plane mesh")
            except Exception as e:
                logger.error(f"Failed to estimate plane: {e}")
                mesh = None
        else:
            logger.info(f"Skipping plane estimation: area={overlap_area:.3f}, points={len(points)}")
            mesh = None

        return {
            "glb": mesh,
            "translation": torch.tensor([[0.0, 0.0, 0.0]]),
            "scale": torch.tensor([[1.0, 1.0, 1.0]]),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        }
