# MV-SAM3D Parameter Documentation

This document provides detailed documentation for all command-line parameters of `run_inference_weighted.py`.

## Table of Contents

- [Basic Parameters](#basic-parameters)
- [Inference Parameters](#inference-parameters)
- [Weighting Parameters](#weighting-parameters)
- [Visualization Parameters](#visualization-parameters)
- [DA3 Integration Parameters](#da3-integration-parameters)
- [Usage Examples](#usage-examples)

---

## Basic Parameters

### `--input_path` (Required)
- **Type**: `str`
- **Description**: Input data path (directory or file)
- **Example**: `--input_path ./data/example`

### `--mask_prompt`
- **Type**: `str`
- **Default**: `None`
- **Description**: Mask folder name. If the input directory has multiple mask subfolders, specify which one to use.
- **Example**: `--mask_prompt stuffed_toy`

### `--image_names`
- **Type**: `str`
- **Default**: `None`
- **Description**: Specify image names to use (comma-separated). If not specified, all images in the directory will be used.
- **Example**: `--image_names 0,1,2,3,4,5,6,7`
- **Note**: Image numbering starts from 0

### `--model_tag`
- **Type**: `str`
- **Default**: `"hf"`
- **Description**: Model tag, specifies which checkpoint directory to use
- **Example**: `--model_tag hf`

---

## Inference Parameters

### `--seed`
- **Type**: `int`
- **Default**: `42`
- **Description**: Random seed for reproducibility
- **Example**: `--seed 42`

### `--stage1_steps`
- **Type**: `int`
- **Default**: `50`
- **Description**: Number of iteration steps for Stage 1 (sparse structure sampling). More steps typically yield better quality but take longer.
- **Example**: `--stage1_steps 50`

### `--stage2_steps`
- **Type**: `int`
- **Default**: `25`
- **Description**: Number of iteration steps for Stage 2 (structured latent space sampling).
- **Example**: `--stage2_steps 25`

### `--decode_formats`
- **Type**: `str`
- **Default**: `"gaussian,mesh"`
- **Description**: Decode formats list (comma-separated). Supported formats: `gaussian`, `mesh`
- **Example**: `--decode_formats gaussian,mesh` or `--decode_formats mesh`

---

## Weighting Parameters

### `--no_weighting`
- **Type**: `flag` (boolean)
- **Default**: `False`
- **Description**: Disable weighted fusion, use simple averaging. Only for comparison experiments.
- **Example**: `--no_weighting`

### `--entropy_alpha`
- **Type**: `float`
- **Default**: `30.0`
- **Description**: Gibbs temperature parameter, controls the sharpness (contrast) of weighted fusion.
  - **Higher values** (`> 30.0`): More selective, only trust high-confidence views, may result in sparse output
  - **Lower values** (`< 30.0`): Smoother, more uniform contributions from all views
  - **Recommended range**: `10.0` - `50.0`
- **Example**: `--entropy_alpha 30.0`

### `--attention_layer`
- **Type**: `int`
- **Default**: `6`
- **Description**: Attention layer number for computing weights. Different layers capture different feature levels.
- **Recommended**: `6` (default layer, balances performance and detail)

### `--attention_step`
- **Type**: `int`
- **Default**: `0`
- **Description**: Diffusion step for computing weights. Usually use the first step (`0`) where features are most pronounced.
- **Example**: `--attention_step 0`

### `--min_weight`
- **Type**: `float`
- **Default**: `0.001`
- **Description**: Minimum weight threshold to prevent any view's weight from becoming exactly zero.
- **Example**: `--min_weight 0.001`

---

## Weight Source Parameters

### `--weight_source`
- **Type**: `str`
- **Default**: `"entropy"`
- **Options**: `"entropy"`, `"visibility"`, `"mixed"`
- **Description**: Source selection for multi-view fusion weights.
  
  #### 1. `entropy` (Default)
  - **Description**: Use attention entropy only as weight basis
  - **Principle**: Low entropy (focused attention) → High weight
  - **Use case**: General scenarios without DA3 depth information
  
  #### 2. `visibility`
  - **Description**: Use self-occlusion based visibility as weight basis
  - **Principle**: Visible (not self-occluded) → High weight (binary 0/1)
  - **Method**: Uses DDA ray tracing to detect which latent points are occluded from each view
  - **Requirement**: **Must provide `--da3_output`**
  - **Use case**: Scenarios where occlusion relationships matter
  
  #### 3. `mixed`
  - **Description**: Combine entropy and visibility
  - **Requirement**: **Must provide `--da3_output`**
  - **Use case**: Scenarios requiring both attention quality and geometric visibility

- **Example**: `--weight_source mixed`

### `--visibility_alpha`
- **Type**: `float`
- **Default**: `30.0`
- **Description**: Gibbs temperature parameter for visibility weighting (similar to `--entropy_alpha`).
  - Higher values: More selective, larger difference between visible and occluded weights
  - Lower values: Smoother, smaller weight differences
- **Only applies when**: `--weight_source` is `"visibility"` or `"mixed"`
- **Example**: `--visibility_alpha 30.0`

### `--self_occlusion_tolerance`
- **Type**: `float`
- **Default**: `4.0`
- **Description**: Tolerance for self-occlusion detection (in voxel units). Ignores occluding voxels within this distance of the target to avoid false positives from grazing angles.
  - Higher values: More lenient, fewer false occlusions
  - Lower values: More strict, may have issues with flat surfaces at grazing angles
  - **Recommended range**: `2.0` - `6.0`
- **Only applies when**: `--weight_source` is `"visibility"` or `"mixed"`
- **Example**: `--self_occlusion_tolerance 4.0`

### `--weight_combine_mode`
- **Type**: `str`
- **Default**: `"average"`
- **Options**: `"average"`, `"multiply"`
- **Description**: How to combine entropy and visibility weights in mixed mode.
  
  #### 1. `average` (Default)
  - **Formula**: `w = (1-r) * w_entropy + r * w_visibility`
  - **Parameter**: `r` is controlled by `--visibility_weight_ratio`
  
  #### 2. `multiply`
  - **Formula**: `w = w_entropy * w_visibility`, then normalize
  - **Effect**: Only views with both high entropy confidence and high visibility get high weights

- **Only applies when**: `--weight_source` is `"mixed"`
- **Example**: `--weight_combine_mode average`

### `--visibility_weight_ratio`
- **Type**: `float`
- **Default**: `0.5`
- **Range**: `0.0` - `1.0`
- **Description**: Visibility weight ratio in `average` mode.
  - `0.0`: Use only entropy weights
  - `0.5`: Equal mix of entropy and visibility
  - `1.0`: Use only visibility weights
- **Only applies when**: `--weight_source` is `"mixed"` and `--weight_combine_mode` is `"average"`
- **Example**: `--visibility_weight_ratio 0.5`

---

## Visualization Parameters

### `--visualize_weights`
- **Type**: `flag`
- **Default**: `False`
- **Description**: Save weight and entropy visualization results. Creates a `weights/` folder in the output directory containing:
  - Weight distribution for each view
  - Entropy distribution
  - 3D weight visualization (if possible)
- **Example**: `--visualize_weights`

### `--compute_latent_visibility`
- **Type**: `flag`
- **Default**: `False`
- **Description**: Visualize latent visibility per view. Creates GLB files showing which latent points are visible from each camera view.
  - **Output**: `latent_visibility.glb` (overall) and `latent_visibility_per_view/` folder
  - Green points: visible, Red points: occluded
- **Requirement**: **Must provide `--da3_output`**
- **Example**: `--compute_latent_visibility`

### `--save_attention`
- **Type**: `flag`
- **Default**: `False`
- **Description**: Save all attention weights (for analysis). Generates large files.
- **Example**: `--save_attention`

### `--attention_layers`
- **Type**: `str`
- **Default**: `None`
- **Description**: Specify which layers' attention to save (comma-separated). Only effective when `--save_attention` is enabled.
- **Example**: `--attention_layers 4,5,6`

### `--save_stage2_init`
- **Type**: `flag`
- **Default**: `False`
- **Description**: Save Stage 2 initial latent state, for iteration stability analysis.
- **Example**: `--save_stage2_init`

---

## DA3 Integration Parameters

### `--da3_output`
- **Type**: `str`
- **Default**: `None`
- **Description**: Path to DA3 output npz file (generated by `scripts/run_da3.py`).
  - If provided, uses DA3 point clouds instead of built-in depth model
  - DA3 point clouds typically have higher quality, especially in complex scenes
  - **Required format**: npz file containing `pointmaps_sam3d` key
  - **Required for**: visibility-based weighting (`--weight_source visibility` or `mixed`)
- **Example**: `--da3_output ./da3_outputs/example/da3_output.npz`

### `--merge_da3_glb`
- **Type**: `flag`
- **Default**: `False`
- **Description**: Merge SAM3D generated GLB with DA3's `scene.glb` for alignment comparison.
  - **Requirement**: Must provide `--da3_output`
  - **Output**: `result_merged_scene.glb`
- **Example**: `--merge_da3_glb --da3_output ./da3_outputs/example/da3_output.npz`

### `--overlay_pointmap`
- **Type**: `flag`
- **Default**: `False`
- **Description**: Overlay SAM3D result on input point cloud for pose visualization.
  - Can be used with default MoGe depth maps or DA3 point clouds
  - **Output**: `result_overlay.glb`
- **Example**: `--overlay_pointmap`

---

## Usage Examples

### Basic Multi-View Inference
```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5,6,7
```

### Custom Weighting Parameters
```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5,6,7 \
    --entropy_alpha 40.0 \
    --attention_layer 6
```

### Using DA3 Point Clouds
```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5,6,7 \
    --da3_output ./da3_outputs/example/da3_output.npz \
    --overlay_pointmap \
    --merge_da3_glb
```

### Using Visibility Weighting
```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5,6,7 \
    --da3_output ./da3_outputs/example/da3_output.npz \
    --weight_source visibility \
    --visibility_alpha 30.0 \
    --visibility_threshold 0.1
```

### Using Mixed Weighting (Entropy + Visibility)
```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5,6,7 \
    --da3_output ./da3_outputs/example/da3_output.npz \
    --weight_source mixed \
    --entropy_alpha 30.0 \
    --visibility_alpha 30.0 \
    --weight_combine_mode average \
    --visibility_weight_ratio 0.5 \
    --visibility_threshold 0.1
```

---

## Output Files

### Basic Output
- `result.glb`: Main 3D reconstruction result (Gaussian splat or mesh)
- `result.ply`: Point cloud format output
- `params.npz`: Parameter file (contains pose, scale, etc.)
- `inference.log`: Inference log

### Overlay Visualization Output
- `result_overlay.glb`: SAM3D result overlaid on View 0 point cloud

### Merged Scene Output
- `result_merged_scene.glb`: SAM3D object merged with DA3 scene

### Weighting Visualization Output (if enabled)
- `weights/`: Contains weight distribution, entropy distribution, 3D weight visualization, etc.

---

## Parameter Dependencies

### Required Combinations
1. `--merge_da3_glb` must be used with `--da3_output` 
2. `--weight_source visibility` or `--weight_source mixed` **must be used** with `--da3_output` 
3. `--weight_combine_mode` and `--visibility_weight_ratio` only apply when `--weight_source mixed`

### Recommended Combinations
1. High quality reconstruction: `--da3_output` + `--entropy_alpha 30.0`
2. Visibility weighting: `--weight_source visibility` + `--da3_output`
3. Mixed weighting: `--weight_source mixed` + `--da3_output` + `--visibility_weight_ratio 0.5`

---

## FAQ

### Q: `entropy_alpha` How to tune?
A:
- **Default**: 30.0（recommended starting point）
- **Results too sparse**: lower to  20.0 or 10.0
- **Results too smooth**: increase to  40.0 or 50.0

### Q: Which one should I use `weight_source`？
A:
- **`entropy` (Default)**: Suitable for most scenarios, does not require DA3 depth info
- **`visibility`**: Suitable for scenarios with obvious occlusion, requires DA3 depth info
- **`mixed`**: Suitable for complex scenarios requiring both attention quality and geometric visibility

### Q: `visibility_threshold` How to set？
A:
- **Default**: 0.1（Object size 10%, about 3cm for 30cm object）
- **More strict**: lower to  0.05（Only very close points considered visible）
- **More lenient**: increase to  0.15 or 0.2（Allow larger distance tolerance）

---

## Changelog

- **2025-12-05**: 
  - Added visibility-based weighting using self-occlusion detection
  - Added `--weight_source` parameter: `entropy`, `visibility`, `mixed`
  - Added `--self_occlusion_tolerance` for DDA ray tracing
  - Cleaned up experimental camera pose estimation features

