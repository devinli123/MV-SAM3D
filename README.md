# MV-SAM3D: Adaptive Multi-View 3D Reconstruction

An enhanced multi-view extension for [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), featuring **adaptive fusion** strategies for improved 3D reconstruction quality from multiple viewpoints.

> 🔗 **Basic Version**: For a simpler averaging-based approach, check out our [basic multi-view fork](https://github.com/devinli123/multi-view-sam-3d-objects).

## 🔥 Highlights

- **Adaptive Multi-View Fusion**: Unlike simple averaging, we employ a confidence-aware fusion mechanism that automatically weighs contributions from different views based on their reliability.
  
- **Per-Latent Weighting**: Our method operates at the latent level, enabling fine-grained control over how information from different views is combined for each spatial location.

- **Improved Reconstruction Quality**: Better handling of occluded regions and view-dependent artifacts through intelligent fusion.

- **Easy to Use**: Drop-in replacement with minimal changes to the original pipeline.

## 📢 Research in Progress

This is an **active research project**. We are continuously exploring new fusion strategies and will release updates as our research progresses.

## Results Comparison

<table>
<tr>
  <td align="center" width="33%"><b>Single-View (View 3)</b></td>
  <td align="center" width="33%"><b>Single-View (View 6)</b></td>
  <td align="center" width="33%"><b>Multi-View (Adaptive Fusion)</b></td>
</tr>
<tr>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Image</b><br>
    <img src="data/example/images/3.png" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Image</b><br>
    <img src="data/example/images/6.png" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Images</b><br>
    <table width="100%" cellpadding="2" cellspacing="2">
    <tr>
      <td align="center"><img src="data/example/images/1.png" width="80px"/></td>
      <td align="center"><img src="data/example/images/2.png" width="80px"/></td>
      <td align="center"><img src="data/example/images/3.png" width="80px"/></td>
      <td align="center"><img src="data/example/images/4.png" width="80px"/></td>
    </tr>
    <tr>
      <td align="center"><img src="data/example/images/5.png" width="80px"/></td>
      <td align="center"><img src="data/example/images/6.png" width="80px"/></td>
      <td align="center"><img src="data/example/images/7.png" width="80px"/></td>
      <td align="center"><img src="data/example/images/8.png" width="80px"/></td>
    </tr>
    </table>
  </td>
</tr>
<tr>
  <td align="center" colspan="3">
    <b>↓ 3D Reconstruction ↓</b>
  </td>
</tr>
<tr>
  <td align="center" width="33%" style="padding: 5px;">
    <b>3D Result</b><br>
    <img src="data/example/visualization_results/view3_cropped.gif" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>3D Result</b><br>
    <img src="data/example/visualization_results/view6_cropped.gif" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>3D Result</b><br>
    <img src="data/example/visualization_results/all_views_cropped.gif" width="100%" style="max-width: 300px;"/>
  </td>
</tr>
</table>

## Installation

Please follow the installation instructions in the [basic multi-view version](https://github.com/devinli123/multi-view-sam-3d-objects) or in [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects).

## Quick Start

```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_path` | Path to input directory | Required |
| `--mask_prompt` | Mask folder name | None |
| `--image_names` | Image names (comma-separated) | All images |
| `--entropy_alpha` | Fusion sharpness (higher = more selective) | 30.0 |
| `--no_weighting` | Disable adaptive fusion (use simple average) | False |
| `--visualize_weights` | Visualize fusion weights | False |

### Data Structure

```
input_path/
├── images/
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── object_name/  # mask folder
    ├── 0.png
    ├── 1.png
    └── ...
```

**Mask Format**: RGBA format where alpha channel stores mask (alpha=255 for object, alpha=0 for background).

## Acknowledgments

This project builds upon [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) by Meta. We thank the original authors for their excellent work.

## License

This project inherits the [SAM License](./LICENSE) from the original SAM 3D Objects project.
