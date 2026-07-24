"""
Segmentation Models Package

Contains voxel-based segmentation architectures:
- baseline_unet: Standard MONAI 3D U-Net
- att_mamba2_unet: Hybrid Attention-Mamba2 U-Net (SOTA)

Imports are lazy to avoid dependency issues (e.g., mamba_ssm).
Use direct imports in model_factory.py instead.
"""

__all__ = [
    "get_baseline_unet",
    "get_att_mamba2_unet",
]
