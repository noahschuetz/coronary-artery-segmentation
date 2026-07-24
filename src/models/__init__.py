"""
Models Package

Provides segmentation model architectures: baseline_unet, att_mamba2_unet.

Usage:
    from src.models import get_model
    model = get_model(model_config, data_config)
"""

from .model_factory import get_model

# Lazy imports for specific model classes
def __getattr__(name):
    """Lazy import for model classes."""
    if name in ("UNet", "get_baseline_unet"):
        from .segmentation.baseline_unet import get_baseline_unet
        return get_baseline_unet

    if name in ("AttMamba2UNet", "get_att_mamba2_unet"):
        from .segmentation.att_mamba2_net import get_att_mamba2_unet
        return get_att_mamba2_unet

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "get_model",
    "get_baseline_unet",
    "get_att_mamba2_unet",
]
