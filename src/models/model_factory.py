"""
Model Factory

Central hub for instantiating segmentation model architectures.
Supported models: baseline_unet, att_mamba2_unet.
"""


def get_model(model_config: dict, data_config: dict):
    """
    Build a segmentation model from configuration.

    Args:
        model_config: Model configuration with 'name' key.
        data_config: Data configuration (for patch_size, etc.).

    Returns:
        torch.nn.Module: Segmentation model.
    """
    config_args = model_config.copy()
    model_name = config_args.pop("name", None)

    if model_name is None:
        raise ValueError("Model configuration must have a 'name' key.")

    if model_name == "baseline_unet":
        from src.models.segmentation.baseline_unet import get_baseline_unet
        return get_baseline_unet(**config_args)

    elif model_name == "att_mamba2_unet":
        from src.models.segmentation.att_mamba2_net import get_att_mamba2_unet
        return get_att_mamba2_unet(
            patch_size=data_config["patch_size"],
            **config_args
        )

    else:
        raise ValueError(f"Unknown model name: '{model_name}'")
