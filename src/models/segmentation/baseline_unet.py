"""
Defines the baseline 3D U-Net model for benchmarking.

This model uses the standard MONAI implementation of the U-Net.
"""

from monai.networks.nets import UNet

def get_baseline_unet(in_channels=1, out_channels=2, **kwargs):
    """
    Constructs a standard 3D U-Net model.

    Args:
        in_channels: Number of input channels (1 for CT only, 2 if vesselness is appended).
        out_channels: Number of output channels (2 = background + artery).

    Returns:
        monai.networks.nets.UNet: A 3D U-Net instance.
    """
    channels = (32, 64, 128, 256, 512)
    strides = (2, 2, 2, 2)

    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=2,
        norm="instance",
    )

if __name__ == '__main__':
    # A simple test to check if the model builds and to see its structure
    import torch
    
    print("Building 3D U-Net model...")
    model = get_baseline_unet()
    
    # Try passing a sample 3D patch (Batch=1, Channel=1, D=96, H=96, W=96)
    try:
        sample_input = torch.randn(1, 1, 96, 96, 96)
        output = model(sample_input)
        
        print("\n--- Model Test Success ---")
        print(f"Input shape:  {sample_input.shape}")
        print(f"Output shape: {output.shape}")
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {params/1e6:.2f} Million")

    except Exception as e:
        print(f"\nModel test failed: {e}")