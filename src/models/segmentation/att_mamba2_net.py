"""
Attention-Mamba2 U-Net (Att-Mamba2-UNet).

Proposed hybrid segmentation architecture combining three complementary mechanisms:
  - MLC (Multi-Layer Convolution) blocks for efficient local feature extraction
  - Windowed / global multi-head self-attention for spatial context
  - Mamba-2 SSM for linear-complexity sequential modeling

A stride-2 stem immediately downsamples the input (96³ → 48³), reducing the
computational volume 8× before attention-heavy stages. Encoder stages then
apply progressively broader context: MLC at 48³, windowed attention + Mamba-2
at 24³, global attention + Mamba-2 at 12³ and 6³ (bottleneck). The decoder
uses MONAI UnetUpBlock modules with skip connections, and a final transposed
convolution recovers the resolution lost in the stem.

Reference: inspired by AttmNet (Zhang et al., 2022).
"""

import warnings

import torch
import torch.nn as nn
from monai.networks.blocks import UnetUpBlock, UnetOutBlock

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None

def window_partition(x, window_size):
    """
    Partition (B, C, D, H, W) into non-overlapping windows.
    Returns: (num_windows * B, window_size^3, C)
    """
    B, C, D, H, W = x.shape
    # Reshape to (B, C, D//ws, ws, H//ws, ws, W//ws, ws)
    x = x.view(B, C, D // window_size, window_size, H // window_size, window_size, W // window_size, window_size)
    # Permute to (B, D//ws, H//ws, W//ws, ws, ws, ws, C)
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    # Merge into batch dimension: (B * num_windows, ws*ws*ws, C)
    windows = windows.view(-1, window_size * window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, D, H, W):
    """
    Reverse window partition.
    windows: (num_windows * B, window_size^3, C)
    Returns: (B, C, D, H, W)
    """
    # We don't know B directly, but we can infer it
    # windows.shape[0] is B * (D//ws * H//ws * W//ws)
    C = windows.shape[-1]
    
    x = windows.view(-1, D // window_size, H // window_size, W // window_size, window_size, window_size, window_size, C)
    # Permute back to (B, C, D//ws, ws, H//ws, ws, W//ws, ws)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    x = x.view(-1, C, D, H, W)
    return x

# --- MLC Block Components (from AttmNet) ---

class ConvBlock1(nn.Module):
    """Conv3x3 -> Norm -> ReLU"""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class ConvBlock2(nn.Module):
    """Conv1x1 -> Norm -> ReLU"""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class MLCBlock(nn.Module):
    """
    Multi-Layer Convolution Block.
    Efficient purely convolutional block for high-resolution early stages.
    Structure based on AttmNet Diagram B.
    """
    def __init__(self, channels):
        super().__init__()
        # Left branch: ConvBlock1 -> ConvBlock1
        self.branch1 = nn.Sequential(ConvBlock1(channels), ConvBlock1(channels))
        
        # Right branch: ConvBlock2 -> ConvBlock2
        self.branch2 = nn.Sequential(ConvBlock2(channels), ConvBlock2(channels))
        
        # Middle processing: Sum -> Conv3x3 -> ReLU -> ConvBlock2
        self.middle = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBlock2(channels)
        )

    def forward(self, x):
        # Split branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        # Sum branches and process
        mid = self.middle(b1 + b2)
        
        # Residual connection
        return x + mid

class AttMamba2Block(nn.Module):
    """
    Hybrid Block:
    1. Attention (Windowed or Global) [Optional]
    2. Mamba 2 (Global Sequence Modeling)
    3. Depthwise-separable Conv (Local Spatial + Channel Mixing)

    An outer residual skip connects block input to output.
    """
    def __init__(self, channels, num_heads=None, d_state=64, window_size=0, use_attention=True):
        super().__init__()
        if Mamba2 is None:
            raise ImportError("Mamba2 not found. Please install mamba-ssm >= 2.0")

        self.window_size = window_size
        self.use_attention = use_attention

        # Scale num_heads to maintain ~32-dim per head
        if num_heads is None:
            num_heads = max(2, channels // 32)

        # 1. Attention
        if self.use_attention:
            self.norm1 = nn.LayerNorm(channels)
            self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

        # 2. Mamba 2
        self.norm2 = nn.LayerNorm(channels)
        self.mamba = Mamba2(d_model=channels, d_state=d_state, d_conv=4, expand=2)

        # 3. Depthwise-separable convolution (spatial + channel mixing)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.GELU()
        )

    def forward(self, x):
        # Input: (B, C, D, H, W)

        # Fix for MONAI MetaTensor slicing issue:
        if hasattr(x, "as_tensor"):
            x = x.as_tensor()

        residual = x
        B, C, D, H, W = x.shape

        # --- 1. Attention ---
        if self.use_attention:
            # If window_size > 0, we partition the volume into small cubes.
            if self.window_size > 0:
                # Check if dimensions are divisible by window_size
                if D % self.window_size != 0 or H % self.window_size != 0 or W % self.window_size != 0:
                    warnings.warn(
                        f"Spatial dims ({D},{H},{W}) not divisible by window_size={self.window_size}; "
                        f"skipping windowed attention for this input."
                    )
                    x_attn_out = x
                else:
                    x_windows = window_partition(x, self.window_size) # (B*nW, ws^3, C)
                    x_norm = self.norm1(x_windows)

                    # Optimization: Process attention in chunks to avoid OOM
                    batch_size_attn = x_norm.shape[0]
                    chunk_size = 64

                    if batch_size_attn > chunk_size:
                        attn_out_list = []
                        for i in range(0, batch_size_attn, chunk_size):
                            chunk = x_norm[i:i+chunk_size]
                            out_chunk, _ = self.attn(chunk, chunk, chunk)
                            attn_out_list.append(out_chunk)
                        attn_out = torch.cat(attn_out_list, dim=0)
                    else:
                        attn_out, _ = self.attn(x_norm, x_norm, x_norm)

                    x_attn_out = window_reverse(x_windows + attn_out, self.window_size, D, H, W)
            else:
                # Global Attention (only safe for small resolutions)
                x_flat = x.flatten(2).transpose(1, 2) # (B, L, C)
                x_norm = self.norm1(x_flat)
                attn_out, _ = self.attn(x_norm, x_norm, x_norm)
                x_attn_out = (x_flat + attn_out).transpose(1, 2).view(B, C, D, H, W)
        else:
            x_attn_out = x

        # --- 2. Mamba (Global Context) ---
        x_flat = x_attn_out.flatten(2).transpose(1, 2)
        x_norm = self.norm2(x_flat)
        mamba_out = self.mamba(x_norm)
        x_mamba_out = (x_flat + mamba_out).transpose(1, 2).view(B, C, D, H, W)

        # --- 3. Convolution (Local Features) ---
        conv_out = self.conv(x_mamba_out)
        out = x_mamba_out + conv_out

        # Outer residual: block input → block output
        return residual + out

class MAMBlock(nn.Module):
    """
    MAM Block = MLC + AttMamba
    Based on AttmNet Diagram A.
    Combines efficient local convolution (MLC) with global context (AttMamba).
    """
    def __init__(self, channels, num_heads=None, d_state=64, window_size=0, use_attention=True):
        super().__init__()
        self.mlc = MLCBlock(channels)
        self.att_mamba = AttMamba2Block(
            channels,
            num_heads=num_heads,
            d_state=d_state,
            window_size=window_size,
            use_attention=use_attention
        )

    def forward(self, x):
        x = self.mlc(x)
        x = self.att_mamba(x)
        return x

class AttMamba2UNet(nn.Module):
    """
    Attention-Mamba2 U-Net for 3D volumetric segmentation.

    Args:
        in_channels: Number of input channels (default 2: CT + vesselness).
        out_channels: Number of output classes (default 2: background + vessel).
        features: Encoder channel counts per stage, e.g. (32, 64, 128, 256).
        patch_size: Expected input patch size; unused at runtime but kept for
                    compatibility with the model factory interface.
    """
    def __init__(self, in_channels=1, out_channels=2, features=(32, 64, 128, 256), patch_size=(96, 96, 96)):
        super().__init__()

        # Stem: stride-2 7×7 conv immediately halves spatial resolution
        # (96³ → 48³), reducing computational volume 8× before attention stages.
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, features[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm3d(features[0]),
            nn.LeakyReLU(inplace=True)
        )
        
        # Encoder
        # Stage 1 (48³): pure MLC — attention overhead not justified at this resolution
        self.enc1 = MLCBlock(features[0])
        self.down1 = nn.Sequential(
            nn.Conv3d(features[0], features[1], kernel_size=2, stride=2),
            nn.InstanceNorm3d(features[1]),
        )

        # Stage 2 (24³, ~14k tokens): MAM with windowed attention (window=8³)
        self.enc2 = MAMBlock(features[1], window_size=8, use_attention=True)
        self.down2 = nn.Sequential(
            nn.Conv3d(features[1], features[2], kernel_size=2, stride=2),
            nn.InstanceNorm3d(features[2]),
        )

        # Stage 3 (12³, ~1.7k tokens): 2× MAM with global attention
        self.enc3 = nn.Sequential(
            MAMBlock(features[2], window_size=0, use_attention=True),
            MAMBlock(features[2], window_size=0, use_attention=True),
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(features[2], features[3], kernel_size=2, stride=2),
            nn.InstanceNorm3d(features[3]),
        )

        # Bottleneck (6³): 2× MAM with global attention
        self.bottleneck = nn.Sequential(
            MAMBlock(features[3], window_size=0, use_attention=True),
            MAMBlock(features[3], window_size=0, use_attention=True),
        )
        
        # Decoder
        self.up3 = UnetUpBlock(spatial_dims=3, in_channels=features[3], out_channels=features[2], stride=2, kernel_size=3, upsample_kernel_size=2, norm_name="instance")
        self.up2 = UnetUpBlock(spatial_dims=3, in_channels=features[2], out_channels=features[1], stride=2, kernel_size=3, upsample_kernel_size=2, norm_name="instance")
        self.up1 = UnetUpBlock(spatial_dims=3, in_channels=features[1], out_channels=features[0], stride=2, kernel_size=3, upsample_kernel_size=2, norm_name="instance")
        
        # Final upsampling to recover resolution lost in Stem (stride 2)
        self.final_up = nn.ConvTranspose3d(features[0], features[0], kernel_size=2, stride=2)
        
        self.out = UnetOutBlock(spatial_dims=3, in_channels=features[0], out_channels=out_channels)

    def forward(self, x):
        # Encoder
        x0 = self.stem(x)       # (B, 32, D/2, H/2, W/2)
        x0 = self.enc1(x0)      
        
        x1 = self.down1(x0)     # (B, 64, D/4, ...)
        x1 = self.enc2(x1)
        
        x2 = self.down2(x1)     # (B, 128, D/8, ...)
        x2 = self.enc3(x2)
        
        x3 = self.down3(x2)     # (B, 256, D/16, ...)
        x3 = self.bottleneck(x3)
        
        # Decoder
        u2 = self.up3(x3, x2)
        u1 = self.up2(u2, x1)
        u0 = self.up1(u1, x0)   # (B, 32, D/2, H/2, W/2)
        
        # Final upsample
        u0 = self.final_up(u0)  # (B, 32, D, H, W)
        
        logits = self.out(u0)
        return logits

def get_att_mamba2_unet(patch_size, **kwargs):
    """Instantiate AttMamba2UNet; called by model_factory.get_model()."""
    return AttMamba2UNet(patch_size=patch_size, **kwargs)
