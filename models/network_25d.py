"""
2.5D SwinFIR UNet for PFGM++ with 3D Convolutions.

Adapted from 2D network.py to handle small volumetric inputs.
Uses 3D convolutions throughout while maintaining the SwinFIR architecture.

Architecture:
    Input: [noisy_CT, MRI] concatenated (2 channels, D slices)
    Shape: [B, 2, D, H, W]
    ↓
    Initial Conv3D → embed_dim
    ↓
    Encoder stages (with 3D downsampling):
    - 3D convolutions with PFGM++ radius conditioning
    - 3D Downsample at each stage
    ↓
    Bottleneck:
    - 3D residual blocks
    ↓
    Decoder stages (with 3D upsampling):
    - 3D Upsample
    - Concatenate with skip connection
    - 3D residual blocks
    ↓
    Final Conv3D → output_channels
    ↓
    Output: Predicted vector field (1 channel, D slices)
    Shape: [B, 1, D, H, W]

Note: For simplicity, we use 3D convolutions without Swin attention
(which would require significant adaptation for 3D). We keep the
radius conditioning and multi-scale UNet structure.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RadiusEmbedding
from .swin_attention_3d import SwinTransformerBlock3D


def get_num_groups(channels: int, num_groups: int = 32, min_channels_per_group: int = 4) -> int:
    """Calculate valid number of groups for GroupNorm."""
    while channels % num_groups != 0 or channels // num_groups < min_channels_per_group:
        num_groups = num_groups // 2
        if num_groups < 1:
            return 1
    return num_groups


class ResBlock3D(nn.Module):
    """
    3D Residual Block with radius conditioning and optional 3D volumetric attention.
    
    For attention: applies 3D Swin attention on entire volume (D, H, W)
    using 3D window partitioning.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius_emb_dim: int,
        use_attention: bool = False,
        num_heads: int = 8,
        window_size: int = 8,
        window_size_3d: Tuple[int, int, int] = (2, 8, 8),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        # First conv block
        self.norm1 = nn.GroupNorm(get_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Radius embedding projection
        self.radius_proj = nn.Linear(radius_emb_dim, out_channels)
        
        # Optional 3D window-based attention (following paper's simple approach)
        if use_attention:
            # Use passed window_size_3d parameter (configurable)
            # Use TWO Swin blocks: first without shift, second with shift (paper's approach)
            shift_size_3d = (window_size_3d[0] // 2, window_size_3d[1] // 2, window_size_3d[2] // 2)
            self.attn_blocks = nn.ModuleList([
                # Block 1: No shift
                SwinTransformerBlock3D(
                    dim=out_channels,
                    input_resolution=(4, 256, 256),
                    num_heads=num_heads,
                    window_size=window_size_3d,
                    shift_size=(0, 0, 0),
                    mlp_ratio=mlp_ratio,
                    drop_path=0.0,
                ),
                # Block 2: With shift for cross-window communication
                SwinTransformerBlock3D(
                    dim=out_channels,
                    input_resolution=(4, 256, 256),
                    num_heads=num_heads,
                    window_size=window_size_3d,
                    shift_size=shift_size_3d,
                    mlp_ratio=mlp_ratio,
                    drop_path=0.0,
                ),
            ])
        else:
            self.attn_blocks = None
        
        # Second conv block
        self.norm2 = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, radius_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, D, H, W]
            radius_emb: Radius embedding [B, radius_emb_dim]
        
        Returns:
            Output tensor [B, out_channels, D, H, W]
        """
        h = x
        
        # First conv
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add radius conditioning
        radius_proj = self.radius_proj(radius_emb)[:, :, None, None, None]
        h = h + radius_proj
        
        # Apply 3D attention on entire volume if enabled
        if self.attn_blocks is not None:
            B, C, D, H, W = h.shape
            # Reshape for 3D attention: [B, C, D, H, W] -> [B, D*H*W, C]
            h_flat = h.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)
            # Apply 3D attention blocks sequentially
            for attn_block in self.attn_blocks:
                h_flat = attn_block(h_flat, (D, H, W))
            # Reshape back: [B, D*H*W, C] -> [B, C, D, H, W]
            h = h_flat.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        
        # Second conv
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip_conv(x)


class Downsample3D(nn.Module):
    """3D Downsampling layer - TRUE 3D UNet style: downsamples ALL dimensions."""
    
    def __init__(self, channels: int):
        super().__init__()
        # Standard 3D UNet: downsample all dimensions (D, H, W) by 2x
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D Upsampling layer - TRUE 3D UNet style: upsamples ALL dimensions."""
    
    def __init__(self, channels: int):
        super().__init__()
        # Use ConvTranspose3d for proper 3D upsampling (standard 3D UNet approach)
        self.conv = nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample all dimensions (D, H, W) by 2x
        return self.conv(x)


class SwinFIRUNet25D(nn.Module):
    """
    2.5D SwinFIR-based UNet for PFGM++ with 3D convolutions.
    
    Args:
        in_channels: Number of input channels (2: noisy_CT + MRI condition).
        out_channels: Number of output channels (1: predicted vector field).
        num_slices: Depth dimension (number of consecutive slices).
        base_channels: Base channel dimension.
        channel_mults: Channel multipliers for each stage.
        attention_resolutions: Resolutions where attention is applied.
        num_res_blocks: Number of residual blocks per stage.
        radius_emb_dim: Dimension of radius embedding.
        window_size: Window size for spatial attention (H, W) - used if window_size_3d is single tuple.
        window_size_3d: Either single tuple (D,H,W) or dict mapping stage_idx -> (D,H,W).
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        num_slices: int = 5,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        attention_resolutions: Tuple[int, ...] = (2, 4),
        num_res_blocks: int = 2,
        radius_emb_dim: int = 256,
        window_size: int = 8,
        window_size_3d = (2, 8, 8),
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_slices = num_slices
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.attention_resolutions = attention_resolutions
        self.num_res_blocks = num_res_blocks
        self.window_size = window_size
        
        # Convert window_size_3d to per-stage dict if it's a single tuple
        if isinstance(window_size_3d, (tuple, list)):
            # Single window size for all stages
            self.window_size_3d = {i: tuple(window_size_3d) for i in range(len(channel_mults))}
        else:
            # Already a dict mapping stage -> window_size
            self.window_size_3d = window_size_3d
        
        # Radius embedding (PFGM++ conditioning)
        self.radius_emb = RadiusEmbedding(embed_dim=radius_emb_dim)
        
        # Initial 3D convolution
        self.input_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        current_channels = base_channels
        num_stages = len(channel_mults)
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attention = (mult in attention_resolutions)
            
            # Get stage-specific window size
            stage_window_size_3d = self.window_size_3d.get(i, (2, 8, 8))
            
            stage_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                stage_blocks.append(
                    ResBlock3D(
                        in_channels=current_channels if j == 0 else out_ch,
                        out_channels=out_ch,
                        radius_emb_dim=radius_emb_dim,
                        use_attention=use_attention,
                        window_size=window_size,
                        window_size_3d=stage_window_size_3d,
                        dropout=dropout,
                    )
                )
                current_channels = out_ch
            
            self.encoder_stages.append(stage_blocks)
            
            # Downsample (except last stage)
            if i < num_stages - 1:
                self.downsamples.append(Downsample3D(current_channels))
        
        # Bottleneck (use last stage's window size)
        bottleneck_window_size_3d = self.window_size_3d.get(len(channel_mults) - 1, (2, 8, 8))
        self.bottleneck = nn.ModuleList([
            ResBlock3D(
                in_channels=current_channels,
                out_channels=current_channels,
                radius_emb_dim=radius_emb_dim,
                window_size=window_size,
                window_size_3d=bottleneck_window_size_3d,
                dropout=dropout,
            )
            for _ in range(num_res_blocks)
        ])
        
        # Decoder
        self.decoder_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i in reversed(range(num_stages)):
            mult = channel_mults[i]
            prev_mult = channel_mults[i + 1] if i < num_stages - 1 else channel_mults[-1]
            out_ch = base_channels * mult
            in_ch = base_channels * prev_mult
            use_attention = (mult in attention_resolutions)
            
            # Get stage-specific window size
            stage_window_size_3d = self.window_size_3d.get(i, (2, 8, 8))
            
            # Upsample (except first decoder stage)
            if i < num_stages - 1:
                self.upsamples.append(Upsample3D(in_ch))
            
            # Skip connection conv (to match channels after concatenation)
            # First decoder stage (i == num_stages-1): just bottleneck channels
            # Other stages: upsampled channels (in_ch) + skip channels (out_ch)
            if i == num_stages - 1:
                skip_in_ch = in_ch  # Just bottleneck output, no concatenation
            else:
                skip_in_ch = in_ch + out_ch  # After upsample (in_ch) + skip (out_ch)
            self.skip_convs.append(nn.Conv3d(skip_in_ch, out_ch, kernel_size=1))
            
            stage_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                stage_blocks.append(
                    ResBlock3D(
                        in_channels=out_ch,
                        out_channels=out_ch,
                        radius_emb_dim=radius_emb_dim,
                        use_attention=use_attention,
                        window_size=window_size,
                        window_size_3d=stage_window_size_3d,
                        dropout=dropout,
                    )
                )
            
            self.decoder_stages.append(stage_blocks)
            current_channels = out_ch
        
        # Final layers
        self.final_norm = nn.GroupNorm(get_num_groups(base_channels), base_channels)
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)
        
        # Initialize output conv to small values
        nn.init.zeros_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                if m is self.final_conv:
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mri: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy CT volume [B, 1, D, H, W]
            mri: MRI condition volume [B, 1, D, H, W]
            r: PFGM++ radius [B] or [B, 1]
        
        Returns:
            Predicted vector field [B, 1, D, H, W]
        """
        # Ensure all inputs are float32
        x = x.float()
        mri = mri.float()
        r = r.float()
        
        # Ensure r is 1D
        if r.dim() > 1:
            r = r.squeeze(-1)
        
        # Concatenate noisy CT with MRI condition
        x = torch.cat([x, mri], dim=1)  # [B, 2, D, H, W]
        
        # Radius embedding
        r_emb = self.radius_emb(r)  # [B, radius_emb_dim]
        
        # Initial conv
        h = self.input_conv(x)  # [B, base_channels, D, H, W]
        
        # Encoder with skip connections
        skips = []
        for i, (stage, downsample) in enumerate(zip(
            self.encoder_stages,
            self.downsamples + [None]  # No downsample after last stage
        )):
            for block in stage:
                h = block(h, r_emb)
            
            skips.append(h)
            
            if downsample is not None:
                h = downsample(h)
        
        # Bottleneck
        for block in self.bottleneck:
            h = block(h, r_emb)
        
        # Decoder with skip connections
        for i, (stage, skip_conv) in enumerate(zip(
            self.decoder_stages,
            self.skip_convs
        )):
            # Upsample (except first decoder stage)
            if i > 0:
                h = self.upsamples[i - 1](h)
                
                # Concatenate with skip connection
                skip = skips[-(i + 1)]
                
                # Match dimensions if needed (due to downsampling asymmetry)
                if h.shape != skip.shape:
                    # Adjust h to match skip dimensions
                    target_shape = skip.shape[2:]  # (D, H, W)
                    if h.shape[2:] != target_shape:
                        h = F.interpolate(h, size=target_shape, mode='trilinear', align_corners=False)
                
                h = torch.cat([h, skip], dim=1)
            
            # Process skip concatenation
            h = skip_conv(h)
            
            # Apply blocks
            for block in stage:
                h = block(h, r_emb)
        
        # Final output
        h = self.final_norm(h)
        h = F.silu(h)
        output = self.final_conv(h)
        
        return output


def create_swin_fir_unet_25d(
    in_channels: int = 2,
    out_channels: int = 1,
    num_slices: int = 5,
    base_channels: int = 64,
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
    attention_resolutions: Tuple[int, ...] = (2, 4),
    num_res_blocks: int = 2,
    radius_emb_dim: int = 256,
    window_size: int = 8,
    window_size_3d: Tuple[int, int, int] = (2, 8, 8),
    dropout: float = 0.0,
) -> SwinFIRUNet25D:
    """
    Factory function to create a 2.5D SwinFIR UNet.
    
    Args:
        in_channels: Number of input channels (2: CT + MRI).
        out_channels: Number of output channels (1: predicted field).
        num_slices: Number of slices in depth dimension.
        base_channels: Base channel count.
        channel_mults: Channel multipliers per stage.
        attention_resolutions: Resolutions where attention is applied.
        num_res_blocks: Number of residual blocks per stage.
        radius_emb_dim: Dimension of radius embeddings.
        window_size: Window size for spatial attention (H, W).
        window_size_3d: 3D window size (D, H, W) for volumetric attention.
        dropout: Dropout rate.
    
    Returns:
        Configured 2.5D SwinFIR UNet model.
    """
    return SwinFIRUNet25D(
        in_channels=in_channels,
        out_channels=out_channels,
        num_slices=num_slices,
        base_channels=base_channels,
        channel_mults=channel_mults,
        attention_resolutions=attention_resolutions,
        num_res_blocks=num_res_blocks,
        radius_emb_dim=radius_emb_dim,
        window_size=window_size,
        window_size_3d=window_size_3d,
        dropout=dropout,
    )

