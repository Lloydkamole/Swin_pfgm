"""
Magnitude-Preserving SwinFIR UNet for PFGM++ (EDM2-style).

Main network architecture combining:
- UNet encoder-decoder structure
- Magnitude-preserving SwinFIR blocks (MP-SFB)
- EDM2 principles: no GroupNorm, weight normalization, magnitude preservation
- PFGM++ radius conditioning

Key differences from original network.py:
- MPConv instead of nn.Conv2d
- PixelNorm instead of GroupNorm
- mp_silu instead of standard activations
- mp_sum/mp_cat for combining features
- No learnable biases
"""

import torch
import torch.nn as nn
from typing import Tuple

from .embeddings import RadiusEmbedding
from .mp_fir_blocks import MPFIRResBlock, MPDownsample, MPUpsample
from .mp_layers import MPConv, mp_cat, mp_silu


class MPSwinFIRUNet(nn.Module):
    """
    Magnitude-Preserving SwinFIR-based UNet for PFGM++.
    
    Implements EDM2's magnitude-preserving principles throughout:
    - Weight-normalized convolutions
    - Pixel normalization instead of GroupNorm
    - Magnitude-preserving activations and operations
    
    Args:
        in_channels: Number of input channels (2: noisy_CT + MRI condition).
        out_channels: Number of output channels (1: predicted vector field).
        base_channels: Base channel dimension.
        channel_mults: Channel multipliers for each stage.
        num_res_blocks: Number of residual blocks per stage.
        radius_emb_dim: Dimension of radius embedding.
        attention_resolutions: Resolutions where attention is applied.
        num_heads: Number of attention heads.
        window_size: Window size for Swin attention.
        mlp_ratio: MLP expansion ratio in Swin blocks.
        use_sfb: Whether to use SFB (Spatial-Frequency Block).
        dropout: Dropout rate.
        drop_path_rate: Stochastic depth rate.
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        radius_emb_dim: int = 256,
        attention_resolutions: Tuple[int, ...] = (1, 2, 4),
        num_heads: int = 8,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        use_sfb: bool = True,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        img_size: int = 256,  # Add image size to track resolution
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.window_size = window_size
        self.img_size = img_size
        
        # Calculate stochastic depth rates
        num_stages = len(channel_mults)
        total_blocks = num_stages * num_res_blocks * 2  # encoder + decoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # Radius embedding (PFGM++ conditioning)
        self.radius_emb = RadiusEmbedding(embed_dim=radius_emb_dim)
        
        # Initial MP convolution
        self.input_conv = MPConv(in_channels, base_channels, kernel=[3, 3])
        
        # Encoder
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        current_channels = base_channels
        current_resolution = img_size
        block_idx = 0
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attention = (mult in attention_resolutions)
            
            stage_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                stage_blocks.append(
                    MPFIRResBlock(
                        in_channels=current_channels if j == 0 else out_ch,
                        out_channels=out_ch,
                        radius_emb_dim=radius_emb_dim,
                        use_swin=use_attention,
                        use_sfb=use_sfb,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[block_idx],
                        dropout=dropout,
                        input_resolution=(current_resolution, current_resolution),
                    )
                )
                current_channels = out_ch
                block_idx += 1
            
            self.encoder_stages.append(stage_blocks)
            
            # Downsample (except last stage)
            if i < num_stages - 1:
                self.downsamples.append(MPDownsample(current_channels))
                current_resolution = current_resolution // 2
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            MPFIRResBlock(
                in_channels=current_channels,
                out_channels=current_channels,
                radius_emb_dim=radius_emb_dim,
                use_swin=True,
                use_sfb=use_sfb,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[block_idx],
                dropout=dropout,
                input_resolution=(current_resolution, current_resolution),
            )
            for _ in range(num_res_blocks)
        ])
        block_idx += num_res_blocks
        
        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        
        for i in reversed(range(num_stages - 1)):
            mult = channel_mults[i]
            out_ch = base_channels * mult
            use_attention = (mult in attention_resolutions)
            
            # Upsample
            self.upsamples.append(MPUpsample(current_channels))
            current_resolution = current_resolution * 2
            
            # After upsampling and concatenation, input channels = current + skip
            skip_channels = out_ch
            concat_channels = current_channels + skip_channels
            
            stage_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                stage_blocks.append(
                    MPFIRResBlock(
                        in_channels=concat_channels if j == 0 else out_ch,
                        out_channels=out_ch,
                        radius_emb_dim=radius_emb_dim,
                        use_swin=use_attention,
                        use_sfb=use_sfb,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[block_idx],
                        dropout=dropout,
                        input_resolution=(current_resolution, current_resolution),
                    )
                )
                concat_channels = out_ch
                block_idx += 1
            
            self.decoder_stages.append(stage_blocks)
            current_channels = out_ch
        
        # Final output projection
        self.output_conv = MPConv(current_channels, out_channels, kernel=[3, 3])
    
    def forward(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MP-SwinFIR UNet.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
               (concatenated noisy_CT and MRI condition)
            radius: PFGM++ radius [B] or [B, 1]
        
        Returns:
            Predicted vector field [B, out_channels, H, W]
        """
        # Embed radius
        radius_emb = self.radius_emb(radius)  # [B, radius_emb_dim]
        
        # Initial convolution
        h = mp_silu(self.input_conv(x))
        
        # Encoder with skip connections
        encoder_outputs = []
        
        for stage_idx, stage_blocks in enumerate(self.encoder_stages):
            # Process blocks in stage
            for block in stage_blocks:
                h = block(h, radius_emb)
            
            # Save skip connection (except last stage)
            if stage_idx < len(self.encoder_stages) - 1:
                encoder_outputs.append(h)
                # Downsample
                h = self.downsamples[stage_idx](h)
        
        # Bottleneck
        for block in self.bottleneck:
            h = block(h, radius_emb)
        
        # Decoder with skip connections
        for stage_idx, (upsample, stage_blocks) in enumerate(zip(self.upsamples, self.decoder_stages)):
            # Upsample
            h = upsample(h)
            
            # Get skip connection (in reverse order)
            skip = encoder_outputs[-(stage_idx + 1)]
            
            # Magnitude-preserving concatenation
            h = mp_cat(h, skip, dim=1, t=0.5)
            
            # Process blocks in stage
            for block in stage_blocks:
                h = block(h, radius_emb)
        
        # Final output
        output = self.output_conv(h)
        
        return output
    
    def get_feature_channels(self) -> list:
        """Return list of channel counts at each encoder stage."""
        channels = []
        for mult in self.channel_mults:
            channels.append(self.base_channels * mult)
        return channels


def create_mp_swinfir_unet(
    img_size: int = 256,
    in_channels: int = 2,
    out_channels: int = 1,
    base_channels: int = 64,
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
    num_res_blocks: int = 2,
    attention_resolutions: Tuple[int, ...] = (1, 2, 4),
    num_heads: int = 8,
    window_size: int = 8,
    use_sfb: bool = True,
    dropout: float = 0.0,
    drop_path_rate: float = 0.1,
) -> MPSwinFIRUNet:
    """
    Factory function to create MP-SwinFIR UNet.
    
    Args:
        img_size: Input image size (assumes square).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        base_channels: Base channel dimension.
        channel_mults: Channel multipliers per stage.
        num_res_blocks: Number of residual blocks per stage.
        attention_resolutions: Which stages get attention.
        num_heads: Number of attention heads.
        window_size: Swin attention window size.
        use_sfb: Whether to use Spatial-Frequency Blocks.
        dropout: Dropout rate.
        drop_path_rate: Stochastic depth rate.
    
    Returns:
        Initialized MP-SwinFIR UNet model.
    """
    model = MPSwinFIRUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        window_size=window_size,
        use_sfb=use_sfb,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
    )
    
    return model
