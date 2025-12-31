"""
Magnitude-Preserving FIR Blocks for EDM2-style SwinFIR.

Implements SwinFIR building blocks with EDM2's magnitude-preserving principles:
- No GroupNorm (replaced with pixel normalization)
- Weight-normalized convolutions
- Magnitude-preserving activations and operations
- No learnable biases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .mp_layers import MPConv, mp_silu, mp_sum, mp_cat, normalize
from .swin_attention import SwinTransformerBlock


class PixelNorm(nn.Module):
    """
    Pixel normalization as alternative to GroupNorm.
    Normalizes feature vectors independently at each spatial location.
    """
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Normalized tensor [B, C, H, W]
        """
        return normalize(x, dim=1, eps=self.eps)


class MPFourierUnit(nn.Module):
    """
    Magnitude-preserving Fourier Transform unit.
    
    Transforms input to frequency domain, applies MP convolution,
    and transforms back to spatial domain.
    """
    
    def __init__(self, embed_dim: int, fft_norm: str = 'ortho'):
        super().__init__()
        
        # MP Conv in frequency domain (operates on real+imag stacked)
        self.conv_layer = MPConv(embed_dim * 2, embed_dim * 2, kernel=[1, 1])
        self.fft_norm = fft_norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Output tensor of shape [B, C, H, W]
        """
        batch = x.shape[0]
        input_dtype = x.dtype
        input_shape = x.shape[-2:]
        
        # FFT requires float32
        x = x.float()
        
        # Transform to frequency domain
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        
        # Stack real and imaginary parts: [B, C, 2, H, W//2+1]
        ffted = torch.stack((ffted.real, ffted.imag), dim=2)
        
        # Reshape for conv: [B, C*2, H, W//2+1]
        ffted = ffted.view(batch, -1, ffted.size(3), ffted.size(4))
        
        # Apply MP convolution in frequency domain
        ffted = mp_silu(self.conv_layer(ffted.float()))
        
        # Reshape back: [B, C, 2, H, W//2+1]
        ffted = ffted.view(batch, -1, 2, ffted.size(2), ffted.size(3))
        
        # Convert back to complex
        ffted = torch.complex(ffted[:, :, 0, :, :], ffted[:, :, 1, :, :])
        
        # Inverse FFT back to spatial domain
        output = torch.fft.irfftn(ffted, s=input_shape, dim=fft_dim, norm=self.fft_norm)
        
        return output.to(input_dtype)


class MPSpectralTransform(nn.Module):
    """
    Magnitude-preserving Spectral Transform module.
    
    Architecture:
    - MP Conv 1x1 to reduce channels
    - MPFourierUnit for global processing
    - MP Conv 1x1 to restore channels
    - Optional final 3x3 MP conv
    """
    
    def __init__(self, embed_dim: int, last_conv: bool = False):
        super().__init__()
        
        # Channel reduction
        self.conv1 = MPConv(embed_dim, embed_dim // 2, kernel=[1, 1])
        
        # Fourier processing
        self.fu = MPFourierUnit(embed_dim // 2)
        
        # Channel expansion
        self.conv2 = MPConv(embed_dim // 2, embed_dim, kernel=[1, 1])
        
        # Optional spatial conv
        self.last_conv = None
        if last_conv:
            self.last_conv = MPConv(embed_dim, embed_dim, kernel=[3, 3])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Output tensor [B, C, H, W]
        """
        x = mp_silu(self.conv1(x))
        x = self.fu(x)
        x = self.conv2(x)
        
        if self.last_conv is not None:
            x = mp_silu(self.last_conv(x))
        
        return x


class MPSFB(nn.Module):
    """
    Magnitude-preserving Spatial-Frequency Block (SFB).
    
    Parallel spatial and frequency processing with MP operations.
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Spatial branch
        self.spatial_conv = MPConv(embed_dim, embed_dim, kernel=[3, 3])
        
        # Frequency branch
        self.spectral = MPSpectralTransform(embed_dim, last_conv=True)
        
        # Pixel normalization instead of GroupNorm
        self.norm = PixelNorm()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Output tensor [B, C, H, W]
        """
        # Spatial processing
        spatial_out = mp_silu(self.spatial_conv(x))
        
        # Frequency processing
        freq_out = self.spectral(x)
        
        # Magnitude-preserving sum (equal weight)
        out = mp_sum(spatial_out, freq_out, t=0.5)
        
        # Pixel normalization
        out = self.norm(out)
        
        return out


class MPFIRResBlock(nn.Module):
    """
    Magnitude-preserving FIR Residual Block.
    
    Combines:
    - Optional Swin Transformer attention
    - Optional MP SFB (Spatial-Frequency Block)
    - Radius conditioning via adaptive normalization
    - MP convolutions and activations
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius_emb_dim: int,
        use_swin: bool = False,
        use_sfb: bool = True,
        num_heads: int = 8,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        input_resolution: Tuple[int, int] = (256, 256),  # Default resolution
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_swin = use_swin
        self.use_sfb = use_sfb
        
        # Radius conditioning: project to scale and shift parameters
        # Each channel gets its own scale and shift
        self.radius_proj = MPConv(radius_emb_dim, out_channels * 2, kernel=[])
        
        # Input projection if channels change
        self.in_proj = None
        if in_channels != out_channels:
            self.in_proj = MPConv(in_channels, out_channels, kernel=[1, 1])
        
        # Pixel normalization layers
        self.norm1 = PixelNorm()
        self.norm2 = PixelNorm()
        
        # First conv block
        self.conv1 = MPConv(in_channels, out_channels, kernel=[3, 3])
        
        # Swin Transformer block (optional)
        self.swin_block = None
        if use_swin:
            # Note: Using LayerNorm instead of PixelNorm for Swin attention
            # PixelNorm normalizes across channels, LayerNorm normalizes within channels
            self.swin_block = SwinTransformerBlock(
                dim=out_channels,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=dropout,
                attn_drop=dropout,
                drop_path=drop_path,
                norm_layer=nn.LayerNorm,  # Keep LayerNorm for Swin (expects [B,H,W,C])
            )
        
        # SFB block (optional)
        self.sfb_block = None
        if use_sfb:
            self.sfb_block = MPSFB(out_channels)
        
        # Second conv block
        self.conv2 = MPConv(out_channels, out_channels, kernel=[3, 3])
    
    def forward(self, x: torch.Tensor, radius_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            radius_emb: Radius embedding [B, radius_emb_dim]
        
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        # Project input if needed
        identity = x
        if self.in_proj is not None:
            identity = self.in_proj(identity)
        
        # First conv block with pixel norm
        h = self.norm1(x)
        h = mp_silu(self.conv1(h))
        
        # Swin attention (optional)
        if self.swin_block is not None:
            B, C, H, W = h.shape
            # Swin expects [B, L, C] where L = H*W
            h = h.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            h = h.view(B, H*W, C)  # Flatten to [B, L, C]
            h = self.swin_block(h, (H, W))
            # Back to [B, C, H, W]
            h = h.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # SFB block (optional)
        if self.sfb_block is not None:
            h = self.sfb_block(h)
        
        # Second conv block with adaptive normalization
        h = self.norm2(h)
        
        # Radius conditioning: scale and shift
        # Project radius embedding to [B, out_channels*2]
        radius_params = self.radius_proj(radius_emb.unsqueeze(-1).unsqueeze(-1))
        radius_params = radius_params.squeeze(-1).squeeze(-1)  # [B, out_channels*2]
        scale, shift = radius_params.chunk(2, dim=1)  # Each [B, out_channels]
        
        # Apply adaptive normalization: scale * h + shift
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, out_channels, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # [B, out_channels, 1, 1]
        h = scale * h + shift
        
        h = mp_silu(self.conv2(h))
        
        # Magnitude-preserving residual connection
        out = mp_sum(identity, h, t=0.5)
        
        return out


class MPDownsample(nn.Module):
    """Magnitude-preserving downsampling with 2x2 stride-2 conv."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = MPConv(channels, channels, kernel=[2, 2])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, H, W]
        Returns:
            Output [B, C, H//2, W//2]
        """
        # Stride-2 convolution
        w = self.conv.weight.to(x.dtype)
        gain = 1.0
        w = normalize(w) * (gain / torch.sqrt(torch.tensor(w[0].numel())))
        
        return F.conv2d(x, w, stride=2, padding=0)


class MPUpsample(nn.Module):
    """Magnitude-preserving upsampling with nearest + 3x3 conv."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = MPConv(channels, channels, kernel=[3, 3])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, H, W]
        Returns:
            Output [B, C, H*2, W*2]
        """
        # Nearest neighbor upsampling
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        
        # Refine with conv
        x = self.conv(x)
        
        return x
