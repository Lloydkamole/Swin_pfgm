"""
SwinFIR Building Blocks for PFGM-FIR.

This module contains the core building blocks from SwinFIR:
- SFB (Spatial-Frequency Block): Combines spatial convolution with FFT for global receptive field
- SpectralTransform: Fast Fourier Convolution component
- FourierUnit: Core FFT operation
- FIRResBlock: Residual block with SwinFIR components + radius conditioning

The key innovation from SwinFIR is the SFB block that gives image-wide receptive
field in early layers through Fast Fourier Convolution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, Tuple

from .swin_attention import SwinTransformerBlock, AdaLNZeroSwinTransformerBlock
from .deformable_attention import AdaLNZeroDeformableBlock


class LayerNorm2d(nn.Module):
    """
    LayerNorm for 2D feature maps (channels-first format).
    
    This is preferred over GroupNorm for diffusion models because:
    1. Normalizes each sample independently (better for varying batch sizes)
    2. More consistent statistics across different channel configurations
    3. Used by modern architectures like DiT, Stable Diffusion 3, etc.
    
    Args:
        num_features: Number of channels (C in [B, C, H, W])
        eps: Small constant for numerical stability
        elementwise_affine: Whether to learn scale and shift parameters
    """
    
    def __init__(self, num_features: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Normalized tensor of shape [B, C, H, W]
        """
        # Normalize over C, H, W dimensions (keep batch dimension independent)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            # Apply learnable scale and shift (broadcast over H, W)
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return x


def get_num_groups(channels: int, target: int = 32) -> int:
    """Get a valid number of groups for GroupNorm, targeting `target` groups."""
    while channels % target != 0 and target > 1:
        target -= 1
    return target


class AdaGroupNorm(nn.Module):
    """
    Adaptive Group Normalization (AdaGN) for conditioning conv blocks.
    
    Replaces the LayerNorm2d + FiLM pattern with a single module that:
    1. Applies GroupNorm (without learnable affine)
    2. Predicts per-channel scale and shift from the conditioning embedding
    3. Applies the adaptive affine transform
    
    This is the standard conditioning mechanism in ADM (Dhariwal & Nichol, 2021)
    and subsequent diffusion models. It folds FiLM into the normalization layer.
    
    Args:
        num_channels: Number of feature channels
        r_emb_dim: Dimension of the conditioning embedding
        num_groups: Number of groups for GroupNorm (default 32)
    """
    
    def __init__(self, num_channels: int, r_emb_dim: int, num_groups: int = 32):
        super().__init__()
        num_groups = get_num_groups(num_channels, num_groups)
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(r_emb_dim, num_channels * 2),
        )
    
    def forward(self, x: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [B, C, H, W]
            r_emb: Conditioning embedding [B, r_emb_dim]
        
        Returns:
            Adaptively normalized tensor [B, C, H, W]
        """
        x = self.gn(x)
        scale, shift = self.proj(r_emb).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class FourierUnit(nn.Module):
    """
    Core Fourier Transform unit.
    
    Transforms input to frequency domain, applies convolution,
    and transforms back to spatial domain.
    
    This gives image-wide receptive field because convolution
    in frequency domain corresponds to global operations in spatial domain.
    """
    
    def __init__(self, embed_dim: int, fft_norm: str = 'ortho'):
        super().__init__()
        
        # Conv in frequency domain (operates on real+imag stacked)
        self.conv_layer = nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
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
        
        # FFT and torch.complex require float32 - doesn't support bfloat16/float16
        # Force everything to float32 for numerical stability
        x = x.float()
        
        # Transform to frequency domain
        # Output: complex tensor of shape [B, C, H, W//2+1]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        
        # Stack real and imaginary parts: [B, C, 2, H, W//2+1]
        ffted = torch.stack((ffted.real, ffted.imag), dim=2)
        
        # Reshape for conv: [B, C*2, H, W//2+1]
        ffted = ffted.view(batch, -1, ffted.size(3), ffted.size(4))
        
        # Apply convolution in frequency domain (force float32)
        ffted = self.conv_layer(ffted.float()).float()
        ffted = self.relu(ffted)
        
        # Reshape back: [B, C, 2, H, W//2+1]
        ffted = ffted.view(batch, -1, 2, ffted.size(2), ffted.size(3))
        
        # Convert back to complex (requires float32)
        ffted = torch.complex(ffted[:, :, 0, :, :], ffted[:, :, 1, :, :])
        
        # Inverse FFT back to spatial domain
        output = torch.fft.irfftn(ffted, s=input_shape, dim=fft_dim, norm=self.fft_norm)
        
        # Convert back to original dtype for mixed precision compatibility
        return output.to(input_dtype)


class SpectralTransform(nn.Module):
    """
    Spectral Transform module from SwinFIR.
    
    Architecture:
    - Conv 1x1 to reduce channels
    - FourierUnit for global processing
    - Conv 1x1 to restore channels
    - Optional final 3x3 conv
    
    This provides global receptive field through FFT operations.
    """
    
    def __init__(self, embed_dim: int, last_conv: bool = False):
        super().__init__()
        self.last_conv = last_conv
        
        # Reduce channels before FFT (efficiency)
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # FFT processing
        self.fu = FourierUnit(embed_dim // 2)
        
        # Restore channels
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1)
        
        # Optional final conv
        if last_conv:
            self.final_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        else:
            self.final_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Output tensor of shape [B, C, H, W]
        """
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)  # Residual around FourierUnit
        
        if self.final_conv is not None:
            output = self.final_conv(output)
        
        return output


class ResB(nn.Module):
    """
    Simple Residual Block (spatial branch).
    
    Two 3x3 convolutions with LeakyReLU, with scaled residual.
    """
    
    def __init__(self, embed_dim: int, reduction: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // reduction, embed_dim, kernel_size=3, padding=1),
        )
        self.scale = 1.0 / math.sqrt(2)  # Scale residual for stability
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x + self.body(x))


class SFB(nn.Module):
    """
    Spatial-Frequency Block (SFB) from SwinFIR.
    
    The key innovation: combines local spatial features with global frequency features.
    
    Architecture:
    - Spatial branch (S): ResB for local features via convolution
    - Frequency branch (F): SpectralTransform for global features via FFT
    - Fusion: Concatenate + GroupNorm + 1x1 conv
    
    This gives the network both local detail preservation and global context
    understanding simultaneously.
    """
    
    def __init__(self, embed_dim: int, reduction: int = 1):
        super().__init__()
        
        # Spatial branch: local features
        self.S = ResB(embed_dim, reduction)
        
        # Frequency branch: global features
        self.F = SpectralTransform(embed_dim)
        
        # Normalization before fusion (LayerNorm2d for consistent statistics)
        self.norm = LayerNorm2d(embed_dim * 2)
        
        # Fusion: combine spatial and frequency features
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Output tensor of shape [B, C, H, W]
        """
        # Spatial features (local)
        s = self.S(x)
        
        # Frequency features (global)
        f = self.F(x)
        
        # Concatenate, normalize, and fuse
        out = torch.cat([s, f], dim=1)
        out = self.norm(out)
        out = self.fusion(out)
        
        return out


class ChannelAttention(nn.Module):
    """
    Channel Attention module (squeeze-and-excitation style).
    
    Used in the CAB (Channel Attention Block) from SwinFIR.
    """
    
    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class CAB(nn.Module):
    """
    Channel Attention Block from SwinFIR.
    
    Combines convolution with channel attention for adaptive feature recalibration.
    """
    
    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, kernel_size=3, padding=1),
            ChannelAttention(num_feat, squeeze_factor),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


class FIRResBlock(nn.Module):
    """
    FIR Residual Block: Core building block for PFGM-FIR.
    
    Sequential architecture following SwinFIR design:
    1. Conv with radius conditioning (FiLM) - injects noise level info BEFORE attention
    2. Swin Transformer attention blocks (sequential)
    3. SFB blocks (sequential) for global receptive field
    4. Residual connection back to input
    
    Flow: input → conv+FiLM → Swin blocks → SFB blocks → + input → output
    
    This is different from the parallel approach - here Swin and SFB are sequential,
    matching the original SwinFIR paper design.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        radius_emb_dim: Dimension of radius embedding.
        use_swin: Whether to use Swin attention.
        use_sfb: Whether to use SFB blocks.
        input_resolution: (H, W) for Swin attention.
        num_heads: Number of attention heads.
        window_size: Window size for Swin attention.
        mlp_ratio: MLP expansion ratio.
        drop_path: Stochastic depth rate.
        dropout: Dropout rate.
        num_swin_blocks: Number of Swin blocks (W-MSA + SW-MSA pattern).
        num_sfb_blocks: Number of SFB blocks applied after Swin.
        gradient_checkpoint: Enable gradient checkpointing for Swin/SFB blocks
            to trade ~30% speed for ~60% activation memory savings.
        attention_type: Type of attention mechanism ('window' or 'deformable').
        deformable_kwargs: Extra kwargs for DeformableAttention (n_groups,
            offset_range_factor, kv_stride, etc.). Only used when
            attention_type='deformable'.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius_emb_dim: int,
        use_swin: bool = False,
        use_sfb: bool = True,
        input_resolution: Tuple[int, int] = (64, 64),
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        num_swin_blocks: int = 2,
        num_sfb_blocks: int = 1,
        gradient_checkpoint: bool = False,
        attention_type: str = 'window',
        deformable_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_swin = use_swin
        self.use_sfb = use_sfb
        self.num_swin_blocks = num_swin_blocks
        self.num_sfb_blocks = num_sfb_blocks
        self.use_gradient_checkpoint = gradient_checkpoint
        self.attention_type = attention_type
        
        # === Stage 1: Conv + FiLM (radius conditioning BEFORE attention) ===
        # === Stage 1: AdaGN + Conv (radius conditioning via adaptive normalization) ===
        self.adagn = AdaGroupNorm(in_channels, radius_emb_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # === Stage 2: Attention blocks (sequential) ===
        if use_swin and num_swin_blocks > 0:
            if attention_type == 'deformable':
                _dkw = deformable_kwargs or {}
                self.swin_blocks = nn.ModuleList([
                    AdaLNZeroDeformableBlock(
                        dim=out_channels,
                        num_heads=num_heads,
                        r_emb_dim=radius_emb_dim,
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_path,
                        **_dkw,
                    )
                    for _ in range(num_swin_blocks)
                ])
            else:
                self.swin_blocks = nn.ModuleList([
                    AdaLNZeroSwinTransformerBlock(
                        dim=out_channels,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        r_emb_dim=radius_emb_dim,
                        window_size=window_size,
                        shift_size=0 if i % 2 == 0 else window_size // 2,  # W-MSA / SW-MSA alternating
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_path,
                    )
                    for i in range(num_swin_blocks)
                ])
        else:
            self.swin_blocks = None
        
        # === Stage 3: SFB Blocks (sequential, applied AFTER Swin) ===
        if use_sfb and num_sfb_blocks > 0:
            self.sfb_blocks = nn.ModuleList([
                SFB(out_channels)
                for _ in range(num_sfb_blocks)
            ])
        else:
            self.sfb_blocks = None
        
        # === Skip connection (if channels change) ===
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
        
        # Residual scaling factor for stability
        self.residual_scale = 1.0 / math.sqrt(2)
        
        # Initialize conv to small values for better training dynamics
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        r_emb: torch.Tensor,
        x_size: Optional[Tuple[int, int]] = None,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with sequential Swin → SFB flow.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            r_emb: Radius embedding of shape [B, radius_emb_dim]
            x_size: Optional (H, W) for Swin attention (uses actual size if None)
            spatial_mask: Optional binary body mask [B, 1, H_orig, W_orig].
                Downsampled to current feature resolution and passed to
                deformable attention blocks to suppress background K/V.
        
        Returns:
            Output tensor of shape [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        
        # === Skip connection (identity or channel projection) ===
        identity = self.skip(x)
        
        # === Stage 1: AdaGN + Conv (conditioning folded into normalization) ===
        h = self.adagn(x, r_emb)
        h = F.silu(h)
        h = self.conv1(h)
        
        # === Stage 2: Swin Attention with AdaLN-Zero (sequential, with gradient checkpointing) ===
        if self.swin_blocks is not None:
            # Reshape for Swin: [B, C, H, W] -> [B, H*W, C]
            h_flat = h.flatten(2).transpose(1, 2)
            current_size = x_size if x_size is not None else (H, W)

            # Downsample spatial mask to current feature resolution for attention masking
            mask_at_res = None
            if spatial_mask is not None and self.attention_type == 'deformable':
                mask_at_res = F.interpolate(
                    spatial_mask.float(), size=(H, W), mode='nearest'
                )  # [B, 1, H, W]

            for swin_block in self.swin_blocks:
                if self.training and self.use_gradient_checkpoint:
                    if mask_at_res is not None:
                        h_flat = grad_checkpoint(
                            swin_block, h_flat, current_size, r_emb, mask_at_res,
                            use_reentrant=False,
                        )
                    else:
                        h_flat = grad_checkpoint(
                            swin_block, h_flat, current_size, r_emb,
                            use_reentrant=False,
                        )
                else:
                    if mask_at_res is not None:
                        h_flat = swin_block(
                            h_flat, current_size, r_emb, spatial_mask=mask_at_res
                        )
                    else:
                        h_flat = swin_block(h_flat, current_size, r_emb)

            # Reshape back: [B, H*W, C] -> [B, C, H, W]
            h = h_flat.transpose(1, 2).view(B, -1, H, W)
        
        # === Stage 3: SFB Blocks (sequential, AFTER Swin) ===
        if self.sfb_blocks is not None:
            for sfb_block in self.sfb_blocks:
                if self.training and self.use_gradient_checkpoint:
                    h = grad_checkpoint(
                        sfb_block, h,
                        use_reentrant=False,
                    )
                else:
                    h = sfb_block(h)
        
        # === Residual connection: scale only the learned branch ===
        return identity + self.residual_scale * h


class Downsample(nn.Module):
    """
    Downsampling layer.
    
    Uses strided convolution for learnable downsampling.
    """
    
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling layer.
    
    Uses nearest-neighbor interpolation followed by convolution.
    """
    
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
