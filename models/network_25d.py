"""
2.5D SwinFIR UNet for PFGM++ with 3D Convolutions — FIR architecture.

Volumetric counterpart to `network.py`. Mirrors the 2D `FIRResBlock` structure
on 5D tensors [B, C, D, H, W]:

    input → AdaGN+Conv (radius-conditioned) → attention blocks
          → SFB3D blocks (FFT global receptive field) → residual

Two attention paths (selectable via `attention_type`):
    - 'window'     : SwinTransformerBlock3D (existing)
    - 'deformable' : AdaLNZeroDeformableBlock3D (new, 3D DAT++ with QK-Norm)

Also supports body-mask training (`in_channels=3`-style threading) and passes
the body mask through deformable attention to suppress background K/V.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .embeddings import RadiusEmbedding
from .swin_attention_3d import SwinTransformerBlock3D
from .deformable_attention_3d import AdaLNZeroDeformableBlock3D


# =============================================================================
# Helpers
# =============================================================================

def get_num_groups(channels: int, num_groups: int = 32, min_channels_per_group: int = 4) -> int:
    """Calculate valid number of groups for GroupNorm."""
    while channels % num_groups != 0 or channels // num_groups < min_channels_per_group:
        num_groups = num_groups // 2
        if num_groups < 1:
            return 1
    return num_groups


class LayerNorm3d(nn.Module):
    """LayerNorm for 5D feature maps (channels-first), used in SFB3D fusion."""

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
        mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        var = x.var(dim=(1, 2, 3, 4), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return x


class AdaGroupNorm3D(nn.Module):
    """Adaptive GroupNorm for 5D features (radius conditioning via FiLM)."""

    def __init__(self, num_channels: int, r_emb_dim: int, num_groups: int = 32):
        super().__init__()
        num_groups = get_num_groups(num_channels, num_groups)
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(r_emb_dim, num_channels * 2),
        )

    def forward(self, x: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
        x = self.gn(x)
        scale, shift = self.proj(r_emb).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]


# =============================================================================
# 3D SFB (Spatial-Frequency Block)
# =============================================================================

class FourierUnit3D(nn.Module):
    """3D FFT unit: rfftn over (D, H, W), 1x1x1 conv on real+imag, irfftn."""

    def __init__(self, embed_dim: int, fft_norm: str = 'ortho'):
        super().__init__()
        self.conv_layer = nn.Conv3d(embed_dim * 2, embed_dim * 2, kernel_size=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        input_dtype = x.dtype
        input_shape = x.shape[-3:]  # (D, H, W)
        x = x.float()
        fft_dim = (-3, -2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        # Stack real/imag: [B, C, 2, D, H, W//2+1]
        ffted = torch.stack((ffted.real, ffted.imag), dim=2)
        # Reshape for conv: [B, 2C, D, H, W//2+1]
        ffted = ffted.view(batch, -1, ffted.size(3), ffted.size(4), ffted.size(5))
        ffted = self.conv_layer(ffted.float()).float()
        ffted = self.relu(ffted)
        ffted = ffted.view(batch, -1, 2, ffted.size(2), ffted.size(3), ffted.size(4))
        ffted = torch.complex(ffted[:, :, 0], ffted[:, :, 1])
        output = torch.fft.irfftn(ffted, s=input_shape, dim=fft_dim, norm=self.fft_norm)
        return output.to(input_dtype)


class SpectralTransform3D(nn.Module):
    """Reduce → FourierUnit3D → restore (mirrors 2D SpectralTransform)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.fu = FourierUnit3D(embed_dim // 2)
        self.conv2 = nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        out = self.fu(x)
        out = self.conv2(x + out)  # residual around FFT
        return out


class DeformableConv3d(nn.Module):
    """
    Lightweight pure-PyTorch deformable 3D convolution.

    Offsets are predicted in voxel units and zero-initialized. By default only
    H/W offsets are deformable, while the kernel can still span depth.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        offset_scale: float = 1.0,
        use_modulation: bool = False,
        deformable_depth: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            kernel_size = tuple(kernel_size)
        if any(k % 2 == 0 for k in kernel_size):
            raise ValueError("DeformableConv3d expects odd kernel sizes.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = tuple(k // 2 for k in kernel_size)
        self.offset_scale = float(offset_scale)
        self.use_modulation = bool(use_modulation)
        self.deformable_depth = bool(deformable_depth)
        self.kernel_points = kernel_size[0] * kernel_size[1] * kernel_size[2]
        self.offset_dims = 3 if self.deformable_depth else 2

        offset_channels = self.offset_dims * self.kernel_points
        if self.use_modulation:
            offset_channels += self.kernel_points
        self.offset_net = nn.Conv3d(
            in_channels,
            offset_channels,
            kernel_size=3,
            padding=1,
        )
        self.offset_net._skip_reinit = True

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_points)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.offset_net.weight)
        nn.init.zeros_(self.offset_net.bias)
        if self.use_modulation:
            with torch.no_grad():
                start = self.offset_dims * self.kernel_points
                self.offset_net.bias[start:].fill_(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        B, C, D, H, W = x.shape
        kd, kh, kw = self.kernel_size
        pd, ph, pw = self.padding

        offset_mod = self.offset_net(x).float()
        offset_channels = self.offset_dims * self.kernel_points
        offsets = offset_mod[:, :offset_channels]
        offsets = offsets.view(B, self.kernel_points, self.offset_dims, D, H, W)
        offsets = offsets * self.offset_scale

        modulation = None
        if self.use_modulation:
            modulation = offset_mod[:, offset_channels:]
            modulation = torch.sigmoid(modulation.view(B, self.kernel_points, D, H, W))

        work_x = x.float()
        device = x.device
        z_base, y_base, x_base = torch.meshgrid(
            torch.arange(D, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        z_base = z_base.unsqueeze(0)
        y_base = y_base.unsqueeze(0)
        x_base = x_base.unsqueeze(0)

        kernel_z, kernel_y, kernel_x = torch.meshgrid(
            torch.arange(kd, device=device, dtype=torch.float32) - pd,
            torch.arange(kh, device=device, dtype=torch.float32) - ph,
            torch.arange(kw, device=device, dtype=torch.float32) - pw,
            indexing='ij',
        )
        kernel_z = kernel_z.reshape(-1)
        kernel_y = kernel_y.reshape(-1)
        kernel_x = kernel_x.reshape(-1)

        sampled = []
        denom_z = max(D - 1, 1)
        denom_y = max(H - 1, 1)
        denom_x = max(W - 1, 1)
        for k in range(self.kernel_points):
            if self.deformable_depth:
                off_z = offsets[:, k, 0]
                off_y = offsets[:, k, 1]
                off_x = offsets[:, k, 2]
            else:
                off_y = offsets[:, k, 0]
                off_x = offsets[:, k, 1]
                off_z = torch.zeros_like(off_y)

            z = z_base + kernel_z[k] + off_z
            y = y_base + kernel_y[k] + off_y
            x_coord = x_base + kernel_x[k] + off_x
            valid = (
                (z >= 0) & (z <= D - 1) &
                (y >= 0) & (y <= H - 1) &
                (x_coord >= 0) & (x_coord <= W - 1)
            ).unsqueeze(1).to(work_x.dtype)
            grid = torch.stack(
                (
                    2.0 * x_coord / denom_x - 1.0,
                    2.0 * y / denom_y - 1.0,
                    2.0 * z / denom_z - 1.0,
                ),
                dim=-1,
            )
            sample = F.grid_sample(
                work_x,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            )
            sample = sample * valid
            if modulation is not None:
                sample = sample * modulation[:, k].unsqueeze(1)
            sampled.append(sample)

        sampled = torch.stack(sampled, dim=2)  # [B, C, K, D, H, W]
        out = torch.einsum('ock,bckdhw->bodhw', self.weight.float(), sampled)
        if self.bias is not None:
            out = out + self.bias.float().view(1, -1, 1, 1, 1)
        return out.to(input_dtype)


class ResB3D(nn.Module):
    """Simple 3D residual block (spatial branch of SFB3D)."""

    def __init__(
        self,
        embed_dim: int,
        reduction: int = 1,
        use_deformable: bool = False,
        deformable_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        hidden_dim = embed_dim // reduction
        if use_deformable:
            deformable_kwargs = dict(deformable_kwargs or {})
            conv = lambda in_ch, out_ch: DeformableConv3d(  # noqa: E731
                in_ch,
                out_ch,
                **deformable_kwargs,
            )
        else:
            conv = lambda in_ch, out_ch: nn.Conv3d(  # noqa: E731
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
            )
        self.body = nn.Sequential(
            conv(embed_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv(hidden_dim, embed_dim),
        )
        self.scale = 1.0 / math.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x + self.body(x))


class SFB3D(nn.Module):
    """3D Spatial-Frequency Block: local conv branch + global FFT branch fused."""

    def __init__(
        self,
        embed_dim: int,
        reduction: int = 1,
        use_deformable: bool = False,
        deformable_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.S = ResB3D(
            embed_dim,
            reduction,
            use_deformable=use_deformable,
            deformable_kwargs=deformable_kwargs,
        )
        self.F = SpectralTransform3D(embed_dim)
        self.norm = LayerNorm3d(embed_dim * 2)
        self.fusion = nn.Conv3d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.S(x)
        f = self.F(x)
        out = torch.cat([s, f], dim=1)
        out = self.norm(out)
        out = self.fusion(out)
        return out


# =============================================================================
# FIR Residual Block (3D)
# =============================================================================

class FIRResBlock3D(nn.Module):
    """
    3D FIR Residual Block — volumetric counterpart of `FIRResBlock`.

    Flow:
        x → AdaGN3D + SiLU + Conv3D (radius cond.)
          → [attention blocks × num_swin_blocks] (Swin3D or Deformable3D)
          → [SFB3D × num_sfb_blocks]
          → residual: skip(x) + (1/√2) · h
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius_emb_dim: int,
        use_attention: bool = False,
        use_sfb: bool = True,
        num_heads: int = 8,
        window_size_3d: Tuple[int, int, int] = (2, 8, 8),
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        num_swin_blocks: int = 2,
        num_sfb_blocks: int = 1,
        gradient_checkpoint: bool = False,
        attention_type: str = 'window',
        deformable_kwargs: Optional[dict] = None,
        use_deformable_sfb: bool = False,
        deformable_sfb_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.use_sfb = use_sfb
        self.num_swin_blocks = num_swin_blocks
        self.num_sfb_blocks = num_sfb_blocks
        self.use_gradient_checkpoint = gradient_checkpoint
        self.attention_type = attention_type
        self.use_deformable_sfb = use_deformable_sfb

        # === Stage 1: AdaGN3D + Conv (radius FiLM conditioning) ===
        self.adagn = AdaGroupNorm3D(in_channels, radius_emb_dim)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        # === Stage 2: Attention blocks ===
        self.attn_blocks: Optional[nn.ModuleList]
        if use_attention and num_swin_blocks > 0:
            if attention_type == 'deformable':
                _dkw = dict(deformable_kwargs or {})
                self.attn_blocks = nn.ModuleList([
                    AdaLNZeroDeformableBlock3D(
                        dim=out_channels,
                        num_heads=num_heads,
                        r_emb_dim=radius_emb_dim,
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_path,
                        drop=dropout,
                        **_dkw,
                    )
                    for _ in range(num_swin_blocks)
                ])
            else:
                # Window attention: alternate W-MSA and SW-MSA
                shift_size_3d = (
                    window_size_3d[0] // 2,
                    window_size_3d[1] // 2,
                    window_size_3d[2] // 2,
                )
                self.attn_blocks = nn.ModuleList([
                    SwinTransformerBlock3D(
                        dim=out_channels,
                        input_resolution=(4, 256, 256),  # input_resolution unused at fwd
                        num_heads=num_heads,
                        window_size=window_size_3d,
                        shift_size=(0, 0, 0) if i % 2 == 0 else shift_size_3d,
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_path,
                    )
                    for i in range(num_swin_blocks)
                ])
        else:
            self.attn_blocks = None

        # === Stage 3: SFB3D blocks ===
        if use_sfb and num_sfb_blocks > 0:
            self.sfb_blocks = nn.ModuleList([
                SFB3D(
                    out_channels,
                    use_deformable=use_deformable_sfb,
                    deformable_kwargs=deformable_sfb_kwargs,
                )
                for _ in range(num_sfb_blocks)
            ])
        else:
            self.sfb_blocks = None

        # === Skip ===
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.residual_scale = 1.0 / math.sqrt(2)

        # Init
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)

    def forward(
        self,
        x: torch.Tensor,
        r_emb: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, D, H, W].
            r_emb: Radius embedding [B, radius_emb_dim].
            spatial_mask: Optional binary body mask [B, 1, D_orig, H_orig, W_orig]
                (downsampled internally to current feature resolution and only
                actually used by deformable attention).
        """
        B, C, D, H, W = x.shape

        identity = self.skip(x)

        h = self.adagn(x, r_emb)
        h = F.silu(h)
        h = self.conv1(h)

        # === Attention ===
        if self.attn_blocks is not None:
            # [B, C, D, H, W] -> [B, D*H*W, C]
            h_flat = h.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, h.shape[1])

            # Prepare mask at current feature resolution (only for deformable)
            mask_at_res = None
            if spatial_mask is not None and self.attention_type == 'deformable':
                mask_at_res = F.interpolate(
                    spatial_mask.float(), size=(D, H, W), mode='nearest'
                )  # [B, 1, D, H, W]

            for blk in self.attn_blocks:
                if self.attention_type == 'deformable':
                    if self.training and self.use_gradient_checkpoint:
                        if mask_at_res is not None:
                            h_flat = grad_checkpoint(
                                blk, h_flat, (D, H, W), r_emb, mask_at_res,
                                use_reentrant=False,
                            )
                        else:
                            h_flat = grad_checkpoint(
                                blk, h_flat, (D, H, W), r_emb,
                                use_reentrant=False,
                            )
                    else:
                        h_flat = blk(
                            h_flat, (D, H, W), r_emb, spatial_mask=mask_at_res
                        )
                else:
                    # Swin3D path: forward(x, x_size) — no radius/mask
                    if self.training and self.use_gradient_checkpoint:
                        h_flat = grad_checkpoint(
                            blk, h_flat, (D, H, W),
                            use_reentrant=False,
                        )
                    else:
                        h_flat = blk(h_flat, (D, H, W))

            h = h_flat.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)

        # === SFB3D ===
        if self.sfb_blocks is not None:
            for sfb in self.sfb_blocks:
                if self.training and self.use_gradient_checkpoint:
                    h = grad_checkpoint(sfb, h, use_reentrant=False)
                else:
                    h = sfb(h)

        return identity + self.residual_scale * h


# =============================================================================
# Downsample / Upsample (3D)
# =============================================================================

class Downsample3D(nn.Module):
    """3D downsampling via strided conv with configurable (D, H, W) stride."""

    def __init__(self, channels: int, stride: Tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        self.stride = tuple(stride)
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=self.stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling via transpose conv with configurable (D, H, W) stride."""

    def __init__(self, channels: int, stride: Tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        self.stride = tuple(stride)
        self.conv = nn.ConvTranspose3d(
            channels,
            channels,
            kernel_size=self.stride,
            stride=self.stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =============================================================================
# SwinFIRUNet25D
# =============================================================================

class SwinFIRUNet25D(nn.Module):
    """
    2.5D SwinFIR UNet with FIR residual blocks, 3D SFB, and optional 3D
    deformable attention. Volumetric mirror of `SwinFIRUNet`.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        num_slices: int = 8,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        attention_resolutions: Tuple[int, ...] = (2, 4),
        num_res_blocks: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        radius_emb_dim: int = 256,
        window_size: int = 8,  # kept for compatibility (unused at fwd)
        window_size_3d: Union[Tuple[int, int, int], dict] = (2, 8, 8),
        num_swin_blocks: int = 2,
        num_sfb_blocks: int = 1,
        num_bottleneck_blocks: int = 2,
        use_sfb: bool = True,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        gradient_checkpoint: bool = False,
        attention_type: str = 'window',
        deformable_kwargs: Optional[dict] = None,
        use_deformable_sfb: bool = False,
        deformable_sfb_kwargs: Optional[dict] = None,
        downsample_strides: Optional[List[Tuple[int, int, int]]] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_slices = num_slices
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.attention_resolutions = attention_resolutions
        self.num_res_blocks = num_res_blocks
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.window_size = window_size
        self.gradient_checkpoint = gradient_checkpoint
        self.attention_type = attention_type
        self.deformable_kwargs = deformable_kwargs or {}
        self.use_deformable_sfb = use_deformable_sfb
        self.deformable_sfb_kwargs = deformable_sfb_kwargs or {}
        # When in_channels==3, body mask is concatenated as 3rd input channel.
        self.use_mask_channel = (in_channels == 3)

        # Per-stage window sizes
        if isinstance(window_size_3d, (tuple, list)):
            self.window_size_3d = {i: tuple(window_size_3d) for i in range(len(channel_mults))}
        else:
            self.window_size_3d = window_size_3d

        if downsample_strides is None:
            downsample_strides = [(2, 2, 2)] * max(len(channel_mults) - 1, 0)
        downsample_strides = [
            tuple(s) if isinstance(s, (tuple, list)) else (s, s, s)
            for s in downsample_strides
        ]
        if len(downsample_strides) < len(channel_mults) - 1:
            raise ValueError(
                "downsample_strides must provide at least one stride per encoder "
                f"downsample ({len(channel_mults) - 1} needed, got {len(downsample_strides)})"
            )
        self.downsample_strides = downsample_strides[:max(len(channel_mults) - 1, 0)]

        # Stochastic depth schedule
        num_stages = len(channel_mults)
        total_blocks = num_stages * num_res_blocks * 2
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(total_blocks, 1))]

        # Radius embedding
        self.radius_emb = RadiusEmbedding(embed_dim=radius_emb_dim)

        # Input conv
        self.input_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)

        # === Encoder ===
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = base_channels
        block_idx = 0

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attention = (mult in attention_resolutions)
            stage_ws = self.window_size_3d.get(i, (2, 8, 8))

            stage_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                stage_blocks.append(
                    FIRResBlock3D(
                        in_channels=current_channels if j == 0 else out_ch,
                        out_channels=out_ch,
                        radius_emb_dim=radius_emb_dim,
                        use_attention=use_attention,
                        use_sfb=use_sfb,
                        num_heads=num_heads,
                        window_size_3d=stage_ws,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[block_idx],
                        dropout=dropout,
                        num_swin_blocks=num_swin_blocks,
                        num_sfb_blocks=num_sfb_blocks,
                        gradient_checkpoint=gradient_checkpoint,
                        attention_type=attention_type,
                        deformable_kwargs=self.deformable_kwargs,
                        use_deformable_sfb=self.use_deformable_sfb,
                        deformable_sfb_kwargs=self.deformable_sfb_kwargs,
                    )
                )
                current_channels = out_ch
                block_idx += 1
            self.encoder_stages.append(stage_blocks)

            if i < num_stages - 1:
                self.downsamples.append(
                    Downsample3D(current_channels, stride=self.downsample_strides[i])
                )

        # === Bottleneck ===
        bottleneck_ws = self.window_size_3d.get(num_stages - 1, (2, 8, 8))
        self.bottleneck = nn.ModuleList([
            FIRResBlock3D(
                in_channels=current_channels,
                out_channels=current_channels,
                radius_emb_dim=radius_emb_dim,
                use_attention=True,
                use_sfb=use_sfb,
                num_heads=num_heads,
                window_size_3d=bottleneck_ws,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rate,
                dropout=dropout,
                num_swin_blocks=num_swin_blocks,
                num_sfb_blocks=num_sfb_blocks,
                gradient_checkpoint=gradient_checkpoint,
                attention_type=attention_type,
                deformable_kwargs=self.deformable_kwargs,
                use_deformable_sfb=self.use_deformable_sfb,
                deformable_sfb_kwargs=self.deformable_sfb_kwargs,
            )
            for _ in range(num_bottleneck_blocks)
        ])

        # === Decoder ===
        self.decoder_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for idx, i in enumerate(reversed(range(num_stages))):
            mult = channel_mults[i]
            out_ch = base_channels * mult
            use_attention = (mult in attention_resolutions)
            stage_ws = self.window_size_3d.get(i, (2, 8, 8))

            if idx == 0:
                # First decoder stage: no upsample, no skip concat
                skip_in_ch = current_channels
            else:
                # Upsample from current_channels to out_ch via ConvTranspose
                upsample_stride = self.downsample_strides[-idx]
                self.upsamples.append(Upsample3D(current_channels, stride=upsample_stride))
                skip_in_ch = current_channels + out_ch  # cat with skip

            self.skip_convs.append(nn.Conv3d(skip_in_ch, out_ch, kernel_size=1))

            stage_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                stage_blocks.append(
                    FIRResBlock3D(
                        in_channels=out_ch,
                        out_channels=out_ch,
                        radius_emb_dim=radius_emb_dim,
                        use_attention=use_attention,
                        use_sfb=use_sfb,
                        num_heads=num_heads,
                        window_size_3d=stage_ws,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[block_idx] if block_idx < len(dpr) else drop_path_rate,
                        dropout=dropout,
                        num_swin_blocks=num_swin_blocks,
                        num_sfb_blocks=num_sfb_blocks,
                        gradient_checkpoint=gradient_checkpoint,
                        attention_type=attention_type,
                        deformable_kwargs=self.deformable_kwargs,
                        use_deformable_sfb=self.use_deformable_sfb,
                        deformable_sfb_kwargs=self.deformable_sfb_kwargs,
                    )
                )
                block_idx += 1
            self.decoder_stages.append(stage_blocks)
            current_channels = out_ch

        # === Output ===
        self.final_norm = nn.GroupNorm(get_num_groups(base_channels), base_channels)
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        mri: torch.Tensor,
        r: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy CT volume [B, 1, D, H, W].
            mri: MRI condition volume [B, 1, D, H, W].
            r: PFGM++ radius [B] or [B, 1].
            mask: Optional binary body mask [B, 1, D, H, W]. Required when
                  use_mask_channel=True (in_channels=3); also passed through
                  deformable attention to suppress background K/V.
        """
        x = x.float()
        mri = mri.float()
        r = r.float()
        if r.dim() > 1:
            r = r.squeeze(-1)

        # Concat noisy CT + MRI (+ optional mask channel)
        if self.use_mask_channel:
            mask_ch = mask.float() if mask is not None else torch.ones_like(x)
            x = torch.cat([x, mri, mask_ch], dim=1)  # 3 channels
        else:
            x = torch.cat([x, mri], dim=1)  # 2 channels

        # Body mask used for deformable attention masking
        mask_padded = mask.float().clamp(0, 1) if mask is not None else None

        r_emb = self.radius_emb(r)

        h = self.input_conv(x)

        # Encoder
        skips = []
        for i, stage in enumerate(self.encoder_stages):
            for block in stage:
                h = block(h, r_emb, spatial_mask=mask_padded)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        # Bottleneck
        for block in self.bottleneck:
            h = block(h, r_emb, spatial_mask=mask_padded)

        # Decoder
        num_stages = len(self.channel_mults)
        for idx, (stage, skip_conv) in enumerate(zip(self.decoder_stages, self.skip_convs)):
            if idx > 0:
                h = self.upsamples[idx - 1](h)
                skip = skips[-(idx + 1)]
                # Match dims if shapes differ (asymmetric downsampling)
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(
                        h, size=skip.shape[2:], mode='trilinear', align_corners=False
                    )
                h = torch.cat([h, skip], dim=1)
            h = skip_conv(h)
            for block in stage:
                h = block(h, r_emb, spatial_mask=mask_padded)

        h = self.final_norm(h)
        h = F.silu(h)
        return self.final_conv(h)


# =============================================================================
# Factory
# =============================================================================

def create_swin_fir_unet_25d(
    in_channels: int = 2,
    out_channels: int = 1,
    num_slices: int = 8,
    base_channels: int = 64,
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
    attention_resolutions: Tuple[int, ...] = (2, 4),
    num_res_blocks: int = 2,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    radius_emb_dim: int = 256,
    window_size: int = 8,
    window_size_3d: Union[Tuple[int, int, int], dict] = (2, 8, 8),
    num_swin_blocks: int = 2,
    num_sfb_blocks: int = 1,
    num_bottleneck_blocks: int = 2,
    use_sfb: bool = True,
    dropout: float = 0.0,
    drop_path_rate: float = 0.1,
    gradient_checkpoint: bool = False,
    attention_type: str = 'window',
    deformable_kwargs: Optional[dict] = None,
    use_deformable_sfb: bool = False,
    deformable_sfb_kwargs: Optional[dict] = None,
    downsample_strides: Optional[List[Tuple[int, int, int]]] = None,
) -> SwinFIRUNet25D:
    """Factory for 2.5D SwinFIR UNet — matches `create_swin_fir_unet` 2D API."""
    return SwinFIRUNet25D(
        in_channels=in_channels,
        out_channels=out_channels,
        num_slices=num_slices,
        base_channels=base_channels,
        channel_mults=channel_mults,
        attention_resolutions=attention_resolutions,
        num_res_blocks=num_res_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        radius_emb_dim=radius_emb_dim,
        window_size=window_size,
        window_size_3d=window_size_3d,
        num_swin_blocks=num_swin_blocks,
        num_sfb_blocks=num_sfb_blocks,
        num_bottleneck_blocks=num_bottleneck_blocks,
        use_sfb=use_sfb,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        gradient_checkpoint=gradient_checkpoint,
        attention_type=attention_type,
        deformable_kwargs=deformable_kwargs,
        use_deformable_sfb=use_deformable_sfb,
        deformable_sfb_kwargs=deformable_sfb_kwargs,
        downsample_strides=downsample_strides,
    )
