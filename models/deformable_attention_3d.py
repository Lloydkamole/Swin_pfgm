"""
3D Deformable Attention++ with QK-Norm for PFGM-FIR 2.5D.

Volumetric extension of `deformable_attention.py`:
- 5D input tensors [B, C, D, H, W] (flattened to [B, D*H*W, C] for attention)
- 3D offset prediction (dx, dy, dz)
- 3D reference grid + `F.grid_sample` on 5D volumes
- Optional 3D KV downsampling

Drop-in replacement for SwinTransformerBlock3D in `network_25d.py` when
`attention_type='deformable'`.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_attention import Mlp, DropPath


class LayerNormProxy3D(nn.Module):
    """LayerNorm for [B, C, D, H, W] tensors (channels-first)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, D, H, W] -> [B, D, H, W, C] -> LN -> [B, C, D, H, W]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


def _to_3tuple(v: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    return tuple(v)  # type: ignore[return-value]


class DeformableAttention3D(nn.Module):
    """
    3D Deformable Attention++ with QK-Norm.

    Volumetric analog of `DeformableAttention`. Each head group learns a 3D
    spatial offset field (dx, dy, dz) used to deform the K/V sampling grid
    inside `F.grid_sample` (5D mode).

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        n_groups: Number of offset groups.
        qkv_bias: Bias for Q, K, V projections.
        attn_drop: Dropout for attention weights (passed to SDPA).
        proj_drop: Dropout for output projection.
        offset_range_factor: Max offset magnitude in units of grid spacing.
        use_qk_norm: Apply RMSNorm to Q and K.
        offset_kernel_size: Kernel for offset prediction conv. Can be int or
            (kD, kH, kW) tuple. Defaults to (3, 5, 5) when int=5, keeping a
            smaller kernel along the (typically very small) depth axis.
        kv_stride: Downsample factor for K/V resolution. Int or
            (sD, sH, sW). When int, only the (H, W) dims are strided so the
            small depth dim is preserved.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        n_groups: Optional[int] = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        offset_range_factor: float = 2.0,
        use_qk_norm: bool = True,
        offset_kernel_size: Union[int, Tuple[int, int, int]] = 5,
        kv_stride: Union[int, Tuple[int, int, int]] = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_groups = n_groups if n_groups is not None else max(num_heads // 2, 1)
        self.head_dim = dim // num_heads
        self.offset_range_factor = offset_range_factor
        self.use_qk_norm = use_qk_norm

        # Normalize stride: int → (1, k, k) (don't shrink the small depth axis)
        if isinstance(kv_stride, int):
            self.kv_stride = (1, kv_stride, kv_stride)
        else:
            self.kv_stride = tuple(kv_stride)  # type: ignore[assignment]

        # Normalize offset kernel: int=5 → (3, 5, 5)
        if isinstance(offset_kernel_size, int):
            kD = min(3, offset_kernel_size) if offset_kernel_size >= 3 else offset_kernel_size
            self.offset_kernel = (kD, offset_kernel_size, offset_kernel_size)
        else:
            self.offset_kernel = tuple(offset_kernel_size)  # type: ignore[assignment]

        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        assert num_heads % self.n_groups == 0, (
            f"num_heads {num_heads} not divisible by n_groups {self.n_groups}"
        )
        self.n_group_channels = dim // self.n_groups

        # Q, K, V projections (separate; K/V applied after spatial sampling)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_out = nn.Linear(dim, dim)

        # Offset prediction: depthwise 3D conv → pointwise to 3 channels (dx, dy, dz)
        pad = tuple(k // 2 for k in self.offset_kernel)
        self.conv_offset = nn.Sequential(
            nn.Conv3d(
                self.n_group_channels, self.n_group_channels,
                kernel_size=self.offset_kernel, padding=pad,
                groups=self.n_group_channels, bias=False,  # depthwise
            ),
            LayerNormProxy3D(self.n_group_channels),
            nn.GELU(),
            nn.Conv3d(self.n_group_channels, 3, kernel_size=1, bias=True),
        )
        # Zero-init final layer → offsets start at 0 → regular grid initially
        nn.init.zeros_(self.conv_offset[-1].weight)
        nn.init.zeros_(self.conv_offset[-1].bias)

        # KV downsampling
        if any(s > 1 for s in self.kv_stride):
            self.kv_downsample = nn.AvgPool3d(
                kernel_size=self.kv_stride, stride=self.kv_stride,
            )
        else:
            self.kv_downsample = None

        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    @torch.no_grad()
    def _get_reference_grid(
        self, D: int, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        """Normalized 3D reference grid in [-1, 1] with last dim (x, y, z)."""
        grid_z = torch.linspace(-1, 1, D, device=device)
        grid_y = torch.linspace(-1, 1, H, device=device)
        grid_x = torch.linspace(-1, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij')
        # F.grid_sample (5D) expects last-dim order (x, y, z)
        grid = torch.stack([xx, yy, zz], dim=-1)  # [D, H, W, 3]
        return grid

    def forward(
        self,
        x: torch.Tensor,
        D: int,
        H: int,
        W: int,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, C] where L = D * H * W.
            D, H, W: Volumetric dimensions.
            spatial_mask: Optional binary body mask [B, 1, D_m, H_m, W_m].
                When provided, background K/V features are zeroed and the
                attention bias masks out sampled-background positions.

        Returns:
            Output tensor [B, L, C].
        """
        B, L, C = x.shape
        assert L == D * H * W, f"Input length {L} != D*H*W = {D}*{H}*{W}"

        # [B, L, C] -> [B, C, D, H, W]
        x_spatial = x.transpose(1, 2).reshape(B, C, D, H, W)

        # === Mask the K/V *source* features ===
        if spatial_mask is not None:
            mask_feat = F.interpolate(
                spatial_mask.float(), size=(D, H, W), mode='nearest'
            )  # [B, 1, D, H, W]
            x_spatial = x_spatial * mask_feat

        # === Query at full resolution ===
        q = self.proj_q(x)  # [B, L, C]
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # q: [B, num_heads, L, head_dim]

        # === KV (optionally downsampled spatially) ===
        if self.kv_downsample is not None:
            x_kv = self.kv_downsample(x_spatial)  # [B, C, Dk, Hk, Wk]
        else:
            x_kv = x_spatial

        _, _, Dk, Hk, Wk = x_kv.shape

        # === Per-group offsets ===
        x_groups = x_kv.reshape(
            B * self.n_groups, self.n_group_channels, Dk, Hk, Wk
        )
        raw_offsets = self.conv_offset(x_groups)  # [B*n_groups, 3, Dk, Hk, Wk]

        # Grid spacing in normalized [-1, 1] coords: 2/(res-1)
        gs_z = 2.0 / max(Dk - 1, 1)
        gs_y = 2.0 / max(Hk - 1, 1)
        gs_x = 2.0 / max(Wk - 1, 1)

        offsets = torch.tanh(raw_offsets)  # values in [-1, 1]
        off_x = offsets[:, 0:1] * (self.offset_range_factor * gs_x)
        off_y = offsets[:, 1:2] * (self.offset_range_factor * gs_y)
        off_z = offsets[:, 2:3] * (self.offset_range_factor * gs_z)
        offsets = torch.cat([off_x, off_y, off_z], dim=1)  # [B*ng, 3, Dk, Hk, Wk]

        # === Build deformed grid ===
        ref_grid = self._get_reference_grid(Dk, Hk, Wk, x.device)  # [Dk, Hk, Wk, 3]
        # offsets: [B*ng, 3, Dk, Hk, Wk] -> [B*ng, Dk, Hk, Wk, 3]
        offsets = offsets.permute(0, 2, 3, 4, 1)
        deformed_grid = ref_grid.unsqueeze(0) + offsets  # [B*ng, Dk, Hk, Wk, 3]
        deformed_grid = deformed_grid.clamp(-1, 1)

        # === Sample K/V at full resolution ===
        x_full_groups = x_spatial.reshape(
            B * self.n_groups, self.n_group_channels, D, H, W
        )
        sampled = F.grid_sample(
            x_full_groups, deformed_grid,
            mode='bilinear', padding_mode='zeros', align_corners=True,
        )  # [B*ng, Cg, Dk, Hk, Wk]

        # === Sample mask through deformed grids (phantom detection) ===
        sampled_mask = None
        if spatial_mask is not None:
            mask_full = F.interpolate(
                spatial_mask.float(), size=(D, H, W), mode='nearest'
            )  # [B, 1, D, H, W]
            mask_groups = mask_full.unsqueeze(1).expand(
                -1, self.n_groups, -1, -1, -1, -1
            ).reshape(B * self.n_groups, 1, D, H, W)
            sampled_mask = F.grid_sample(
                mask_groups, deformed_grid,
                mode='nearest', padding_mode='zeros', align_corners=True,
            )  # [B*ng, 1, Dk, Hk, Wk]
            sampled_mask = sampled_mask.reshape(
                B, self.n_groups, Dk, Hk, Wk
            ).min(dim=1).values  # [B, Dk, Hk, Wk]

        # Reshape: [B*ng, Cg, Dk, Hk, Wk] -> [B, C, Dk*Hk*Wk] -> [B, Lkv, C]
        L_kv = Dk * Hk * Wk
        sampled = sampled.reshape(B, C, Dk, Hk, Wk).flatten(2).transpose(1, 2)

        k = self.proj_k(sampled)
        v = self.proj_v(sampled)
        k = k.reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # === Attention bias for sampled-background suppression ===
        if sampled_mask is not None:
            kv_mask = sampled_mask.reshape(B, 1, 1, L_kv)
            attn_bias = torch.zeros(
                B, 1, 1, L_kv, device=x.device, dtype=q.dtype,
            )
            attn_bias.masked_fill_(kv_mask < 0.5, float('-inf'))
        else:
            attn_bias = None

        drop_p = self.attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop_p, attn_mask=attn_bias,
        )  # [B, num_heads, L, head_dim]

        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.proj_out(out)
        out = self.proj_drop(out)
        return out


class AdaLNZeroDeformableBlock3D(nn.Module):
    """
    3D Deformable attention block with AdaLN-Zero conditioning.

    Volumetric analog of `AdaLNZeroDeformableBlock`. Same call pattern as the
    Swin 3D blocks plus an `r_emb` and optional `spatial_mask`:

        forward(x, x_size, r_emb, spatial_mask=None)

    where `x_size` is `(D, H, W)`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        r_emb_dim: int,
        n_groups: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        offset_range_factor: float = 2.0,
        use_qk_norm: bool = True,
        offset_kernel_size: Union[int, Tuple[int, int, int]] = 5,
        kv_stride: Union[int, Tuple[int, int, int]] = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # AdaLN-Zero: r_emb → 6 modulation vectors
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(r_emb_dim, 6 * dim),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        self.attn = DeformableAttention3D(
            dim=dim,
            num_heads=num_heads,
            n_groups=n_groups,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            offset_range_factor=offset_range_factor,
            use_qk_norm=use_qk_norm,
            offset_kernel_size=offset_kernel_size,
            kv_stride=kv_stride,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int, int],
        r_emb: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        D, H, W = x_size
        B, L, C = x.shape
        assert L == D * H * W, f"Input size mismatch: {L} != {D}*{H}*{W}"

        modulation = self.adaLN_modulation(r_emb)  # [B, 6*C]
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(6, dim=-1)

        # Attention branch
        shortcut = x
        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h, D, H, W, spatial_mask=spatial_mask)
        x = shortcut + gate_msa.unsqueeze(1) * self.drop_path(h)

        # MLP branch
        x = x + gate_mlp.unsqueeze(1) * self.drop_path(
            self.mlp(self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1))
        )
        return x
