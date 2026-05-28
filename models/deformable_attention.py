"""
Deformable Attention++ with QK-Norm for PFGM-FIR.

Drop-in replacement for WindowAttention / AdaLNZeroSwinTransformerBlock.
Based on DAT++ (Xia et al., CVPR 2022 / NeurIPS 2024) with QK-Norm from SD3.

Key differences from WindowAttention:
- Data-dependent K/V sampling via learned spatial offsets (no fixed windows)
- Per-group offset learning → natural head diversity (each group attends differently)
- QK-Norm (RMSNorm on Q, K) prevents attention entropy collapse
- Uses F.scaled_dot_product_attention for flash attention support

Usage:
    # Controlled via attention_type flag in SwinFIRUNet / FIRResBlock:
    model = create_swin_fir_unet(config='base', attention_type='deformable')
    # Revert to original:
    model = create_swin_fir_unet(config='base', attention_type='window')
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_attention import Mlp, DropPath


class LayerNormProxy(nn.Module):
    """LayerNorm for [B, C, H, W] tensors (channels-first)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, H, W, C] -> LN -> [B, C, H, W]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DeformableAttention(nn.Module):
    """
    Deformable Attention++ with QK-Norm.

    Instead of attending within fixed windows, each head group learns data-dependent
    spatial offsets that shift where keys and values are sampled from. This gives:
    1. Content-aware attention (focuses on relevant anatomy, not fixed grids)
    2. Natural head diversity (different groups learn different offset patterns)

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        n_groups: Number of offset groups (heads split into groups, each with
            independent offsets). Default: num_heads // 2.
        qkv_bias: If True, add bias to Q, K, V projections.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
        offset_range_factor: Controls maximum offset magnitude relative to
            grid spacing. Higher = larger deformations.
        use_qk_norm: Apply RMSNorm to Q and K (prevents entropy collapse).
        offset_kernel_size: Kernel size for the offset prediction convolution.
        kv_stride: Downsample factor for K/V resolution. Use > 1 for
            high-resolution stages (e.g., 128x128) to reduce memory.
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
        offset_kernel_size: int = 5,
        kv_stride: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_groups = n_groups if n_groups is not None else max(num_heads // 2, 1)
        self.head_dim = dim // num_heads
        self.offset_range_factor = offset_range_factor
        self.use_qk_norm = use_qk_norm
        self.kv_stride = kv_stride

        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        assert num_heads % self.n_groups == 0, (
            f"num_heads {num_heads} not divisible by n_groups {self.n_groups}"
        )

        self.n_group_channels = dim // self.n_groups

        # Separate Q, K, V projections (not fused, since K/V may be at different resolution)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_out = nn.Linear(dim, dim)

        # Offset prediction network (per-group via batch reshape)
        # Input: [B*n_groups, n_group_channels, Hk, Wk]
        # Output: [B*n_groups, 2, Hk, Wk] (x, y offsets)
        pad = offset_kernel_size // 2
        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.n_group_channels, self.n_group_channels,
                kernel_size=offset_kernel_size, padding=pad,
                groups=self.n_group_channels, bias=False,  # depthwise
            ),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, kernel_size=1, bias=True),
        )
        # Zero-init: offsets start at 0 → regular grid at initialization
        nn.init.zeros_(self.conv_offset[-1].weight)
        nn.init.zeros_(self.conv_offset[-1].bias)

        # KV downsampling for high-resolution stages
        if kv_stride > 1:
            self.kv_downsample = nn.AvgPool2d(kernel_size=kv_stride, stride=kv_stride)
        else:
            self.kv_downsample = None

        # QK-Norm (SD3/Flux-style)
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        self.attn_drop = attn_drop  # passed to SDPA
        self.proj_drop = nn.Dropout(proj_drop)

    @torch.no_grad()
    def _get_reference_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create a normalized reference grid in [-1, 1] for grid_sample."""
        # grid_y, grid_x: [H, W] each, values in [-1, 1]
        grid_y = torch.linspace(-1, 1, H, device=device)
        grid_x = torch.linspace(-1, 1, W, device=device)
        grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing='ij'), dim=-1)  # [H, W, 2] (y, x)
        # grid_sample expects (x, y) order
        grid = grid.flip(-1)  # [H, W, 2] now (x, y)
        return grid

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, C] where L = H * W.
            H: Spatial height.
            W: Spatial width.
            spatial_mask: Optional binary body mask [B, 1, H_mask, W_mask].
                When provided, background K/V features are zeroed out so that
                deformable attention cannot attend to background regions.

        Returns:
            Output tensor [B, L, C].
        """
        B, L, C = x.shape
        assert L == H * W, f"Input length {L} != H*W = {H}*{W}"

        # Reshape to spatial for offset computation
        x_spatial = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]

        # === Attention masking: zero out background K/V source features ===
        # Q remains unmasked so queries retain full spatial context; only the
        # key/value sampling pool is restricted to body regions.
        if spatial_mask is not None:
            mask_feat = F.interpolate(
                spatial_mask.float(), size=(H, W), mode='nearest'
            )  # [B, 1, H, W]
            x_spatial = x_spatial * mask_feat

        # === Query: always at full resolution ===
        q = self.proj_q(x)  # [B, L, C]
        q = q.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        # q: [B, num_heads, H*W, head_dim]

        # === KV: optionally at reduced resolution ===
        if self.kv_downsample is not None:
            x_kv = self.kv_downsample(x_spatial)  # [B, C, Hk, Wk]
        else:
            x_kv = x_spatial  # [B, C, H, W]

        _, _, Hk, Wk = x_kv.shape

        # === Compute per-group offsets ===
        # Reshape: [B, C, Hk, Wk] -> [B*n_groups, n_group_channels, Hk, Wk]
        x_groups = x_kv.reshape(B * self.n_groups, self.n_group_channels, Hk, Wk)
        raw_offsets = self.conv_offset(x_groups)  # [B*n_groups, 2, Hk, Wk]

        # Scale offsets: tanh clamps to [-1, 1], then scale by range factor * grid spacing
        # Grid spacing is 2/(res-1) for a [-1,1] grid
        grid_spacing_y = 2.0 / max(Hk - 1, 1)
        grid_spacing_x = 2.0 / max(Wk - 1, 1)
        offsets = torch.tanh(raw_offsets)  # [B*n_groups, 2, Hk, Wk], values in [-1, 1]
        # Scale x and y offsets separately
        offsets_x = offsets[:, 0:1] * (self.offset_range_factor * grid_spacing_x)
        offsets_y = offsets[:, 1:2] * (self.offset_range_factor * grid_spacing_y)
        offsets = torch.cat([offsets_x, offsets_y], dim=1)  # [B*n_groups, 2, Hk, Wk]

        # === Build deformed sampling grid ===
        ref_grid = self._get_reference_grid(Hk, Wk, x.device)  # [Hk, Wk, 2] (x, y)
        # offsets: [B*n_groups, 2, Hk, Wk] -> [B*n_groups, Hk, Wk, 2] (x, y)
        offsets = offsets.permute(0, 2, 3, 1)  # [B*n_groups, Hk, Wk, 2]
        deformed_grid = ref_grid.unsqueeze(0) + offsets  # [B*n_groups, Hk, Wk, 2]
        deformed_grid = deformed_grid.clamp(-1, 1)

        # === Sample K/V features from full-res map at deformed positions ===
        # We sample from the full-resolution feature map so offsets can reach beyond
        # the KV grid when kv_stride > 1
        x_full_groups = x_spatial.reshape(
            B * self.n_groups, self.n_group_channels, H, W
        )  # [B*n_groups, Cg, H, W]

        sampled = F.grid_sample(
            x_full_groups, deformed_grid,
            mode='bilinear', padding_mode='zeros', align_corners=True,
        )  # [B*n_groups, Cg, Hk, Wk]

        # === Sample mask through deformed grids to detect phantom positions ===
        # A "phantom" K position is a foreground grid position whose deformed
        # offsets land on background pixels.  grid_sample returns zeros there,
        # proj_k produces its bias vector, and k_norm normalises it to unit
        # scale — creating an attractive phantom K that steals attention from
        # genuine content.  Masking based on the *sampled* mask (not the grid
        # mask) catches these phantoms.
        if spatial_mask is not None:
            mask_full = F.interpolate(
                spatial_mask.float(), size=(H, W), mode='nearest'
            )  # [B, 1, H, W]
            # Expand to match n_groups: [B, 1, H, W] -> [B*n_groups, 1, H, W]
            mask_groups = mask_full.unsqueeze(1).expand(
                -1, self.n_groups, -1, -1, -1
            ).reshape(B * self.n_groups, 1, H, W)
            sampled_mask = F.grid_sample(
                mask_groups, deformed_grid,
                mode='nearest', padding_mode='zeros', align_corners=True,
            )  # [B*n_groups, 1, Hk, Wk]
            # Conservative: mask if ANY group's sample lands on background
            sampled_mask = sampled_mask.reshape(
                B, self.n_groups, Hk, Wk
            ).min(dim=1).values  # [B, Hk, Wk]

        # Reshape back: [B, C, Hk, Wk] -> [B, Hk*Wk, C]
        sampled = sampled.reshape(B, C, Hk, Wk).flatten(2).transpose(1, 2)

        # Project K, V from sampled features
        k = self.proj_k(sampled)  # [B, Hk*Wk, C]
        v = self.proj_v(sampled)  # [B, Hk*Wk, C]

        k = k.reshape(B, Hk * Wk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Hk * Wk, self.num_heads, self.head_dim).transpose(1, 2)
        # k, v: [B, num_heads, Hk*Wk, head_dim]

        # === QK-Norm ===
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # === Attention KV mask (prevent attending to background) ===
        # Uses the *sampled* mask: catches both true-background grid positions
        # AND phantom foreground positions whose deformed offsets sample from
        # background regions.  This prevents the proj_k bias / k_norm phantom
        # from stealing attention from genuine content.
        if spatial_mask is not None:
            kv_mask = sampled_mask.reshape(B, 1, 1, Hk * Wk)  # [B, 1, 1, L_kv]
            attn_bias = torch.zeros(
                B, 1, 1, Hk * Wk, device=x.device, dtype=q.dtype,
            )
            attn_bias.masked_fill_(kv_mask < 0.5, float('-inf'))
        else:
            attn_bias = None

        # === Attention via SDPA (flash attention when available) ===
        drop_p = self.attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop_p, attn_mask=attn_bias,
        )  # [B, num_heads, H*W, head_dim]

        # === Output projection ===
        out = out.transpose(1, 2).reshape(B, H * W, C)  # [B, L, C]
        out = self.proj_out(out)
        out = self.proj_drop(out)

        return out


class AdaLNZeroDeformableBlock(nn.Module):
    """
    Deformable Attention block with AdaLN-Zero conditioning (DiT-style).

    Drop-in replacement for AdaLNZeroSwinTransformerBlock. Same forward signature:
        forward(x, x_size, r_emb)

    Key differences:
    - Uses DeformableAttention instead of WindowAttention
    - No window partitioning or cyclic shifting (offsets handle spatial diversity)
    - QK-Norm prevents entropy collapse for sharper, content-focused attention

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        r_emb_dim: Radius embedding dimension for AdaLN-Zero conditioning.
        n_groups: Number of offset groups for head diversity.
        mlp_ratio: MLP hidden dim expansion ratio.
        qkv_bias: Add bias to Q, K, V projections.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
        offset_range_factor: Max offset magnitude relative to grid spacing.
        use_qk_norm: Apply RMSNorm to Q and K.
        offset_kernel_size: Kernel for offset prediction conv.
        kv_stride: Downsample factor for K/V resolution.
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
        offset_kernel_size: int = 5,
        kv_stride: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Layer norms WITHOUT learnable affine (affine comes from AdaLN conditioning)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # AdaLN-Zero modulation: projects r_emb to 6 vectors
        # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(r_emb_dim, 6 * dim),
        )
        # Zero-init: blocks start as identity
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # Deformable attention (replaces WindowAttention)
        self.attn = DeformableAttention(
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

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_size: Optional[Tuple[int, int]],
        r_emb: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with AdaLN-Zero conditioning.

        Same signature as AdaLNZeroSwinTransformerBlock for drop-in compatibility.

        Args:
            x: Input tensor [B, L, C] where L = H*W.
            x_size: (H, W) spatial dimensions.
            r_emb: Conditioning embedding [B, r_emb_dim].
            spatial_mask: Optional binary body mask [B, 1, H, W] at current
                feature resolution. Passed through to DeformableAttention.

        Returns:
            Output tensor [B, L, C].
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, f"Input size mismatch: {L} != {H}*{W}"

        # Compute all 6 modulation parameters at once
        modulation = self.adaLN_modulation(r_emb)  # [B, 6*dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        # === Attention branch with AdaLN-Zero ===
        shortcut = x
        x = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        # Deformable attention (no window partition needed)
        x = self.attn(x, H, W, spatial_mask=spatial_mask)

        # Gated residual (gate_msa starts at 0 → identity at init)
        x = shortcut + gate_msa.unsqueeze(1) * self.drop_path(x)

        # === MLP branch with AdaLN-Zero ===
        x = x + gate_mlp.unsqueeze(1) * self.drop_path(
            self.mlp(self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1))
        )

        return x
