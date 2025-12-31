"""
3D Swin Transformer Attention for volumetric medical imaging.

Adapted from 2D Swin Transformer to work with 3D volumes (D, H, W).
Uses 3D window partitioning to enable attention across depth dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def window_partition_3d(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Partition 3D feature volume into non-overlapping 3D windows.
    
    Args:
        x: Input tensor of shape [B, D, H, W, C]
        window_size: Window size (Wd, Wh, Ww)
    
    Returns:
        windows: Tensor of shape [num_windows*B, Wd*Wh*Ww, C]
    """
    B, D, H, W, C = x.shape
    Wd, Wh, Ww = window_size
    
    # Reshape into windows
    x = x.view(B, D // Wd, Wd, H // Wh, Wh, W // Ww, Ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, Wd * Wh * Ww, C)
    
    return windows


def window_reverse_3d(windows: torch.Tensor, window_size: Tuple[int, int, int], D: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse 3D window partition to reconstruct feature volume.
    
    Args:
        windows: Tensor of shape [num_windows*B, Wd*Wh*Ww, C]
        window_size: Window size (Wd, Wh, Ww)
        D: Depth of original feature volume
        H: Height of original feature volume
        W: Width of original feature volume
    
    Returns:
        x: Tensor of shape [B, D, H, W, C]
    """
    Wd, Wh, Ww = window_size
    B = int(windows.shape[0] / (D * H * W / Wd / Wh / Ww))
    
    x = windows.view(B, D // Wd, H // Wh, W // Ww, Wd, Wh, Ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    
    return x


class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-head Self-Attention with relative position bias.
    
    Extends 2D Swin attention to 3D volumes, allowing attention across
    depth, height, and width dimensions within local 3D windows.
    
    Args:
        dim: Number of input channels
        window_size: 3D window size (Wd, Wh, Ww)
        num_heads: Number of attention heads
        qkv_bias: If True, add learnable bias to Q, K, V projections
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wd, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias table
        # Size: (2*Wd-1) * (2*Wh-1) * (2*Ww-1), num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )
        
        # Compute relative position index for each token in the 3D window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize with standard weights (not zero - let it learn)
        nn.init.xavier_uniform_(self.qkv.weight)
        if qkv_bias:
            nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
        # Initialize relative position bias with small values
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [num_windows*B, N, C] where N = Wd*Wh*Ww
            mask: Attention mask of shape [num_windows, N, N] or None
        
        Returns:
            Output tensor of shape [num_windows*B, N, C]
        """
        B_, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B_, num_heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B_, num_heads, N, N
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1
        )  # Wd*Wh*Ww, Wd*Wh*Ww, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, N, N
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply mask (for shifted window attention)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class Mlp(nn.Module):
    """MLP as used in Swin Transformer."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        # Standard initialization (not zero - let residual path learn)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) regularization."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class SwinTransformerBlock3D(nn.Module):
    """
    3D Swin Transformer Block.
    
    Extends 2D Swin block to 3D volumes with 3D window partitioning.
    
    Args:
        dim: Number of input channels
        input_resolution: Input resolution (D, H, W)
        num_heads: Number of attention heads
        window_size: 3D window size (default: 2x8x8 for depth=5 volumes)
        shift_size: Shift size for SW-MSA. 0 = W-MSA, >0 = SW-MSA
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        qkv_bias: If True, add learnable bias to Q, K, V
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
    """
    
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int, int],
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 8, 8),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Adjust window size if input is smaller
        window_size_list = list(window_size)
        shift_size_list = list(shift_size)
        for i in range(3):
            if input_resolution[i] <= window_size[i]:
                shift_size_list[i] = 0
                window_size_list[i] = input_resolution[i]
        self.window_size = tuple(window_size_list)
        self.shift_size = tuple(shift_size_list)
        
        # Verify shift_size validity
        for i in range(3):
            assert 0 <= self.shift_size[i] < self.window_size[i], \
                f"shift_size[{i}] must be in [0, window_size[{i}])"
        
        # Layer norms
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # 3D Window attention
        self.attn = WindowAttention3D(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
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
        
        # Compute attention mask for shifted windows (if needed)
        if any(s > 0 for s in self.shift_size):
            attn_mask = self._compute_attn_mask_3d(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
    
    def _compute_attn_mask_3d(self, input_resolution: Tuple[int, int, int]) -> torch.Tensor:
        """
        Compute 3D attention mask for SW-MSA.
        
        Creates a mask that prevents attention between different regions
        when 3D windows are cyclically shifted.
        """
        D, H, W = input_resolution
        
        # Ensure dimensions are divisible by window size
        # If not, we'll crop during forward pass
        Wd, Wh, Ww = self.window_size
        D_working = (D // Wd) * Wd
        H_working = (H // Wh) * Wh
        W_working = (W // Ww) * Ww
        
        if D_working == 0 or H_working == 0 or W_working == 0:
            # Window size too large, skip masking
            return None
        
        img_mask = torch.zeros((1, D_working, H_working, W_working, 1))
        
        d_slices = (
            slice(0, -self.window_size[0]),
            slice(-self.window_size[0], -self.shift_size[0]) if self.shift_size[0] > 0 else slice(-self.window_size[0], None),
            slice(-self.shift_size[0], None) if self.shift_size[0] > 0 else slice(0, 0),
        )
        h_slices = (
            slice(0, -self.window_size[1]),
            slice(-self.window_size[1], -self.shift_size[1]) if self.shift_size[1] > 0 else slice(-self.window_size[1], None),
            slice(-self.shift_size[1], None) if self.shift_size[1] > 0 else slice(0, 0),
        )
        w_slices = (
            slice(0, -self.window_size[2]),
            slice(-self.window_size[2], -self.shift_size[2]) if self.shift_size[2] > 0 else slice(-self.window_size[2], None),
            slice(-self.shift_size[2], None) if self.shift_size[2] > 0 else slice(0, 0),
        )
        
        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    if d.start != d.stop and h.start != h.stop and w.start != w.stop:
                        img_mask[:, d, h, w, :] = cnt
                        cnt += 1
        
        mask_windows = window_partition_3d(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # Use -1e4 instead of -100 for better numerical stability with softmax
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1e4))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask
    
    def forward(self, x: torch.Tensor, x_size: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, L, C] where L = D*H*W
            x_size: Optional (D, H, W) tuple. If None, uses input_resolution.
        
        Returns:
            Output tensor of shape [B, L, C]
        """
        D, H, W = x_size if x_size is not None else self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, f"Input size mismatch: {L} != {D}*{H}*{W}"
        
        # Check if dimensions are divisible by window size
        Wd, Wh, Ww = self.window_size
        if D % Wd != 0 or H % Wh != 0 or W % Ww != 0:
            # Dimensions not divisible - crop to nearest multiple
            D_working = (D // Wd) * Wd
            H_working = (H // Wh) * Wh
            W_working = (W // Ww) * Ww
            
            if D_working == 0 or H_working == 0 or W_working == 0:
                # Window too large, skip attention
                return x
            
            # Extract working region
            x_view = x.view(B, D, H, W, C)
            x_crop = x_view[:, :D_working, :H_working, :W_working, :].contiguous()
            x_crop_flat = x_crop.view(B, D_working * H_working * W_working, C)
            
            # Process cropped region
            shortcut_crop = x_crop_flat
            x_proc = self.norm1(x_crop_flat)
            x_proc = x_proc.view(B, D_working, H_working, W_working, C)
        else:
            D_working, H_working, W_working = D, H, W
            shortcut = x
            x_proc = self.norm1(x)
            x_proc = x_proc.view(B, D, H, W, C)
        
        # Cyclic shift for SW-MSA
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x_proc, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x_proc
        
        # Partition into 3D windows
        x_windows = window_partition_3d(shifted_x, self.window_size)
        
        # Window attention
        if (D_working, H_working, W_working) == self.input_resolution:
            attn_mask = self.attn_mask
        else:
            # Recompute mask for different resolution
            attn_mask = self._compute_attn_mask_3d((D_working, H_working, W_working))
            if attn_mask is not None:
                attn_mask = attn_mask.to(x.device)
        
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        shifted_x = window_reverse_3d(attn_windows, self.window_size, D_working, H_working, W_working)
        
        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x_proc = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x_proc = shifted_x
        
        x_proc = x_proc.view(B, D_working * H_working * W_working, C)
        
        # Residual connection + MLP
        if D_working != D or H_working != H or W_working != W:
            # We cropped - only process cropped region
            x_proc = shortcut_crop + self.drop_path(x_proc)
            x_proc = x_proc + self.drop_path(self.mlp(self.norm2(x_proc)))
            
            # Paste back into original size
            x_view = x.view(B, D, H, W, C).clone()
            x_proc_view = x_proc.view(B, D_working, H_working, W_working, C)
            x_view[:, :D_working, :H_working, :W_working, :] = x_proc_view
            return x_view.view(B, D * H * W, C)
        else:
            x_proc = shortcut + self.drop_path(x_proc)
            x_proc = x_proc + self.drop_path(self.mlp(self.norm2(x_proc)))
            return x_proc
