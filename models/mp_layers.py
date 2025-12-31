"""
Magnitude-Preserving Layers for EDM2-style architectures.

Implements the core principles from "Analyzing and Improving the Training Dynamics of Diffusion Models":
- Weight normalization to prevent weight growth
- Magnitude-preserving activations
- Magnitude-preserving operations (sum, concatenation)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x, dim=None, eps=1e-4):
    """
    Normalize tensor to unit magnitude with respect to given dimensions.
    Default = all dimensions except the first (batch dimension).
    
    Args:
        x: Input tensor
        dim: Dimensions to normalize over. None = all except batch
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def mp_silu(x):
    """
    Magnitude-preserving SiLU activation (Equation 81 from EDM2).
    
    Standard SiLU reduces magnitude by factor of ~0.596 on average.
    This version compensates for that reduction.
    """
    return F.silu(x) / 0.596


def mp_sum(a, b, t=0.5):
    """
    Magnitude-preserving sum (Equation 88 from EDM2).
    
    Combines two tensors while preserving expected magnitude.
    
    Args:
        a, b: Input tensors to combine
        t: Balance between a (0) and b (1). Default 0.5 = equal weight
    
    Returns:
        Combined tensor with preserved magnitude
    """
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)


def mp_cat(a, b, dim=1, t=0.5):
    """
    Magnitude-preserving concatenation (Equation 103 from EDM2).
    
    Concatenates two tensors while preserving expected magnitude.
    
    Args:
        a, b: Input tensors to concatenate
        dim: Dimension along which to concatenate
        t: Balance between a (0) and b (1). Default 0.5 = equal weight
    
    Returns:
        Concatenated tensor with preserved magnitude
    """
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = float(np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2)))
    wa = float(C / np.sqrt(Na) * (1 - t))
    wb = float(C / np.sqrt(Nb) * t)
    return torch.cat([wa * a, wb * b], dim=dim)


class MPFourier(nn.Module):
    """
    Magnitude-preserving Fourier features (Equation 75 from EDM2).
    
    Projects input to Fourier features while preserving magnitude.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels))
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size] or [batch_size, 1]
        
        Returns:
            Fourier features of shape [batch_size, num_channels]
        """
        x = x.to(torch.float32).flatten()  # Ensure 1D
        y = x.unsqueeze(-1) * self.freqs.unsqueeze(0) + self.phases.unsqueeze(0)
        y = y.cos() * np.sqrt(2)
        return y


class MPConv(nn.Module):
    """
    Magnitude-preserving convolution or fully-connected layer (Equation 47)
    with forced weight normalization (Equation 66).
    
    This layer normalizes weights to unit magnitude and scales them appropriately
    to preserve activation magnitudes.
    """
    def __init__(self, in_channels, out_channels, kernel):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel: Kernel size as list [h, w] for conv, or [] for linear
        """
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel))
    
    def forward(self, x, gain=1):
        """
        Args:
            x: Input tensor
            gain: Additional gain factor
        
        Returns:
            Output with preserved magnitude
        """
        w = self.weight.to(torch.float32)
        
        # Forced weight normalization during training
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))
        
        # Traditional weight normalization
        w = normalize(w)
        
        # Magnitude-preserving scaling
        w = w * (gain / np.sqrt(w[0].numel()))
        w = w.to(x.dtype)
        
        # Apply operation
        if w.ndim == 2:
            # Fully connected
            # Handle input of shape [B, C] or [B, C, 1, 1]
            if x.ndim > 2:
                # Flatten spatial dimensions if present
                x_shape = x.shape
                x = x.view(x.shape[0], x.shape[1], -1).squeeze(-1)
                result = x @ w.t()
                # Restore original shape structure
                if len(x_shape) == 4:
                    result = result.unsqueeze(-1).unsqueeze(-1)
                return result
            return x @ w.t()
        
        # Convolution
        assert w.ndim == 4
        return F.conv2d(x, w, padding=(w.shape[-1] // 2,))


class MPSiLU(nn.Module):
    """Magnitude-preserving SiLU activation as a module."""
    def forward(self, x):
        return mp_silu(x)
