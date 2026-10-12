"""
PFGM-FIR 2.5D Training Script with Hugging Face Accelerate.

Trains a 2.5D model using 3D convolutions on small volumetric patches.
Multi-GPU training support with FSDP.

Full feature parity with train_accelerate.py (2D version):
- PFGM++ (inverse-beta perturbation/sampling) adapted for 5D tensors
- EMA (Exponential Moving Average) for smoother weights
- CFG++ (manifold-constrained guidance) with dynamic scaling
- Test-Time Augmentation (spatial flips)
- Body mask support (mask.nii loading)
- Classifier-free guidance with linear decay

Key differences from train_accelerate.py:
- Uses 2.5D data loader (small volumetric patches)
- Uses 3D convolutions throughout
- Processes and predicts multiple slices simultaneously
- All tensors are [B, C, D, H, W] (5D)

Usage:
    # Single GPU:
    python train_accelerate_25d.py --data-dir /path/to/paired-data
    
    # Multi-GPU DDP (duplicates model on each GPU):
    accelerate launch --multi_gpu --num_processes 2 --gpu_ids 0,1 train_accelerate_25d.py --data-dir /path/to/paired-data
    
    # FSDP requires a separately configured, validated Accelerate environment.
"""

import argparse
import copy
import json
import gc
import os
import sys
import math
import time
import random
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torchvision.utils as vutils

# Hugging Face Accelerate
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed, ProjectConfiguration

# Metrics
try:
    import torchmetrics
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not installed. Metrics will be disabled.")

# Import 2.5D loader
from loader_25d import MonaiNifti25DDataset, get_monai_nifti_transforms_25d

# Import 2.5D model
from models.network_25d import create_swin_fir_unet_25d

# Compatibility fix for Pillow >= 10
try:
    from PIL import Image as _PILImage
    if not hasattr(_PILImage, "ANTIALIAS"):
        _PILImage.ANTIALIAS = _PILImage.Resampling.LANCZOS
except Exception:
    pass


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

class EMA:
    """
    Exponential Moving Average for model weights.
    
    Maintains a shadow copy of model parameters that are updated with
    exponential moving average. This typically improves generation quality
    by 1-3 dB PSNR in diffusion models.
    
    Usage:
        ema = EMA(model, decay=0.9999)
        
        # In training loop after optimizer.step():
        ema.update()
        
        # For inference/validation:
        ema.apply_shadow()  # Use EMA weights
        ... do inference ...
        ema.restore()       # Restore original weights
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: torch.device = None):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}
        self.num_updates = 0
        self._init_shadow()
    
    def _init_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.device is not None:
                    self.shadow[name] = param.data.clone().to(self.device)
                else:
                    self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        return {
            'shadow': self.shadow,
            'num_updates': self.num_updates,
            'decay': self.decay,
        }
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.num_updates = state_dict['num_updates']
        self.decay = state_dict.get('decay', self.decay)


# =============================================================================
# CONFIGURATION
# =============================================================================

config = {
    # Data Settings
    "data_dir": None,  # Set with --data-dir or a JSON configuration.
    "mr_pattern": "mr.nii",
    "ct_pattern": "ct.nii",
    "train_split": 0.9,
    "image_size": 256,
    
    # 2.5D Settings
    "num_slices": 4,           # Number of consecutive slices (divisible by 2 for attention)
    "slice_stride": 2,         # Stride between volume centers (None = non-overlapping)
    
    # Model Architecture
    "base_channels": 64,
    "channel_mults": [1, 2, 3, 4],  # 4 stages with 64, 128, 192, 256 channels
    "attention_resolutions": [3, 4],  # Attention when mult in [3,4] → stages 2,3
    "num_res_blocks": 1,
    "radius_emb_dim": 256,
    "window_size": 4,  # Spatial window size (H, W) - fallback only
    # Multi-resolution window sizes per stage
    "window_size_3d": {
        1: [2, 4, 4],
        2: [2, 4, 4],   # Stage 2: 192 channels, attention enabled
        3: [1, 4, 4],   # Stage 3: 256 channels, attention enabled
    },
    # Per-stage encoder downsampling strides as [D, H, W].
    # For 4 stages, only the first 3 entries are used by encoder downsamples;
    # the decoder mirrors them in reverse. Use [1, 2, 2] to preserve depth.
    "downsample_strides_3d": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
    "dropout": 0.1,
    "drop_path": 0.05,                    # Stochastic depth rate (matches 2D)
    "gradient_checkpoint": True,  # Trade ~30% speed for ~60% activation memory savings

    # SwinFIR / Attention architecture (mirrors 2D `train_accelerate.py`)
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "num_swin_blocks": 2,                # Attention blocks per FIRResBlock3D
    "num_sfb_blocks": 1,                 # 3D SFB blocks per FIRResBlock3D
    "num_bottleneck_blocks": 1,          # FIRResBlock3D count in bottleneck
    "use_sfb": True,                     # Enable 3D Spatial-Frequency Blocks
    "use_deformable_sfb": False,         # Replace SFB3D spatial-branch convs with deformable convs
    "deformable_sfb_kwargs": {
        "kernel_size": [1, 3, 3],
        "offset_scale": 1.0,
        "use_modulation": False,
        "deformable_depth": False,       # False = learn H/W offsets only
    },
    "attention_type": "deformable",      # 'window' | 'deformable' (3D DAT++)
    "deformable_kwargs": {               # Forwarded to DeformableAttention3D
        "n_groups": 8,
        "kv_stride": 1,                  # Downsample only H,W (depth preserved)
        "offset_range_factor": 8.0,
        "use_qk_norm": True,
    },

    # EDM Parameters (mirrors 2D)
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "sigma_data": 0.45,
    "P_mean": -0.8,
    "P_std": 1.0,
    "rho": 7.0,
    "loss_type": "l2",

    # EDM Stochastic Sampler (Heun + S_churn)
    "S_churn": 20.0,
    "S_min": 0.0,
    "S_max": float('inf'),
    "S_noise": 1.0,

    # PFGM++ Parameters
    "pfgmpp": True,                      # Enable PFGM++ mode (vs pure EDM)
    "D": 128,                            # Augmented dimension for PFGM++ (D→∞ recovers EDM)

    # Classifier-Free Guidance (CFG) — matches 2D defaults
    "cond_drop_prob": 0.1,               # Initial probability of dropping MRI conditioning
    "cond_drop_prob_end_epoch": 0,       # 0 = constant drop rate (no decay)
    "cfg_scale": 2.0,                    # Guidance scale at inference

    # CFG++ and Dynamic Scaling (for sampling/generation)
    "use_cfg_plus_plus": True,           # Use CFG++ manifold constraints
    "cfg_rescale_factor": 0.7,           # Rescale factor for CFG++
    "use_dynamic_cfg": True,             # Use dynamic CFG scaling
    "cfg_scale_min": 1.0,               # Minimum CFG scale (at end of sampling)
    "cfg_scale_max": 2.0,               # Maximum CFG scale (at start of sampling)
    
    # Masked Loss (Anatomical Focusing)
    "use_masked_loss": False,
    "masked_loss_ratio": 0.5,
    "mask_threshold": -0.7,
    
    # Body Mask Training (nnsyn-style anatomically-focused training)
    "use_body_mask": True,               # Enable body mask loading and masked training
    "mask_pattern": "mask",          # Suffix pattern to match mask files
    "mask_outside_value": -1.0,          # Value for regions outside the mask
    
    # Training Parameters
    "num_epochs": 300,
    "batch_size": 5,  # Smaller batch size for 3D convolutions
    "lr": 1e-4,  # Matches 2D canonical
    "weight_decay": 0.01,
    "grad_clip": 1.0,  # Matches 2D canonical
    "warmup_steps": 500,  # Matches 2D canonical
    
    # Sampling / Generation
    "num_sample_steps": 40,
    "num_val_samples": 3,  # Fewer samples for visualization
    
    # Logging & Checkpointing
    "output_dir": "./outputs_accelerate_25d",
    "log_every": 5,
    "val_every": 1,
    "save_every": 20,
    "sample_every": 1,
    "resume": None,
    
    # Gradient & ODE Monitoring
    "log_gradients": True,
    "log_ode_trajectory": True,
    "grad_log_interval": 100,
    
    # System
    "seed": 42,
    "num_workers": 8,
    "cache_volumes": True,
    "mixed_precision": "bf16",  # "no", "fp16", or "bf16" — bf16 halves memory usage
    
    # EMA (Exponential Moving Average)
    "use_ema": True,                     # Enable EMA for smoother weights
    "ema_decay": 0.9999,                 # EMA decay rate
    "ema_update_every": 1,               # Update EMA every N steps
    
    # Test-Time Augmentation (TTA) for validation generation
    "use_tta": False,                    # Enable Flip TTA during val generation (4x slower)
    "tta_flips": ["lr", "ud", "lr_ud"], # Which flips (spatial only, no depth)
}


# =============================================================================
# Utility Functions
# =============================================================================

def _filename_matches(filename: str, pattern: str) -> bool:
    return pattern.lower() in filename.lower()


def collect_data_files(data_path: str, mr_pattern: str = "mr.nii", ct_pattern: str = "ct.nii",
                       mask_pattern: str = "mask", use_body_mask: bool = False):
    """Collect MR, CT, and optionally mask file pairs from directory structure."""
    data_files = []
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    for patient_dir in sorted(os.listdir(data_path)):
        patient_path = os.path.join(data_path, patient_dir)
        if not os.path.isdir(patient_path):
            continue
        
        mr_file = None
        ct_file = None
        mask_file = None
        for filename in os.listdir(patient_path):
            if _filename_matches(filename, mr_pattern):
                mr_file = os.path.join(patient_path, filename)
            if _filename_matches(filename, ct_pattern):
                ct_file = os.path.join(patient_path, filename)
            if use_body_mask and _filename_matches(filename, mask_pattern):
                mask_file = os.path.join(patient_path, filename)
        
        if mr_file and ct_file:
            entry = {"mr": mr_file, "ct": ct_file, "id": patient_dir}
            if use_body_mask:
                entry["mask"] = mask_file  # Can be None if not found
                if mask_file is None:
                    print(f"Warning: No mask found for patient {patient_dir}")
            data_files.append(entry)
    
    return data_files


def check_gradient_flow(named_parameters):
    """Check gradient flow through network layers."""
    grad_info = {}
    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            grad_info[name] = {
                'mean': param.grad.abs().mean().item(),
                'max': param.grad.abs().max().item(),
                'has_nan': torch.isnan(param.grad).any().item(),
            }
    return grad_info


# =============================================================================
# EDM Functions (adapted for 2.5D — 5D tensors)
# =============================================================================

def edm_precond(
    model: nn.Module,
    x: torch.Tensor,
    mri: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Apply EDM preconditioning to model output (2.5D version).
    
    Args:
        x: Noisy CT volume [B, 1, D, H, W]
        mri: MRI condition volume [B, 1, D, H, W]
        sigma: Noise level [B]
        mask: Optional body mask [B, 1, D, H, W] threaded to the model so
              deformable attention can suppress background K/V.
    """
    x = x.float()
    mri = mri.float()
    sigma = sigma.float()
    
    sigma_5d = sigma.view(-1, 1, 1, 1, 1)
    c_skip = sigma_data ** 2 / (sigma_5d ** 2 + sigma_data ** 2)
    c_out = sigma_5d * sigma_data / (sigma_5d ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma_5d ** 2 + sigma_data ** 2).sqrt()
    c_noise = sigma.log() / 4  # [B] - FourierEmbedding handles reshaping
    
    x_scaled = c_in * x
    F_x = model(x_scaled, mri, c_noise, mask=mask)
    D_x = c_skip * x + c_out * F_x
    
    return D_x


def sample_sigma(
    batch_size: int,
    P_mean: float = -1.2,
    P_std: float = 1.2,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Sample sigma values according to EDM log-normal distribution."""
    rnd_normal = torch.randn(batch_size, device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    sigma = sigma.clamp(min=sigma_min, max=sigma_max)
    return sigma


# =============================================================================
# PFGM++ Functions (adapted for 5D tensors [B, C, D, H, W])
# =============================================================================

def pfgmpp_perturbation(
    images: torch.Tensor,
    sigma: torch.Tensor,
    D: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply PFGM++ perturbation using inverse-beta (beta-prime) distribution.
    Adapted for 5D tensors [B, C, D_slices, H, W].
    
    Args:
        images: Clean volumes [B, C, D_slices, H, W]
        sigma: Noise levels [B]
        D: Augmented dimension (D→∞ recovers Gaussian diffusion)
    
    Returns:
        perturbed_images: Volumes with PFGM++ noise [B, C, D_slices, H, W]
        used_sigmas: The sigma values used (for embedding)
    """
    B = images.shape[0]
    N = images[0].numel()  # Data dimension (C*D_slices*H*W)
    device = images.device
    
    # r = sigma * sqrt(D) - the PFGM++ transfer formula
    r = sigma.double() * np.sqrt(D)
    
    # Sample from inverse-beta (beta-prime) distribution — GPU-native (no host
    # sync). Mirrors the canonical 2D `train_accelerate.py` path.
    beta_dist = torch.distributions.Beta(
        torch.tensor(N / 2.0, device=device, dtype=torch.float64),
        torch.tensor(D / 2.0, device=device, dtype=torch.float64),
    )
    samples_norm = beta_dist.sample((B,))
    inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
    
    # Radius = r * sqrt(inverse_beta)
    radius = r * torch.sqrt(inverse_beta + 1e-8)
    
    # Sample uniform direction on N-sphere via normalized Gaussian
    gaussian = torch.randn(B, N, device=device, dtype=torch.float64)
    unit_gaussian = gaussian / (torch.norm(gaussian, p=2, dim=1, keepdim=True) + 1e-8)
    
    # Perturbation = unit_direction * radius
    perturbation = unit_gaussian * radius.view(B, 1)
    
    # Reshape back to volume shape and add to clean images
    perturbation = perturbation.view_as(images).float()
    perturbed_images = images + perturbation
    
    return perturbed_images, sigma


def pfgmpp_target(
    perturbed_images: torch.Tensor,
    clean_images: torch.Tensor,
    sigma: torch.Tensor,
    D: int = 128,
) -> torch.Tensor:
    """
    Compute PFGM++ Poisson field target.
    Adapted for 5D tensors [B, C, D_slices, H, W].
    
    The target is the clean image (denoising target), same as EDM,
    but the perturbation and loss weighting differ.
    
    Args:
        perturbed_images: Noisy volumes [B, C, D_slices, H, W]
        clean_images: Clean target volumes [B, C, D_slices, H, W]
        sigma: Noise levels [B]
        D: Augmented dimension
    
    Returns:
        target: PFGM++ target [B, C, D_slices, H, W]
    """
    return clean_images


def pfgmpp_sample_prior(
    batch_size: int,
    channels: int,
    depth: int,
    height: int,
    width: int,
    sigma_max: float,
    D: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Sample from PFGM++ prior distribution for sampling initialization.
    Adapted for 5D tensors [B, C, D_slices, H, W].
    
    Instead of Gaussian N(0, sigma_max^2), PFGM++ uses:
    - Radius from inverse-beta distribution
    - Direction uniform on N-sphere
    
    Args:
        batch_size: Number of samples
        channels, depth, height, width: Volume dimensions
        sigma_max: Maximum sigma (converted to r_max = sigma_max * sqrt(D))
        D: Augmented dimension
        device: Target device
        dtype: Target dtype
    
    Returns:
        Initial latents for sampling [B, C, D_slices, H, W]
    """
    N = channels * depth * height * width  # Data dimension
    
    # r_max = sigma_max * sqrt(D)
    r_max = sigma_max * np.sqrt(D)
    
    # Sample from inverse-beta distribution — GPU-native (no host sync).
    beta_dist = torch.distributions.Beta(
        torch.tensor(N / 2.0, device=device, dtype=torch.float64),
        torch.tensor(D / 2.0, device=device, dtype=torch.float64),
    )
    samples_norm = beta_dist.sample((batch_size,))
    inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
    
    # Radius = r_max * sqrt(inverse_beta)
    radius = r_max * torch.sqrt(inverse_beta + 1e-8)
    
    # Sample uniform direction on N-sphere
    gaussian = torch.randn(batch_size, N, device=device, dtype=torch.float64)
    unit_gaussian = gaussian / (torch.norm(gaussian, p=2, dim=1, keepdim=True) + 1e-8)
    
    # Initial latent = unit_direction * radius
    latent = unit_gaussian * radius.view(batch_size, 1)
    
    # Reshape to volume
    latent = latent.view(batch_size, channels, depth, height, width).to(dtype)
    
    return latent


# =============================================================================
# Loss Function
# =============================================================================

def edm_loss(
    model: nn.Module,
    x_clean: torch.Tensor,
    mri: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
    loss_type: str = 'l2',
    cond_drop_prob: float = 0.0,
    use_masked_loss: bool = False,
    masked_loss_ratio: float = 0.6,
    mask_threshold: float = -0.7,
    pfgmpp: bool = False,
    D: int = 128,
    body_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute EDM/PFGM++ loss for 2.5D model with optional masking.
    
    Supports two masking modes:
    1. Threshold-based: Generate mask from CT (use_masked_loss=True, body_mask=None)
    2. Body mask: Use explicit body contour mask from dataloader (body_mask not None)
    
    Args:
        x_clean: Clean CT volume [B, 1, D, H, W]
        mri: MRI condition volume [B, 1, D, H, W]
        sigma: Noise levels [B]
        pfgmpp: If True, use PFGM++ perturbation instead of Gaussian
        D: Augmented dimension for PFGM++
        body_mask: Optional explicit body mask [B, 1, D, H, W] from dataloader
    """
    # Apply perturbation: PFGM++ (inverse-beta) or EDM (Gaussian)
    if pfgmpp:
        x_noisy, sigma = pfgmpp_perturbation(x_clean, sigma, D=D)
    else:
        n = torch.randn_like(x_clean) * sigma.view(-1, 1, 1, 1, 1)
        x_noisy = x_clean + n
    
    # Classifier-free guidance: randomly drop conditioning
    if cond_drop_prob > 0.0 and model.training:
        drop_mask = torch.rand(mri.shape[0], device=mri.device) < cond_drop_prob
        mri = mri.clone()
        mri[drop_mask] = 0.0
    
    D_x = edm_precond(model, x_noisy, mri, sigma, sigma_data, mask=body_mask)
    
    sigma_v = sigma.view(-1, 1, 1, 1, 1)
    weight = (sigma_v ** 2 + sigma_data ** 2) / ((sigma_v * sigma_data) ** 2)
    weight = weight.clamp(max=1000.0)
    
    # Apply body mask if provided (nnsyn-style masked training)
    mask_used = None
    use_mask_this_batch = False
    
    if body_mask is not None:
        use_mask_this_batch = True
        mask_used = body_mask
        weight = weight * body_mask
    elif use_masked_loss and model.training:
        if torch.rand(1).item() < masked_loss_ratio:
            use_mask_this_batch = True
            mask_used = (x_clean > mask_threshold).float()
            weight = weight * mask_used
    
    if loss_type == 'l1':
        loss = (weight * (D_x - x_clean).abs()).mean()
    elif loss_type == 'l2':
        loss = (weight * (D_x - x_clean) ** 2).mean()
    elif loss_type == 'huber':
        loss = (weight * F.smooth_l1_loss(D_x, x_clean, reduction='none')).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Metrics for logging
    with torch.no_grad():
        if mask_used is not None:
            masked_pred = D_x * mask_used
            masked_target = x_clean * mask_used
            num_masked_pixels = mask_used.sum()
            if num_masked_pixels > 0:
                mse = ((masked_pred - masked_target) ** 2).sum() / num_masked_pixels
                mse = mse.item()
            else:
                mse = F.mse_loss(D_x, x_clean).item()
        else:
            mse = F.mse_loss(D_x, x_clean).item()
        pred_std = D_x.std().item()
        target_std = x_clean.std().item()
    
    metrics = {
        'mse': mse,
        'pred_std': pred_std,
        'target_std': target_std,
        'sigma_mean': sigma.mean().item(),
        'sigma_max': sigma.max().item(),
        'sigma_min': sigma.min().item(),
    }
    
    if use_mask_this_batch and mask_used is not None:
        metrics['mask_ratio'] = mask_used.mean().item()
        metrics['masked'] = 1.0
    else:
        metrics['masked'] = 0.0
    
    return loss, metrics


# =============================================================================
# Metrics
# =============================================================================

def compute_masked_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    psnr_metric=None,
    ssim_metric=None,
    mae_metric=None,
) -> Dict[str, float]:
    """
    Compute image quality metrics with optional masking (nnsyn-style).
    Works on 2D center slices [B, C, H, W] extracted from 5D volumes.
    
    Args:
        pred: Predicted image [B, C, H, W] in [0, 1] range
        target: Target image [B, C, H, W] in [0, 1] range
        mask: Binary body mask [B, C, H, W] (1=body, 0=outside)
    """
    metrics = {}
    
    # Reset torchmetrics state to avoid accumulating across calls (mirrors 2D)
    if psnr_metric is not None:
        psnr_metric.reset()
    if ssim_metric is not None:
        ssim_metric.reset()
    if mae_metric is not None:
        mae_metric.reset()
    
    if mask is not None:
        pred_masked = pred * mask
        target_masked = target * mask
    else:
        pred_masked = pred
        target_masked = target
    
    if psnr_metric is not None:
        psnr_val = psnr_metric(pred_masked, target_masked)
        metrics['psnr'] = psnr_val.item()
    
    if ssim_metric is not None:
        if mask is not None:
            pred_for_ssim = pred * mask + (1 - mask) * 0.0
            target_for_ssim = target * mask + (1 - mask) * 0.0
            ssim_val = ssim_metric(pred_for_ssim, target_for_ssim)
        else:
            ssim_val = ssim_metric(pred_masked, target_masked)
        metrics['ssim'] = ssim_val.item()
    
    if mae_metric is not None:
        if mask is not None:
            num_masked = mask.sum()
            if num_masked > 0:
                mae_val = (pred_masked - target_masked).abs().sum() / num_masked
                metrics['mae'] = mae_val.item()
            else:
                metrics['mae'] = mae_metric(pred, target).item()
        else:
            mae_val = mae_metric(pred_masked, target_masked)
            metrics['mae'] = mae_val.item()
    
    return metrics


# =============================================================================
# CFG++ and TTA
# =============================================================================

def apply_cfg_plus_plus(denoised_cond: torch.Tensor, denoised_uncond: torch.Tensor, 
                         cfg_scale: float, rescale_factor: float = 0.7) -> torch.Tensor:
    """
    Apply CFG++ with manifold constraints (rescaling).
    Adapted for 5D tensors [B, C, D, H, W].
    
    Standard CFG can push samples off the data manifold, causing oversaturation.
    CFG++ rescales the output to match the statistics of the conditional prediction.
    """
    # Standard CFG
    denoised_cfg = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
    
    if rescale_factor > 0:
        # Compute std of conditional and CFG outputs (per sample, over C,D,H,W)
        std_cond = denoised_cond.std(dim=[1, 2, 3, 4], keepdim=True)
        std_cfg = denoised_cfg.std(dim=[1, 2, 3, 4], keepdim=True)
        
        # Rescale CFG output to match conditional std
        rescale = std_cond / (std_cfg + 1e-8)
        
        # Interpolate between standard CFG and rescaled CFG
        denoised_cfg = rescale_factor * (denoised_cfg * rescale) + (1 - rescale_factor) * denoised_cfg
    
    return denoised_cfg


def _flip_tensor(x: torch.Tensor, flip_type: str) -> torch.Tensor:
    """Flip a tensor spatially (works on both 4D and 5D). Spatial flips only."""
    if flip_type == "lr":
        return torch.flip(x, dims=[-1])       # left-right (flip W)
    elif flip_type == "ud":
        return torch.flip(x, dims=[-2])       # up-down (flip H)
    elif flip_type == "lr_ud":
        return torch.flip(x, dims=[-2, -1])   # both
    else:
        raise ValueError(f"Unknown flip type: {flip_type}")


@torch.no_grad()
def edm_sample_tta(
    model: nn.Module,
    mri: torch.Tensor,
    tta_flips: list = ("lr", "ud", "lr_ud"),
    **sample_kwargs,
) -> torch.Tensor:
    """
    Flip Test-Time Augmentation wrapper around edm_sample.
    
    Runs the sampler on the original input + each flipped version,
    flips predictions back to the original orientation, then averages.
    Spatial flips only (H, W dims) — no depth flipping.
    
    Args:
        model: Denoising model
        mri: Conditioning MRI volumes [B, 1, D, H, W]
        tta_flips: List of flip types: "lr", "ud", "lr_ud"
    
    Returns:
        Averaged CT prediction [B, 1, D, H, W]
    """
    sample_kwargs_no_stats = {k: v for k, v in sample_kwargs.items()
                              if k != 'return_trajectory_stats'}
    sample_kwargs_no_stats['return_trajectory_stats'] = False

    # Extract mask so we can flip it alongside the MRI for each TTA pass.
    mask = sample_kwargs_no_stats.pop('mask', None)

    # 1) Original (unflipped) prediction
    pred_original = edm_sample(model, mri, mask=mask, **sample_kwargs_no_stats)
    preds = [pred_original]

    # 2) Flipped predictions (flip mask in lockstep with MRI)
    for flip_type in tta_flips:
        mri_flipped = _flip_tensor(mri, flip_type)
        mask_flipped = _flip_tensor(mask, flip_type) if mask is not None else None
        pred_flipped = edm_sample(
            model, mri_flipped, mask=mask_flipped, **sample_kwargs_no_stats
        )
        pred_unflipped = _flip_tensor(pred_flipped, flip_type)
        preds.append(pred_unflipped)

    # 3) Average all predictions
    pred_avg = torch.stack(preds, dim=0).mean(dim=0)
    return pred_avg


# =============================================================================
# Sampling
# =============================================================================

@torch.no_grad()
def edm_sample(
    model: nn.Module,
    mri: torch.Tensor,
    num_steps: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
    S_churn: float = 0.0,
    S_min: float = 0.0,
    S_max: float = float('inf'),
    S_noise: float = 1.0,
    return_trajectory_stats: bool = False,
    cfg_scale: float = 1.0,
    pfgmpp: bool = False,
    D: int = 128,
    # CFG++ and Dynamic Scaling options
    use_cfg_plus_plus: bool = False,
    cfg_rescale_factor: float = 0.7,
    use_dynamic_cfg: bool = False,
    cfg_scale_min: float = 1.0,
    cfg_scale_max: float = 3.0,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Generate CT volumes from MRI using EDM/PFGM++ sampling (2.5D version).
    
    Enhanced with:
    - CFG++ (manifold constraints)
    - Dynamic CFG scaling
    - PFGM++ prior initialization
    
    Args:
        mri: MRI condition volume [B, 1, D, H, W]
        return_trajectory_stats: If True, returns (samples, stats_dict)
        cfg_scale: Classifier-free guidance scale
        pfgmpp: If True, use PFGM++ prior for initialization
        D: Augmented dimension for PFGM++
        use_cfg_plus_plus: Enable CFG++ manifold constraints
        cfg_rescale_factor: Rescale factor for CFG++
        use_dynamic_cfg: Enable dynamic CFG scaling
        cfg_scale_min: Minimum CFG scale (at end of sampling)
        cfg_scale_max: Maximum CFG scale (at start of sampling)
    """
    B, C, Ds, H, W = mri.shape
    device = mri.device
    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    
    # Initialize latents: PFGM++ prior or Gaussian
    if pfgmpp:
        x_next = pfgmpp_sample_prior(
            batch_size=B,
            channels=1,  # CT output channels
            depth=Ds,
            height=H,
            width=W,
            sigma_max=sigma_max,
            D=D,
            device=device,
            dtype=torch.float64,
        )
    else:
        x_next = torch.randn(B, 1, Ds, H, W, device=device, dtype=torch.float64) * t_steps[0]
    
    # Track ODE trajectory statistics
    traj_norms = [x_next.norm().item()]
    denoised_norms = []
    score_norms = []
    update_norms = []
    
    # Precompute unconditional input for CFG
    mri_uncond = torch.zeros_like(mri) if cfg_scale != 1.0 or use_dynamic_cfg else None
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Dynamic CFG: linearly interpolate from cfg_scale_max to cfg_scale_min
        if use_dynamic_cfg:
            progress = i / (num_steps - 1)  # 0 at start, 1 at end
            current_cfg_scale = cfg_scale_max + progress * (cfg_scale_min - cfg_scale_max)
        else:
            current_cfg_scale = cfg_scale
        
        # Increase noise temporarily (stochasticity)
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        
        sigma_batch = torch.full((B,), t_hat.item(), device=device, dtype=torch.float32)
        denoised_cond = edm_precond(model, x_hat.float(), mri, sigma_batch, sigma_data, mask=mask).double()
        
        # Classifier-free guidance (standard or CFG++)
        if current_cfg_scale != 1.0:
            denoised_uncond = edm_precond(model, x_hat.float(), mri_uncond, sigma_batch, sigma_data, mask=mask).double()
            
            if use_cfg_plus_plus:
                denoised = apply_cfg_plus_plus(denoised_cond, denoised_uncond,
                                                current_cfg_scale, cfg_rescale_factor)
            else:
                denoised = denoised_uncond + current_cfg_scale * (denoised_cond - denoised_uncond)
        else:
            denoised = denoised_cond
        
        d_cur = (x_hat - denoised) / t_hat
        update = (t_next - t_hat) * d_cur
        x_next = x_hat + update
        
        # Heun's 2nd order correction
        if i < num_steps - 1 and t_next > 0:
            sigma_batch = torch.full((B,), t_next.item(), device=device, dtype=torch.float32)
            denoised_cond_next = edm_precond(model, x_next.float(), mri, sigma_batch, sigma_data, mask=mask).double()
            
            if current_cfg_scale != 1.0:
                denoised_uncond_next = edm_precond(model, x_next.float(), mri_uncond, sigma_batch, sigma_data, mask=mask).double()
                
                if use_cfg_plus_plus:
                    denoised_next = apply_cfg_plus_plus(denoised_cond_next, denoised_uncond_next,
                                                         current_cfg_scale, cfg_rescale_factor)
                else:
                    denoised_next = denoised_uncond_next + current_cfg_scale * (denoised_cond_next - denoised_uncond_next)
            else:
                denoised_next = denoised_cond_next
            
            d_prime = (x_next - denoised_next) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        # Track statistics at checkpoints (every ~20% of steps)
        checkpoint_interval = max(1, num_steps // 5)
        if i % checkpoint_interval == 0 or i == num_steps - 2:
            traj_norms.append(x_next.norm().item())
            denoised_norms.append(denoised.norm().item())
            score_norms.append(d_cur.norm().item())
            update_norms.append(update.norm().item())
    
    result = x_next.float()
    
    if return_trajectory_stats:
        stats = {
            'traj_norms': traj_norms,
            'denoised_norms': denoised_norms,
            'score_norms': score_norms,
            'update_norms': update_norms,
            'initial_norm': traj_norms[0],
            'final_norm': traj_norms[-1],
            'mean_denoised_norm': np.mean(denoised_norms) if denoised_norms else 0,
            'mean_score_norm': np.mean(score_norms) if score_norms else 0,
            'mean_update_norm': np.mean(update_norms) if update_norms else 0,
        }
        return result, stats
    
    return result


# =============================================================================
# Training Function
# =============================================================================

def build_model(cfg: dict):
    """Construct the unchanged 2.5D architecture from a resolved configuration."""
    ws_3d = cfg["window_size_3d"]
    if isinstance(ws_3d, dict):
        ws_3d = {k: tuple(v) if isinstance(v, list) else v for k, v in ws_3d.items()}
    else:
        ws_3d = tuple(ws_3d) if isinstance(ws_3d, list) else ws_3d

    downsample_strides_3d = [
        tuple(s) if isinstance(s, list) else s
        for s in cfg.get("downsample_strides_3d", [(2, 2, 2)] * (len(cfg["channel_mults"]) - 1))
    ]

    return create_swin_fir_unet_25d(
        in_channels=2,
        out_channels=1,
        num_slices=cfg["num_slices"],
        base_channels=cfg["base_channels"],
        channel_mults=tuple(cfg["channel_mults"]),
        attention_resolutions=tuple(cfg["attention_resolutions"]),
        num_res_blocks=cfg["num_res_blocks"],
        num_heads=cfg.get("num_heads", 8),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        radius_emb_dim=cfg["radius_emb_dim"],
        window_size=cfg["window_size"],
        window_size_3d=ws_3d,
        num_swin_blocks=cfg.get("num_swin_blocks", 2),
        num_sfb_blocks=cfg.get("num_sfb_blocks", 1),
        num_bottleneck_blocks=cfg.get("num_bottleneck_blocks", 2),
        use_sfb=cfg.get("use_sfb", True),
        dropout=cfg["dropout"],
        drop_path_rate=cfg.get("drop_path", 0.0),
        gradient_checkpoint=cfg.get("gradient_checkpoint", False),
        attention_type=cfg.get("attention_type", "window"),
        deformable_kwargs=cfg.get("deformable_kwargs", None),
        use_deformable_sfb=cfg.get("use_deformable_sfb", False),
        deformable_sfb_kwargs=cfg.get("deformable_sfb_kwargs", None),
        downsample_strides=downsample_strides_3d,
    )



def train(cfg: dict):
    """Main training function with Accelerate for 2.5D model."""

    # Enable cuDNN autotuner for stable input shapes (mirrors 2D trainer).
    torch.backends.cudnn.benchmark = True

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg["output_dir"]) / f"{timestamp}_pfgm_fir_25d"
    
    # Initialize Accelerator
    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
    )
    
    accelerator = Accelerator(
        mixed_precision=cfg["mixed_precision"],
        cpu=cfg.get("cpu", False),
        gradient_accumulation_steps=1,
        project_config=project_config,
        log_with="tensorboard",
    )
    
    # Set seed
    set_seed(cfg["seed"])
    
    # Only main process prints
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        save_config(cfg, output_dir / "config.json")
        
        print(f"\n{'='*60}")
        print(f"PFGM-FIR 2.5D Training with Accelerate")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Mixed precision: {cfg['mixed_precision']}")
        print(f"Num slices: {cfg['num_slices']}")
        print(f"PFGM++: {cfg.get('pfgmpp', False)}")
        print(f"{'='*60}\n")
    
    # Build 2.5D model
    model = build_model(cfg)
    downsample_strides_3d = cfg.get("downsample_strides_3d")

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params/1e6:.2f}M")
        print(f"  Architecture: 2.5D with 3D convolutions")
        print(f"    base_channels: {cfg['base_channels']}")
        print(f"    channel_mults: {cfg['channel_mults']}")
        print(f"    attention_resolutions: {cfg['attention_resolutions']}")
        print(f"    num_res_blocks: {cfg['num_res_blocks']}")
        print(f"    num_slices: {cfg['num_slices']}")
        print(f"    downsample_strides_3d: {downsample_strides_3d}")
        print(f"    use_deformable_sfb: {cfg.get('use_deformable_sfb', False)}")
        print(f"    gradient_checkpoint: {cfg.get('gradient_checkpoint', False)}")
        print(f"    mixed_precision: {cfg['mixed_precision']}")
    
    # Check if FSDP is enabled
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    if accelerator.is_main_process:
        print(f"Distributed type: {accelerator.distributed_type}")
        if is_fsdp:
            print("FSDP enabled - optimizer will be created after model wrapping")
    
    # For non-FSDP, create optimizer before prepare
    optimizer = None
    if not is_fsdp:
        optimizer = AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.95),
        )
    
    # =========================================================================
    # Data loading
    # =========================================================================
    if accelerator.is_main_process:
        print("Loading data files...")
    
    use_body_mask = cfg.get("use_body_mask", False)
    mask_pattern = cfg.get("mask_pattern", "mask")
    
    all_files = collect_data_files(
        cfg["data_dir"],
        mr_pattern=cfg["mr_pattern"],
        ct_pattern=cfg["ct_pattern"],
        mask_pattern=mask_pattern,
        use_body_mask=use_body_mask,
    )
    
    random.shuffle(all_files)
    split_idx = max(1, int(len(all_files) * cfg["train_split"]))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:] if len(all_files) > split_idx else None
    
    if accelerator.is_main_process:
        (output_dir / "split.json").write_text(json.dumps({
            "train": train_files, "validation": val_files or [],
        }, indent=2) + "\n")
        print(f"Found {len(train_files)} train, {len(val_files) if val_files else 0} val patients")
        if use_body_mask:
            print(f"Body mask training ENABLED - using mask pattern: {mask_pattern}")
        print("Creating 2.5D datasets...")
    
    train_transforms = get_monai_nifti_transforms_25d(mode="train", use_body_mask=use_body_mask)
    train_dataset = MonaiNifti25DDataset(
        data=train_files,
        transform=train_transforms,
        target_size=cfg["image_size"],
        num_slices=cfg["num_slices"],
        stride=cfg.get("slice_stride"),
        use_body_mask=use_body_mask,
        cache_volumes=cfg.get("cache_volumes", True),
        mode="train",
    )
    
    if accelerator.is_main_process:
        print(f"Train dataset: {len(train_dataset)} volumes")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    
    if accelerator.is_main_process:
        print(f"Train loader: {len(train_loader)} batches")
    
    val_loader = None
    if val_files:
        val_transforms = get_monai_nifti_transforms_25d(mode="val", use_body_mask=use_body_mask)
        val_dataset = MonaiNifti25DDataset(
            data=val_files,
            transform=val_transforms,
            target_size=cfg["image_size"],
            num_slices=cfg["num_slices"],
            stride=cfg.get("slice_stride"),
            use_body_mask=use_body_mask,
            cache_volumes=cfg.get("cache_volumes", True),
            mode="val",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
    
    # =========================================================================
    # Learning rate scheduler
    # =========================================================================
    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = cfg.get("warmup_steps", 500)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = None
    if not is_fsdp:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # =========================================================================
    # Prepare with Accelerator
    # =========================================================================
    if accelerator.is_main_process:
        print("Preparing model and data with Accelerator...")
    
    accelerator.wait_for_everyone()
    
    if is_fsdp:
        model = accelerator.prepare(model)
        optimizer = AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.95),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        optimizer, train_loader, scheduler = accelerator.prepare(optimizer, train_loader, scheduler)
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    
    if accelerator.is_main_process:
        print("Model and data prepared!")
    
    # =========================================================================
    # Initialize EMA
    # =========================================================================
    ema = None
    if cfg.get("use_ema", True):
        unwrapped_model = accelerator.unwrap_model(model)
        ema = EMA(
            unwrapped_model,
            decay=cfg.get("ema_decay", 0.9999),
            device=accelerator.device,
        )
        if accelerator.is_main_process:
            print(f"EMA enabled with decay={cfg.get('ema_decay', 0.9999)}")
    
    # Initialize tracking
    trackable_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)):
            trackable_cfg[k] = str(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            trackable_cfg[k] = v if v is not None else "None"
        else:
            trackable_cfg[k] = str(v)
    accelerator.init_trackers("pfgm_fir_25d", config=trackable_cfg)
    
    if accelerator.is_main_process:
        print(f"Training batches per epoch: {len(train_loader)}")
        print(f"Total training steps: {total_steps}")
        print(f"Starting training...\n")
    
    # =========================================================================
    # Metrics
    # =========================================================================
    if TORCHMETRICS_AVAILABLE and accelerator.is_main_process:
        psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(accelerator.device)
        ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(accelerator.device)
        mae_metric = torchmetrics.MeanAbsoluteError().to(accelerator.device)
    else:
        psnr_metric = ssim_metric = mae_metric = None
    
    global_step = 0
    best_val_loss = float('inf')
    
    # =========================================================================
    # Training loop
    # =========================================================================
    for epoch in range(cfg["num_epochs"]):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Linearly decay cond_drop_prob from initial value to 0.0
        cond_drop_start = cfg.get("cond_drop_prob", 0.0)
        cond_drop_end_epoch = cfg.get("cond_drop_prob_end_epoch", 40)
        if cond_drop_end_epoch > 0 and cond_drop_start > 0:
            cond_drop_prob = max(0.0, cond_drop_start * (1.0 - epoch / cond_drop_end_epoch))
        else:
            cond_drop_prob = cond_drop_start
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg['num_epochs']}",
            disable=not accelerator.is_main_process
        )
        
        for batch_idx, batch in enumerate(pbar):
            mr = batch["mr"]  # [B, 1, D, H, W]
            ct = batch["ct"]  # [B, 1, D, H, W]
            body_mask = batch.get("mask", None)
            
            batch_size = mr.shape[0]
            sigma = sample_sigma(
                batch_size=batch_size,
                P_mean=cfg["P_mean"],
                P_std=cfg["P_std"],
                sigma_min=cfg["sigma_min"],
                sigma_max=cfg["sigma_max"],
                device=accelerator.device,
            )
            
            with accelerator.accumulate(model):
                loss, metrics = edm_loss(
                    model, ct, mr, sigma,
                    sigma_data=cfg["sigma_data"],
                    loss_type=cfg["loss_type"],
                    cond_drop_prob=cond_drop_prob,
                    use_masked_loss=cfg.get("use_masked_loss", False),
                    masked_loss_ratio=cfg.get("masked_loss_ratio", 0.6),
                    mask_threshold=cfg.get("mask_threshold", -0.7),
                    pfgmpp=cfg.get("pfgmpp", False),
                    D=cfg.get("D", 128),
                    body_mask=body_mask,
                )
                
                accelerator.backward(loss)
                
                if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update EMA weights
                if ema is not None and global_step % cfg.get("ema_update_every", 1) == 0:
                    ema.update()
            
            # Gather loss across processes
            gathered_loss = accelerator.gather(loss.detach()).mean()
            epoch_loss += gathered_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{gathered_loss.item():.3e}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "cdp": f"{cond_drop_prob:.3f}",
            })
            
            if batch_idx % cfg["log_every"] == 0:
                log_dict = {
                    "train/loss": gathered_loss.item(),
                    "train/mse": metrics['mse'],
                    "train/pred_std": metrics['pred_std'],
                    "train/target_std": metrics['target_std'],
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/sigma_mean": metrics['sigma_mean'],
                    "train/sigma_max": metrics['sigma_max'],
                    "train/sigma_min": metrics['sigma_min'],
                    "train/masked": metrics['masked'],
                    "train/cond_drop_prob": cond_drop_prob,
                }
                if 'mask_ratio' in metrics:
                    log_dict["train/mask_ratio"] = metrics['mask_ratio']
                
                accelerator.log(log_dict, step=global_step)
            
            # Check gradient flow periodically
            if cfg.get("log_gradients", False) and batch_idx % cfg.get("grad_log_interval", 100) == 0:
                unwrapped = accelerator.unwrap_model(model)
                grad_info = check_gradient_flow(unwrapped.named_parameters())
                if grad_info:
                    grad_means = [v['mean'] for v in grad_info.values()]
                    grad_maxs = [v['max'] for v in grad_info.values()]
                    has_nan = any(v['has_nan'] for v in grad_info.values())
                    accelerator.log({
                        "grad/mean_avg": float(np.mean(grad_means)),
                        "grad/mean_max": float(np.max(grad_means)),
                        "grad/max_avg": float(np.mean(grad_maxs)),
                        "grad/max_max": float(np.max(grad_maxs)),
                        "grad/has_nan": float(has_nan),
                    }, step=global_step)
            
            global_step += 1
        
        # =====================================================================
        # Epoch stats
        # =====================================================================
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start
        
        accelerator.log({
            "train/epoch_loss": avg_loss,
            "system/epoch_time": epoch_time,
        }, step=epoch)
        
        if torch.cuda.is_available() and accelerator.is_main_process:
            gpu_mem_mb = torch.cuda.max_memory_allocated(accelerator.device) / 1024**2
            accelerator.log({"system/gpu_memory_MB": gpu_mem_mb}, step=epoch)
        
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}: avg_loss={avg_loss:.3e}, time={epoch_time:.1f}s")
        
        # =====================================================================
        # Sample generation (visualize center slice)
        # =====================================================================
        if (epoch + 1) % cfg["sample_every"] == 0:
            model.eval()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # All ranks must participate in DataLoader iteration to avoid NCCL timeout
            sample_batch = next(iter(train_loader))
            mr_samples = sample_batch["mr"][:cfg["num_val_samples"]]
            ct_samples = sample_batch["ct"][:cfg["num_val_samples"]]
            sample_mask = sample_batch.get("mask", None)
            if sample_mask is not None:
                sample_mask = sample_mask[:cfg["num_val_samples"]]
            
            if accelerator.is_main_process:
                with torch.no_grad():
                    unwrapped_model = accelerator.unwrap_model(model)
                    
                    # Apply EMA weights for better generation quality
                    if ema is not None:
                        ema.apply_shadow()
                    
                    # Build sample kwargs
                    sample_kwargs = dict(
                        num_steps=cfg["num_sample_steps"],
                        sigma_min=cfg["sigma_min"],
                        sigma_max=cfg["sigma_max"],
                        sigma_data=cfg["sigma_data"],
                        rho=cfg["rho"],
                        S_churn=cfg.get("S_churn", 0.0),
                        S_min=cfg.get("S_min", 0.0),
                        S_max=cfg.get("S_max", float('inf')),
                        S_noise=cfg.get("S_noise", 1.0),
                        cfg_scale=cfg.get("cfg_scale", 1.0),
                        pfgmpp=cfg.get("pfgmpp", False),
                        D=cfg.get("D", 128),
                        use_cfg_plus_plus=cfg.get("use_cfg_plus_plus", False),
                        cfg_rescale_factor=cfg.get("cfg_rescale_factor", 0.7),
                        use_dynamic_cfg=cfg.get("use_dynamic_cfg", False),
                        cfg_scale_min=cfg.get("cfg_scale_min", 1.0),
                        cfg_scale_max=cfg.get("cfg_scale_max", 3.0),
                        mask=sample_mask.to(accelerator.device) if sample_mask is not None else None,
                    )
                    
                    if cfg.get("log_ode_trajectory", False):
                        pred_ct, ode_stats = edm_sample(
                            unwrapped_model, mr_samples,
                            return_trajectory_stats=True,
                            **sample_kwargs,
                        )
                        accelerator.log({
                            "ode/initial_norm": ode_stats['initial_norm'],
                            "ode/final_norm": ode_stats['final_norm'],
                            "ode/mean_denoised_norm": ode_stats['mean_denoised_norm'],
                            "ode/mean_score_norm": ode_stats['mean_score_norm'],
                            "ode/mean_update_norm": ode_stats['mean_update_norm'],
                        }, step=epoch)
                    else:
                        pred_ct = edm_sample(
                            unwrapped_model, mr_samples,
                            **sample_kwargs,
                        )
                    
                    # Extract center slice for visualization [B, 1, D, H, W] → [B, 1, H, W]
                    center_idx = cfg["num_slices"] // 2
                    pred_ct_center = pred_ct[:, :, center_idx, :, :]
                    ct_samples_center = ct_samples[:, :, center_idx, :, :]
                    mr_samples_center = mr_samples[:, :, center_idx, :, :]
                    
                    pred_ct_center = torch.clamp(pred_ct_center, -1.0, 1.0)
                    pred_ct_01 = (pred_ct_center + 1.0) / 2.0
                    ct_samples_01 = (ct_samples_center + 1.0) / 2.0
                    
                    # Prepare mask for metrics (extract center slice)
                    mask_for_metrics = None
                    if sample_mask is not None and cfg.get("use_body_mask", False):
                        mask_for_metrics = sample_mask[:, :, center_idx, :, :].to(pred_ct.device)
                    
                    if psnr_metric is not None:
                        gen_metrics = compute_masked_metrics(
                            pred_ct_01, ct_samples_01,
                            mask=mask_for_metrics,
                            psnr_metric=psnr_metric,
                            ssim_metric=ssim_metric,
                            mae_metric=mae_metric,
                        )
                        
                        log_dict = {
                            "gen/psnr": gen_metrics.get('psnr', 0.0),
                            "gen/ssim": gen_metrics.get('ssim', 0.0),
                        }
                        if 'mae' in gen_metrics:
                            log_dict["gen/mae"] = gen_metrics['mae']
                        
                        accelerator.log(log_dict, step=epoch)
                        mae_str = f", MAE={gen_metrics['mae']:.5f}" if 'mae' in gen_metrics else ""
                        mask_str = " (masked)" if mask_for_metrics is not None else ""
                        print(f"  Generation{mask_str} (center slice): PSNR={gen_metrics.get('psnr', 0.0):.3f}, SSIM={gen_metrics.get('ssim', 0.0):.5f}{mae_str}")
                    
                    # Log training sample images to TensorBoard
                    img_grid = vutils.make_grid(
                        torch.cat([mr_samples_center.cpu(), pred_ct_center.cpu(), ct_samples_center.cpu()], dim=0),
                        nrow=cfg["num_val_samples"],
                        normalize=True,
                        scale_each=True
                    )
                    tb_tracker = accelerator.get_tracker("tensorboard")
                    if tb_tracker is not None:
                        tb_tracker.writer.add_image("Gen/MRI_Pred_GT_CenterSlice", img_grid, epoch)
                    
                    # Restore original weights after EMA sampling
                    if ema is not None:
                        ema.restore()
            
            # Free sampling memory before resuming training
            del sample_batch, mr_samples, ct_samples
            if sample_mask is not None:
                del sample_mask
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            model.train()
        
        # =====================================================================
        # Validation
        # =====================================================================
        if val_loader is not None and (epoch + 1) % cfg["val_every"] == 0:
            model.eval()
            val_loss_accum = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    mr = batch["mr"]
                    ct = batch["ct"]
                    val_body_mask = batch.get("mask", None)
                    
                    sigma = sample_sigma(
                        batch_size=ct.shape[0],
                        P_mean=cfg["P_mean"],
                        P_std=cfg["P_std"],
                        sigma_min=cfg["sigma_min"],
                        sigma_max=cfg["sigma_max"],
                        device=accelerator.device,
                    )
                    
                    loss, _ = edm_loss(
                        model, ct, mr, sigma,
                        sigma_data=cfg["sigma_data"],
                        loss_type=cfg["loss_type"],
                        pfgmpp=cfg.get("pfgmpp", False),
                        D=cfg.get("D", 128),
                        body_mask=val_body_mask,
                    )
                    
                    gathered_loss = accelerator.gather(loss.detach()).mean()
                    val_loss_accum += gathered_loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss_accum / val_batches
            accelerator.log({"val/loss": avg_val_loss}, step=epoch)
            
            if accelerator.is_main_process:
                print(f"  Validation: loss={avg_val_loss:.3e}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                accelerator.save_state(output_dir / "checkpoints" / "best")
                if accelerator.is_main_process:
                    print(f"  New best model saved!")
            
            # Free validation loss memory before generation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Validation sample generation
            val_sample_batch = next(iter(val_loader))
            mr_val_samples = val_sample_batch["mr"][:cfg["num_val_samples"]]
            ct_val_samples = val_sample_batch["ct"][:cfg["num_val_samples"]]
            val_sample_mask = val_sample_batch.get("mask", None)
            if val_sample_mask is not None:
                val_sample_mask = val_sample_mask[:cfg["num_val_samples"]]
            
            if accelerator.is_main_process and psnr_metric is not None:
                unwrapped_model = accelerator.unwrap_model(model)
                
                # Apply EMA weights
                if ema is not None:
                    ema.apply_shadow()
                
                val_sample_kwargs = dict(
                    num_steps=cfg["num_sample_steps"],
                    sigma_min=cfg["sigma_min"],
                    sigma_max=cfg["sigma_max"],
                    sigma_data=cfg["sigma_data"],
                    rho=cfg["rho"],
                    S_churn=cfg.get("S_churn", 0.0),
                    S_min=cfg.get("S_min", 0.0),
                    S_max=cfg.get("S_max", float('inf')),
                    S_noise=cfg.get("S_noise", 1.0),
                    cfg_scale=cfg.get("cfg_scale", 1.0),
                    pfgmpp=cfg.get("pfgmpp", False),
                    D=cfg.get("D", 128),
                    use_cfg_plus_plus=cfg.get("use_cfg_plus_plus", False),
                    cfg_rescale_factor=cfg.get("cfg_rescale_factor", 0.7),
                    use_dynamic_cfg=cfg.get("use_dynamic_cfg", False),
                    cfg_scale_min=cfg.get("cfg_scale_min", 1.0),
                    cfg_scale_max=cfg.get("cfg_scale_max", 3.0),
                    mask=val_sample_mask.to(accelerator.device) if val_sample_mask is not None else None,
                )
                
                if cfg.get("use_tta", False):
                    pred_val_ct = edm_sample_tta(
                        unwrapped_model, mr_val_samples,
                        tta_flips=cfg.get("tta_flips", ["lr", "ud", "lr_ud"]),
                        **val_sample_kwargs,
                    )
                else:
                    pred_val_ct = edm_sample(
                        unwrapped_model, mr_val_samples,
                        **val_sample_kwargs,
                    )
                
                # Extract center slice for metrics and visualization
                center_idx = cfg["num_slices"] // 2
                pred_val_center = pred_val_ct[:, :, center_idx, :, :]
                ct_val_center = ct_val_samples[:, :, center_idx, :, :]
                mr_val_center = mr_val_samples[:, :, center_idx, :, :]
                
                pred_val_center = torch.clamp(pred_val_center, -1.0, 1.0)
                pred_val_ct_01 = (pred_val_center + 1.0) / 2.0
                ct_val_samples_01 = (ct_val_center + 1.0) / 2.0
                
                # Prepare mask for validation metrics (center slice)
                val_mask_for_metrics = None
                if val_sample_mask is not None and cfg.get("use_body_mask", False):
                    val_mask_for_metrics = val_sample_mask[:, :, center_idx, :, :].to(pred_val_ct.device)
                
                val_gen_metrics = compute_masked_metrics(
                    pred_val_ct_01, ct_val_samples_01,
                    mask=val_mask_for_metrics,
                    psnr_metric=psnr_metric,
                    ssim_metric=ssim_metric,
                    mae_metric=mae_metric,
                )
                
                val_gen_log = {
                    "val_gen/psnr": val_gen_metrics.get('psnr', 0.0),
                    "val_gen/ssim": val_gen_metrics.get('ssim', 0.0),
                }
                if 'mae' in val_gen_metrics:
                    val_gen_log["val_gen/mae"] = val_gen_metrics['mae']
                
                accelerator.log(val_gen_log, step=epoch)
                
                val_mae_str = f", MAE={val_gen_metrics['mae']:.5f}" if 'mae' in val_gen_metrics else ""
                val_mask_str = " (masked)" if val_mask_for_metrics is not None else ""
                print(f"  Val Generation{val_mask_str} (center slice): PSNR={val_gen_metrics.get('psnr', 0.0):.3f}, SSIM={val_gen_metrics.get('ssim', 0.0):.5f}{val_mae_str}")
                
                # Log validation sample images to TensorBoard
                val_img_grid = vutils.make_grid(
                    torch.cat([mr_val_center.cpu(), pred_val_center.cpu(), ct_val_center.cpu()], dim=0),
                    nrow=cfg["num_val_samples"],
                    normalize=True,
                    scale_each=True
                )
                tb_tracker = accelerator.get_tracker("tensorboard")
                if tb_tracker is not None:
                    tb_tracker.writer.add_image("Val_Gen/MRI_Pred_GT_CenterSlice", val_img_grid, epoch)
                
                # Restore original weights
                if ema is not None:
                    ema.restore()
            
            # Free validation generation memory
            del val_sample_batch, mr_val_samples, ct_val_samples
            if val_sample_mask is not None:
                del val_sample_mask
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            model.train()
        
        # =====================================================================
        # Save checkpoint
        # =====================================================================
        if (epoch + 1) % cfg["save_every"] == 0:
            accelerator.save_state(output_dir / "checkpoints" / f"epoch_{epoch+1:04d}")
            if accelerator.is_main_process:
                print(f"  Checkpoint saved")
    
    # Final save
    accelerator.save_state(output_dir / "checkpoints" / "final")
    accelerator.end_training()
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Output: {output_dir}")
        print(f"Best val loss: {best_val_loss:.3e}")
        print(f"{'='*60}\n")


def save_config(cfg, path):
    """Write standard JSON, representing the unbounded sampler limit explicitly."""
    serializable = copy.deepcopy(cfg)
    if math.isinf(serializable.get("S_max", 0)):
        serializable["S_max"] = "Infinity"
    Path(path).write_text(json.dumps(serializable, indent=2, allow_nan=False) + "\n")


def load_config(path=None):
    """Load partial JSON overrides without mutating the historical defaults."""
    cfg = copy.deepcopy(config)
    if path is not None:
        overrides = json.loads(Path(path).read_text())
        unknown = set(overrides) - set(cfg) - {"cpu"}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        cfg.update(overrides)
    if cfg.get("S_max") == "Infinity":
        cfg["S_max"] = float("inf")
    if isinstance(cfg["window_size_3d"], dict):
        cfg["window_size_3d"] = {int(k): v for k, v in cfg["window_size_3d"].items()}
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="JSON overrides for the default configuration")
    parser.add_argument("--data-dir", type=Path, help="Directory of paired, normalized NIfTI patient folders")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--cpu", action="store_true", help="CPU execution for small pipeline checks")
    parser.add_argument("--print-config", action="store_true", help="Print resolved configuration and exit")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for name, key in [("data_dir", "data_dir"), ("output_dir", "output_dir"),
                      ("epochs", "num_epochs"), ("batch_size", "batch_size"),
                      ("num_workers", "num_workers")]:
        value = getattr(args, name)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    if args.cpu:
        cfg["cpu"] = True
        cfg["mixed_precision"] = "no"
    if args.print_config:
        printable = copy.deepcopy(cfg)
        if math.isinf(printable["S_max"]):
            printable["S_max"] = "Infinity"
        print(json.dumps(printable, indent=2, allow_nan=False))
        return
    if not cfg["data_dir"]:
        parser.error("provide --data-dir or data_dir in --config")
    if not Path(cfg["data_dir"]).is_dir():
        parser.error("data_dir must be an existing directory")
    if cfg.get("resume"):
        parser.error("checkpoint resume is not implemented; resume must be null")
    if cfg["num_epochs"] < 1 or cfg["batch_size"] < 1 or cfg["num_workers"] < 0:
        parser.error("epochs and batch size must be positive; workers must be nonnegative")
    if cfg["num_slices"] < 2 or cfg["num_slices"] % 2:
        parser.error("use an even num_slices of at least 2 for the current loader")
    train(cfg)


if __name__ == '__main__':
    main()
