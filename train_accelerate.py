"""
PFGM-FIR Training Script with Hugging Face Accelerate.

Multi-GPU training with FSDP support for larger models.

Usage:
    # Single GPU:
    python train_accelerate.py
    
    # Multi-GPU with accelerate:
    accelerate launch --num_processes 2 --gpu_ids 0,1 train_accelerate.py
    
    # Or with config file:
    accelerate launch --config_file accelerate_config.yaml train_accelerate.py
"""

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

# Import loader
from loader import MonaiNiftiSliceDataset, get_monai_nifti_transforms

# Import model
from models import create_swin_fir_unet

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
        """
        Args:
            model: Model to track
            decay: EMA decay rate (0.9999 is typical for diffusion models)
            device: Device to store shadow weights (None = same as model)
        """
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}
        self.num_updates = 0
        
        # Initialize shadow weights
        self._init_shadow()
    
    def _init_shadow(self):
        """Initialize shadow weights from current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.device is not None:
                    self.shadow[name] = param.data.clone().to(self.device)
                else:
                    self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        """Update shadow weights with exponential moving average."""
        self.num_updates += 1
        
        # Use warmup: start with lower decay and increase to target decay
        # This helps during early training when weights change rapidly
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # EMA update: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def apply_shadow(self):
        """Apply shadow (EMA) weights to model, backing up current weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            'shadow': self.shadow,
            'num_updates': self.num_updates,
            'decay': self.decay,
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.num_updates = state_dict['num_updates']
        self.decay = state_dict.get('decay', self.decay)


# =============================================================================
# CONFIGURATION
# =============================================================================

config = {
    # Data Settings
    "data_dir": "/mnt/nvme/lloyd/DL project CT-MRI/pfgm_brain_train_n",
    "mr_pattern": "mr.nii", # Suffix pattern to match MRI files
    "ct_pattern": "ct.nii", # Suffix pattern to match CT files
    "train_split": 0.9,
    "image_size": 256,
    
    # Model Architecture
    "model_config": "custom", # "custom" or predefined config name
    "base_channels": 64,   # Reduced from 128 for memory
    "channel_mults": [1, 2, 4, 8],  # 4 stages: optimal for 256x256 (64→128→256→512 ch)
    "num_res_blocks": 2,  # Number of times the Swin+SFB block repeats per encoder/decoder stage
    "attention_resolutions": [4, 8],  # Enable attention at 256ch and 512ch stages (critical!)
    "num_heads": 8,
    "mlp_ratio": 4.0, # MLP expansion ratio in transformer blocks
    "radius_emb_dim": 256,
    "window_size": 8, # Swin Transformer window size
    "num_swin_blocks": 4,  # Number of Swin attention blocks per FIRResBlock (W-MSA + SW-MSA pattern)
    "num_sfb_blocks": 1,   # Number of SFB blocks per FIRResBlock (sequential after Swin)
    "num_bottleneck_blocks": 2,  # Number of residual blocks in the bottleneck
    "use_sfb": True,  # Use Spatial-Frequency Blocks
    "dropout": 0.00,
    "drop_path": 0.1,
    "gradient_checkpoint": True,  # Trade ~30% speed for ~60% activation memory savings
    
    # EDM Parameters
    "sigma_min": 0.01,
    "sigma_max": 80.0,
    "sigma_data": 0.45, # 0.5 for [-1,1] normalized data, 1.0 for z-normalized
    "P_mean": -1.2, #changed for pfgm++
    "P_std": 1.2,   # changed for pfgm++
    "rho": 7.0,
    "loss_type": "l2",
    
    # PFGM++ Parameters
    "pfgmpp": True,                      # Enable PFGM++ mode (vs pure EDM)
    "D": 128,                            # Augmented dimension for PFGM++ (D→∞ recovers EDM)

    # EDM Stochastic Sampling (Algorithm 2 in EDM paper)
    # 0.0 = deterministic Heun. Small positive values (S_churn ~5-40) inject
    # extra noise during sampling and often improve perceptual quality on PFGM++.
    "S_churn": 20.0,                      # Total stochasticity budget across all steps
    "S_min": 0.05,                        # Apply churn only when sigma >= S_min
    "S_max": 50.0,               # Apply churn only when sigma <= S_max
    "S_noise": 1.0,                      # Noise scale (paper uses 1.003 for ImageNet)

    # Classifier-Free Guidance (CFG)
    "cond_drop_prob": 0.1,               # Initial probability of dropping MRI conditioning (0.0 = no CFG)
    "cond_drop_prob_end_epoch": 0,      # Epoch at which cond_drop_prob reaches 0.0 (linear decay)
    "cfg_scale": 2.0,                    # Guidance scale at inference (1.0 = no guidance)
    
    # CFG++ and Dynamic Scaling (for sampling/generation)
    "use_cfg_plus_plus": True,           # Use CFG++ manifold constraints
    "cfg_rescale_factor": 0.7,           # Rescale factor for CFG++ (0.0 = no rescale, 1.0 = full rescale)
    "use_dynamic_cfg": True,             # Use dynamic CFG scaling
    "cfg_scale_min": 1.0,                # Minimum CFG scale (at end of sampling)
    "cfg_scale_max": 2.0,                # Maximum CFG scale (at start of sampling)
    
    # Masked Loss (Anatomical Focusing)
    "use_masked_loss": False,             # Enable masked loss (body-only)
    "masked_loss_ratio": 0,            # Ratio of samples using masked loss (0.0-1.0) 
    "mask_threshold": 0.0,              # Threshold for body mask in z-normalized range (adjust based on data)
    
    # Body Mask Training (nnsyn-style anatomically-focused training)
    "use_body_mask": True,              # Enable body mask loading and masked training
    "mask_pattern": "mask.nii",          # Suffix pattern to match mask files
    "mask_outside_value": -1.0,          # Value to use for regions outside the mask in generated images
    
    # Training Parameters
    "num_epochs": 150,
    "batch_size": 20,  # Per GPU batch size (reduced for memory)
    "lr": 4e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,  # Enable gradient clipping
    "warmup_steps": 500,
    
    # Sampling / Generation
    "num_sample_steps": 50,
    "num_val_samples": 5,
    
    # Logging & Checkpointing
    "output_dir": "./outputs_accelerate",
    "log_every": 5,
    "val_every": 1,
    "save_every": 20,
    "sample_every": 1,
    "resume": None,
    
    # Gradient & ODE Monitoring
    "log_gradients": True,
    "log_ode_trajectory": True,
    "grad_log_interval": 150,
    
    # System
    "seed": 42,
    "num_workers": 4,
    "mixed_precision": "bf16",  # "no", "fp16", or "bf16"
    
    # EMA (Exponential Moving Average)
    "use_ema": True,                     # Enable EMA for smoother weights
    "ema_decay": 0.9999,                 # EMA decay rate (higher = slower update)
    "ema_update_every": 1,               # Update EMA every N steps
    
    # Test-Time Augmentation (TTA) for validation generation
    "use_tta": False,                    # Enable Flip TTA during val generation (4x slower)
    "tta_flips": ["lr", "ud", "lr_ud"], # Which flips: "lr", "ud", "lr_ud"
    
    # Attention type: 'window' (original Swin) or 'deformable' (DAT++ with QK-Norm)
    "attention_type": "deformable",
    "deformable_kwargs": {"n_groups": 8, "kv_stride": 2, "offset_range_factor": 4.0},           # e.g. {"n_groups": 4, "kv_stride": 1}
}


# =============================================================================
# Utility Functions
# =============================================================================

def collect_data_files(data_path: str, mr_pattern: str = "mr.nii", ct_pattern: str = "ct.nii", 
                       mask_pattern: str = "mask.nii", use_body_mask: bool = False):
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
            if mr_pattern in filename:
                mr_file = os.path.join(patient_path, filename)
            if ct_pattern in filename:
                ct_file = os.path.join(patient_path, filename)
            if use_body_mask and mask_pattern in filename:
                mask_file = os.path.join(patient_path, filename)
        
        if mr_file and ct_file:
            entry = {"mr": mr_file, "ct": ct_file, "id": patient_dir}
            if use_body_mask:
                entry["mask"] = mask_file  # Can be None if not found
                if mask_file is None:
                    print(f"Warning: No mask found for patient {patient_dir}")
            data_files.append(entry)
    
    return data_files


# =============================================================================
# Gradient & ODE Monitoring (from train_pfgm_fir.py)
# =============================================================================

def check_gradient_flow(named_parameters):
    """
    Check gradient flow through network layers.
    
    Returns dict with gradient statistics per layer for monitoring.
    Useful for detecting vanishing/exploding gradients.
    """
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
# EDM Functions
# =============================================================================

def edm_precond(
    model: nn.Module,
    x: torch.Tensor,
    mri: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply EDM preconditioning to model output."""
    x = x.float()
    mri = mri.float()
    sigma = sigma.float()
    
    sigma_4d = sigma.view(-1, 1, 1, 1)
    c_skip = sigma_data ** 2 / (sigma_4d ** 2 + sigma_data ** 2)
    c_out = sigma_4d * sigma_data / (sigma_4d ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma_4d ** 2 + sigma_data ** 2).sqrt()
    c_noise = sigma.log() / 4
    
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
# PFGM++ Functions
# =============================================================================

def pfgmpp_perturbation(
    images: torch.Tensor,
    sigma: torch.Tensor,
    D: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply PFGM++ perturbation using inverse-beta (beta-prime) distribution.
    
    This implements the augmented-space perturbation from PFGM++ paper:
    - r = sigma * sqrt(D)  (transfer formula)
    - Sample radius R from inverse-beta distribution
    - Perturb in N-dimensional data space with uniform direction on N-sphere
    
    Args:
        images: Clean images [B, C, H, W]
        sigma: Noise levels [B]
        D: Augmented dimension (D→∞ recovers Gaussian diffusion)
    
    Returns:
        perturbed_images: Images with PFGM++ noise [B, C, H, W]
        used_sigmas: The sigma values used (for embedding)
    """
    B = images.shape[0]
    N = images[0].numel()  # Data dimension (C*H*W)
    device = images.device
    
    # r = sigma * sqrt(D) - the PFGM++ transfer formula
    r = sigma.double() * np.sqrt(D)
    
    # Sample from inverse-beta (beta-prime) distribution directly on GPU
    # Beta(N/2, D/2) → use torch.distributions to avoid CPU↔GPU transfer
    beta_dist = torch.distributions.Beta(
        torch.tensor(N / 2., device=device, dtype=torch.float64),
        torch.tensor(D / 2., device=device, dtype=torch.float64),
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
    
    # Reshape back to image shape and add to clean images
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
    
    The target is the normalized direction from perturbed to clean,
    scaled by the augmented-space geometry.
    
    For PFGM++, the network predicts the Poisson field direction,
    which points from the noisy sample toward the data manifold.
    
    Args:
        perturbed_images: Noisy images [B, C, H, W]
        clean_images: Clean target images [B, C, H, W]
        sigma: Noise levels [B]
        D: Augmented dimension
    
    Returns:
        target: PFGM++ target for the network [B, C, H, W]
    """
    B = perturbed_images.shape[0]
    N = perturbed_images[0].numel()
    device = perturbed_images.device
    
    # r = sigma * sqrt(D)
    r = sigma.view(-1, 1, 1, 1) * np.sqrt(D)
    
    # Difference vector (perturbed - clean) in data space
    diff = perturbed_images - clean_images
    
    # Compute the norm of the difference
    diff_flat = diff.view(B, -1)
    diff_norm = torch.norm(diff_flat, p=2, dim=1, keepdim=True).view(B, 1, 1, 1)
    
    # PFGM++ target: direction scaled by geometry
    # The target points from perturbed back to clean, normalized
    # With augmented dimension factor
    norm_sq = diff_norm ** 2 + r ** 2
    
    # Target is the clean image (denoising target), same as EDM
    # but the loss weighting differs
    target = clean_images
    
    return target


def pfgmpp_sample_prior(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    sigma_max: float,
    D: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Sample from PFGM++ prior distribution for sampling initialization.
    
    Instead of Gaussian N(0, sigma_max^2), PFGM++ uses:
    - Radius from inverse-beta distribution
    - Direction uniform on N-sphere
    
    Args:
        batch_size: Number of samples
        channels, height, width: Image dimensions
        sigma_max: Maximum sigma (converted to r_max = sigma_max * sqrt(D))
        D: Augmented dimension
        device: Target device
        dtype: Target dtype
    
    Returns:
        Initial latents for sampling [B, C, H, W]
    """
    N = channels * height * width  # Data dimension
    
    # r_max = sigma_max * sqrt(D)
    r_max = sigma_max * np.sqrt(D)
    
    # Sample from inverse-beta distribution (directly on GPU)
    beta_dist = torch.distributions.Beta(
        torch.tensor(N / 2., device=device, dtype=torch.float64),
        torch.tensor(D / 2., device=device, dtype=torch.float64),
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
    
    # Reshape to image
    latent = latent.view(batch_size, channels, height, width).to(dtype)
    
    return latent


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
    Compute EDM/PFGM++ loss with optional classifier-free guidance training and masked loss.
    
    Supports two masking modes:
    1. Threshold-based: Generate mask from CT (old method, use_masked_loss=True, body_mask=None)
    2. Body mask: Use explicit body contour mask from dataloader (body_mask not None)
    
    Args:
        cond_drop_prob: Probability of dropping MRI conditioning (replacing with zeros).
                        Set to 0.0 to disable CFG training.
        use_masked_loss: If True, apply body masking to focus loss on anatomy (threshold-based).
        masked_loss_ratio: Probability of using masked loss (0.0-1.0) for threshold mode.
        mask_threshold: Threshold for body mask (pixels > threshold are body).
        pfgmpp: If True, use PFGM++ perturbation (inverse-beta) instead of Gaussian.
        D: Augmented dimension for PFGM++ (D→∞ recovers EDM/Gaussian).
        body_mask: Optional explicit body mask [B, 1, H, W] from dataloader (nnsyn-style).
    
    Returns:
        loss: Scalar loss value
        metrics: Dictionary of metrics for logging
    """
    # Apply perturbation: PFGM++ (inverse-beta) or EDM (Gaussian)
    if pfgmpp:
        x_noisy, sigma = pfgmpp_perturbation(x_clean, sigma, D=D)
    else:
        n = torch.randn_like(x_clean) * sigma.view(-1, 1, 1, 1)
        x_noisy = x_clean + n
    
    # Classifier-free guidance: randomly drop conditioning
    if cond_drop_prob > 0.0 and model.training:
        # Create mask for which samples drop conditioning
        drop_mask = torch.rand(mri.shape[0], device=mri.device) < cond_drop_prob
        # Zero out MRI for dropped samples
        mri = mri.clone()
        mri[drop_mask] = 0.0
    
    D_x = edm_precond(model, x_noisy, mri, sigma, sigma_data, mask=body_mask)
    
    sigma_v = sigma.view(-1, 1, 1, 1)
    weight = (sigma_v ** 2 + sigma_data ** 2) / ((sigma_v * sigma_data) ** 2)
    weight = weight.clamp(max=1000.0)
    
    # Apply body mask if provided (nnsyn-style masked training)
    mask_used = None
    use_mask_this_batch = False
    
    if body_mask is not None:
        # Use explicit body mask from dataloader
        use_mask_this_batch = True
        mask_used = body_mask
        weight = weight * body_mask
    elif use_masked_loss and model.training:
        # Fallback: Generate mask from CT threshold (old method)
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
    
    # Metrics for logging (matching train_pfgm_fir.py)
    with torch.no_grad():
        # Compute MSE only on masked region if mask is used
        if mask_used is not None:
            # Masked MSE (nnsyn-style)
            masked_pred = D_x * mask_used
            masked_target = x_clean * mask_used
            num_masked_pixels = mask_used.sum().item()
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
    
    # Add mask statistics if used
    if use_mask_this_batch and mask_used is not None:
        mask_ratio = mask_used.mean().item()
        metrics['mask_ratio'] = mask_ratio
        metrics['masked'] = 1.0
    else:
        metrics['masked'] = 0.0
    
    return loss, metrics


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
    
    When mask is provided, metrics are computed only on the body region.
    Outside regions are set to a constant value before computing metrics.
    
    Args:
        pred: Predicted image [B, C, H, W] in [0, 1] range
        target: Target image [B, C, H, W] in [0, 1] range
        mask: Binary body mask [B, C, H, W] (1=body, 0=outside)
        psnr_metric: torchmetrics PSNR object
        ssim_metric: torchmetrics SSIM object
        mae_metric: torchmetrics MAE object
    
    Returns:
        Dictionary with psnr, ssim, mae values
    """
    metrics = {}

    # Reset stateful torchmetrics objects so each call returns the per-batch
    # value rather than the cumulative running mean accumulated since the
    # metric was created (otherwise epoch-N values are dragged down by all
    # earlier noisy batches, which especially hurts SSIM).
    if psnr_metric is not None:
        psnr_metric.reset()
    if ssim_metric is not None:
        ssim_metric.reset()
    if mae_metric is not None:
        mae_metric.reset()

    if mask is not None:
        # Apply mask - set outside region to 0 for fair comparison
        # This follows nnsyn approach where outside is set to constant value
        pred_masked = pred * mask
        target_masked = target * mask
    else:
        pred_masked = pred
        target_masked = target
    
    if psnr_metric is not None:
        psnr_val = psnr_metric(pred_masked, target_masked)
        metrics['psnr'] = psnr_val.item()
    
    if ssim_metric is not None:
        # For SSIM with mask, we follow nnsyn approach
        if mask is not None:
            # Set outside to same value to not affect SSIM
            pred_for_ssim = pred * mask + (1 - mask) * 0.0
            target_for_ssim = target * mask + (1 - mask) * 0.0
            ssim_val = ssim_metric(pred_for_ssim, target_for_ssim)
        else:
            ssim_val = ssim_metric(pred_masked, target_masked)
        metrics['ssim'] = ssim_val.item()
    
    if mae_metric is not None:
        if mask is not None:
            # Compute MAE only on masked region
            num_masked = mask.sum().item()
            if num_masked > 0:
                mae_val = (pred_masked - target_masked).abs().sum() / num_masked
                metrics['mae'] = mae_val.item()
            else:
                metrics['mae'] = mae_metric(pred, target).item()
        else:
            mae_val = mae_metric(pred_masked, target_masked)
            metrics['mae'] = mae_val.item()
    
    return metrics


def apply_cfg_plus_plus(denoised_cond: torch.Tensor, denoised_uncond: torch.Tensor, 
                         cfg_scale: float, rescale_factor: float = 0.7) -> torch.Tensor:
    """
    Apply CFG++ with manifold constraints (rescaling).
    
    Standard CFG can push samples off the data manifold, causing oversaturation.
    CFG++ rescales the output to match the statistics of the conditional prediction,
    keeping samples closer to the learned manifold.
    
    Args:
        denoised_cond: Conditional denoised prediction
        denoised_uncond: Unconditional denoised prediction  
        cfg_scale: Guidance scale
        rescale_factor: How much to rescale (0.0 = standard CFG, 1.0 = full rescale)
    
    Returns:
        CFG++ guided denoised prediction
    """
    # Standard CFG
    denoised_cfg = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
    
    if rescale_factor > 0:
        # Compute std of conditional and CFG outputs (per sample)
        std_cond = denoised_cond.std(dim=[1, 2, 3], keepdim=True)
        std_cfg = denoised_cfg.std(dim=[1, 2, 3], keepdim=True)
        
        # Rescale CFG output to match conditional std
        rescale = std_cond / (std_cfg + 1e-8)
        
        # Interpolate between standard CFG and rescaled CFG
        denoised_cfg = rescale_factor * (denoised_cfg * rescale) + (1 - rescale_factor) * denoised_cfg
    
    return denoised_cfg


def _flip_tensor(x: torch.Tensor, flip_type: str) -> torch.Tensor:
    """Flip a [B, C, H, W] tensor."""
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
    flips predictions back to the original orientation, then averages
    all predictions. This reduces noise and improves metric consistency.
    
    Args:
        model: Denoising model
        mri: Conditioning MRI images [B, 1, H, W]
        tta_flips: List of flip types: "lr", "ud", "lr_ud"
        **sample_kwargs: All other arguments forwarded to edm_sample()
                         (including optional 'mask' for body-mask models)
    
    Returns:
        Averaged CT prediction [B, 1, H, W]
    """
    # Extract mask (must be flipped alongside MRI for spatial consistency)
    mask = sample_kwargs.pop('mask', None)

    # Remove return_trajectory_stats for TTA (only use stats from original)
    sample_kwargs_no_stats = {k: v for k, v in sample_kwargs.items()
                              if k != 'return_trajectory_stats'}
    sample_kwargs_no_stats['return_trajectory_stats'] = False
    
    # 1) Original (unflipped) prediction
    pred_original = edm_sample(model, mri, mask=mask, **sample_kwargs_no_stats)
    preds = [pred_original]
    
    # 2) Flipped predictions
    for flip_type in tta_flips:
        mri_flipped = _flip_tensor(mri, flip_type)
        mask_flipped = _flip_tensor(mask, flip_type) if mask is not None else None
        pred_flipped = edm_sample(
            model, mri_flipped, mask=mask_flipped, **sample_kwargs_no_stats
        )
        # Flip prediction back to original orientation
        pred_unflipped = _flip_tensor(pred_flipped, flip_type)
        preds.append(pred_unflipped)
    
    # 3) Average all predictions
    pred_avg = torch.stack(preds, dim=0).mean(dim=0)
    return pred_avg


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
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate CT images from MRI using EDM/PFGM++ sampling.
    
    This is a deterministic ODE sampler with optional stochasticity.
    When pfgmpp=True, uses PFGM++ prior (inverse-beta) for initialization.
    
    Enhanced with:
    - CFG++ (manifold constraints): Rescales CFG output to match conditional statistics
    - Dynamic CFG scaling: Linearly decreases guidance from cfg_scale_max to cfg_scale_min
    
    Args:
        return_trajectory_stats: If True, returns (samples, stats_dict) with ODE trajectory info
        cfg_scale: Classifier-free guidance scale. 1.0 = no guidance, >1.0 = stronger conditioning.
                   Requires model trained with cond_drop_prob > 0.
        pfgmpp: If True, use PFGM++ prior (inverse-beta) instead of Gaussian.
        D: Augmented dimension for PFGM++.
        use_cfg_plus_plus: Enable CFG++ manifold constraints.
        cfg_rescale_factor: Rescale factor for CFG++ (0=standard, 1=full rescale).
        use_dynamic_cfg: Enable dynamic CFG scaling.
        cfg_scale_min: Minimum CFG scale (at end of sampling).
        cfg_scale_max: Maximum CFG scale (at start of sampling).
        mask: Optional binary body mask [B, 1, H, W]. Required when model was
              trained with use_body_mask=True (in_channels=3). Pass None to
              let the model fall back to an all-ones mask (assume all-body).
    """
    B, C, H, W = mri.shape
    device = mri.device
    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    
    # Initialize latents: PFGM++ prior or Gaussian
    if pfgmpp:
        x_next = pfgmpp_sample_prior(
            batch_size=B,
            channels=1,  # CT output channels
            height=H,
            width=W,
            sigma_max=sigma_max,
            D=D,
            device=device,
            dtype=torch.float64,
        )
    else:
        x_next = torch.randn(B, 1, H, W, device=device, dtype=torch.float64) * t_steps[0]
    
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
            
            # CFG for Heun correction
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

def train(cfg: dict):
    """Main training function with Accelerate."""
    
    # Performance: enable cuDNN auto-tuner for fixed input sizes
    torch.backends.cudnn.benchmark = True
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg["output_dir"]) / f"{timestamp}_pfgm_fir"
    
    # Initialize Accelerator
    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
    )
    
    accelerator = Accelerator(
        mixed_precision=cfg["mixed_precision"],
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
        
        print(f"\n{'='*60}")
        print(f"PFGM-FIR Training with Accelerate")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Mixed precision: {cfg['mixed_precision']}")
        print(f"{'='*60}\n")
    
    # Build model
    if cfg["model_config"] == "custom":
        model = create_swin_fir_unet(
            config="custom",
            in_channels=2,
            out_channels=1,
            base_channels=cfg["base_channels"],
            channel_mults=cfg["channel_mults"],
            num_res_blocks=cfg["num_res_blocks"],
            attention_resolutions=cfg["attention_resolutions"],
            num_heads=cfg["num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            radius_emb_dim=cfg["radius_emb_dim"],
            window_size=cfg["window_size"],
            num_swin_blocks=cfg["num_swin_blocks"],
            num_sfb_blocks=cfg["num_sfb_blocks"],
            num_bottleneck_blocks=cfg["num_bottleneck_blocks"],
            use_sfb=cfg["use_sfb"],
            dropout=cfg["dropout"],
            drop_path_rate=cfg["drop_path"],
            gradient_checkpoint=cfg.get("gradient_checkpoint", False),
            attention_type=cfg.get("attention_type", "window"),
            deformable_kwargs=cfg.get("deformable_kwargs", None),
        )
    else:
        model = create_swin_fir_unet(
            config=cfg["model_config"],
            in_channels=2,
        )
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params/1e6:.2f}M")
        print(f"  Architecture config:")
        print(f"    base_channels: {cfg['base_channels']}")
        print(f"    channel_mults: {cfg['channel_mults']}")
        print(f"    num_res_blocks: {cfg['num_res_blocks']}")
        print(f"    attention_resolutions: {cfg['attention_resolutions']}")
        print(f"    num_heads: {cfg['num_heads']}")
    
    # Check if FSDP is enabled (detected from accelerate config)
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    if accelerator.is_main_process:
        print(f"Distributed type: {accelerator.distributed_type}")
        if is_fsdp:
            print("FSDP enabled - optimizer will be created after model wrapping")
    
    # For non-FSDP, create optimizer before prepare (standard path)
    optimizer = None
    if not is_fsdp:
        optimizer = AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.95),
        )
    
    # Data - only main process loads initially to avoid duplicate scanning
    if accelerator.is_main_process:
        print("Loading data files...")
    
    # Determine if using body mask
    use_body_mask = cfg.get("use_body_mask", False)
    mask_pattern = cfg.get("mask_pattern", "mask.nii")
    
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
        print(f"Found {len(train_files)} train, {len(val_files) if val_files else 0} val patients")
        if use_body_mask:
            print(f"Body mask training ENABLED - using mask pattern: {mask_pattern}")
        print("Creating datasets...")
    
    train_transforms = get_monai_nifti_transforms(mode="train", target_size=cfg["image_size"], use_body_mask=use_body_mask)
    train_dataset = MonaiNiftiSliceDataset(
        data=train_files,
        transform=train_transforms,
        target_size=cfg["image_size"],
        use_body_mask=use_body_mask,
        cache_volumes=True,
        mode="train",
    )
    
    if accelerator.is_main_process:
        print(f"Train dataset: {len(train_dataset)} slices")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg["num_workers"] > 0,
    )
    
    if accelerator.is_main_process:
        print(f"Train loader: {len(train_loader)} batches")
    
    val_loader = None
    if val_files:
        val_transforms = get_monai_nifti_transforms(mode="val", target_size=cfg["image_size"], use_body_mask=use_body_mask)
        val_dataset = MonaiNiftiSliceDataset(
            data=val_files,
            transform=val_transforms,
            target_size=cfg["image_size"],
            use_body_mask=use_body_mask,
            cache_volumes=True,
            mode="val",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            persistent_workers=cfg["num_workers"] > 0,
        )
    
    # Learning rate scheduler (lr_lambda defined here, scheduler created after prepare for FSDP)
    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = cfg.get("warmup_steps", 500)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress))) # Decay to 1% of LR
    
    # For non-FSDP, create scheduler before prepare
    scheduler = None
    if not is_fsdp:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Optional: torch.compile for kernel fusion (may conflict with deformable attention)
    if cfg.get("compile_model", False):
        if accelerator.is_main_process:
            print("Compiling model with torch.compile (mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")

    # Prepare with Accelerate
    if accelerator.is_main_process:
        print("Preparing model and data with Accelerator...")
        print(f"  DEBUG: is_fsdp={is_fsdp}, distributed_type={accelerator.distributed_type}")
    
    # Force a barrier to ensure all processes are synced
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("  DEBUG: Starting prepare()...")
    
    if is_fsdp:
        # FSDP path: prepare model first, then create optimizer from wrapped model
        if accelerator.is_main_process:
            print("  DEBUG: FSDP path - preparing model...")
        model = accelerator.prepare(model)
        
        # Create optimizer after FSDP wrapping (critical for FSDP)
        optimizer = AdamW(
            model.parameters(),  # These are now FSDP-wrapped parameters
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.95),
        )
        
        # Create scheduler with the new optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Prepare optimizer, dataloader, scheduler
        if accelerator.is_main_process:
            print("  DEBUG: FSDP path - preparing optimizer, dataloader, scheduler...")
        optimizer, train_loader, scheduler = accelerator.prepare(
            optimizer, train_loader, scheduler
        )
    else:
        # Standard path: prepare everything together
        if accelerator.is_main_process:
            print("  DEBUG: Standard path - preparing all...")
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    
    if accelerator.is_main_process:
        print("Model and data prepared!")
    
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    
    # Initialize EMA (Exponential Moving Average)
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
    
    # Initialize tracking (convert non-primitive types to strings for TensorBoard)
    trackable_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)):
            trackable_cfg[k] = str(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            trackable_cfg[k] = v if v is not None else "None"
        else:
            trackable_cfg[k] = str(v)
    accelerator.init_trackers("pfgm_fir", config=trackable_cfg)
    
    if accelerator.is_main_process:
        print(f"Training batches per epoch: {len(train_loader)}")
        print(f"Total training steps: {total_steps}")
        print(f"Starting training...\n")
    
    # Metrics
    if TORCHMETRICS_AVAILABLE and accelerator.is_main_process:
        psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(accelerator.device)
        ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(accelerator.device)
        mae_metric = torchmetrics.MeanAbsoluteError().to(accelerator.device)
    else:
        psnr_metric = ssim_metric = mae_metric = None
    
    global_step = 0
    best_val_loss = float('inf')
    
    # Training loop
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
        
        _t_data = 0.0
        _t_fwd = 0.0
        _t_bwd = 0.0
        _t_step_start = time.time()

        for batch_idx, batch in enumerate(pbar):
            _t_data += time.time() - _t_step_start  # data loading time

            mr = batch["mr"]
            ct = batch["ct"]
            # Get body mask if available (nnsyn-style training)
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
            
            _t0 = time.time()
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
                    body_mask=body_mask,  # Pass body mask to loss function
                )
                _t_fwd += time.time() - _t0

                _t0 = time.time()
                accelerator.backward(loss)
                
                if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                _t_bwd += time.time() - _t0
                
                # Update EMA weights
                if ema is not None and global_step % cfg.get("ema_update_every", 1) == 0:
                    ema.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            _t_step_start = time.time()  # reset for next data-load timing
            
            pbar.set_postfix({
                "loss": f"{loss.item():.3e}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "cdp": f"{cond_drop_prob:.3f}",
            })
            
            if batch_idx % cfg["log_every"] == 0:
                log_dict = {
                    "train/loss": loss.item(),
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
                # Add mask ratio if masking was used
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
        
        # Epoch stats
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start if 'epoch_start' in dir() else 0
        
        accelerator.log({
            "train/epoch_loss": avg_loss,
            "system/epoch_time": epoch_time,
        }, step=epoch)
        
        # Log GPU memory if available
        if torch.cuda.is_available() and accelerator.is_main_process:
            gpu_mem_mb = torch.cuda.max_memory_allocated(accelerator.device) / 1024**2
            accelerator.log({"system/gpu_memory_MB": gpu_mem_mb}, step=epoch)
        
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}: avg_loss={avg_loss:.3e}, time={epoch_time:.1f}s")
            if num_batches > 0:
                print(f"  Timing breakdown: data={_t_data/num_batches:.3f}s  "
                      f"fwd={_t_fwd/num_batches:.3f}s  "
                      f"bwd+step={_t_bwd/num_batches:.3f}s  per iter")
        
        # Sample generation
        if (epoch + 1) % cfg["sample_every"] == 0:
            model.eval()
            
            # Free GPU memory before sampling to prevent OOM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # All ranks must participate in DataLoader iteration to avoid NCCL timeout
            sample_batch = next(iter(train_loader))
            mr_samples = sample_batch["mr"][:cfg["num_val_samples"]]
            ct_samples = sample_batch["ct"][:cfg["num_val_samples"]]
            # Get body mask for masked metrics if available
            sample_mask = sample_batch.get("mask", None)
            if sample_mask is not None:
                sample_mask = sample_mask[:cfg["num_val_samples"]]
            
            # Only main process does generation and logging
            if accelerator.is_main_process:
                with torch.no_grad():
                
                    # Get unwrapped model for sampling
                    unwrapped_model = accelerator.unwrap_model(model)
                    
                    # Apply EMA weights for better generation quality
                    if ema is not None:
                        ema.apply_shadow()
                    
                    # Sample with optional ODE trajectory logging
                    if cfg.get("log_ode_trajectory", False):
                        pred_ct, ode_stats = edm_sample(
                            unwrapped_model, mr_samples,
                            num_steps=cfg["num_sample_steps"],
                            sigma_min=cfg["sigma_min"],
                            sigma_max=cfg["sigma_max"],
                            sigma_data=cfg["sigma_data"],
                            rho=cfg["rho"],
                            S_churn=cfg.get("S_churn", 0.0),
                            S_min=cfg.get("S_min", 0.0),
                            S_max=cfg.get("S_max", float('inf')),
                            S_noise=cfg.get("S_noise", 1.0),
                            return_trajectory_stats=True,
                            cfg_scale=cfg.get("cfg_scale", 1.0),
                            pfgmpp=cfg.get("pfgmpp", False),
                            D=cfg.get("D", 128),
                            # CFG++ and Dynamic Scaling
                            use_cfg_plus_plus=cfg.get("use_cfg_plus_plus", False),
                            cfg_rescale_factor=cfg.get("cfg_rescale_factor", 0.7),
                            use_dynamic_cfg=cfg.get("use_dynamic_cfg", False),
                            cfg_scale_min=cfg.get("cfg_scale_min", 1.0),
                            cfg_scale_max=cfg.get("cfg_scale_max", 3.0),
                            mask=sample_mask,
                        )
                        # Log ODE trajectory stats
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
                            # CFG++ and Dynamic Scaling
                            use_cfg_plus_plus=cfg.get("use_cfg_plus_plus", False),
                            cfg_rescale_factor=cfg.get("cfg_rescale_factor", 0.7),
                            use_dynamic_cfg=cfg.get("use_dynamic_cfg", False),
                            cfg_scale_min=cfg.get("cfg_scale_min", 1.0),
                            cfg_scale_max=cfg.get("cfg_scale_max", 3.0),
                            mask=sample_mask,
                        )
                    
                    #pred_ct = torch.clamp(pred_ct, -3.0, 6.0)
                    pred_ct_01 = (pred_ct + 1.0) / 2.0
                    ct_samples_01 = (ct_samples + 1.0) / 2.0
                    
                    # Prepare mask for metrics (scale to [0,1] range)
                    mask_for_metrics = None
                    if sample_mask is not None and cfg.get("use_body_mask", False):
                        mask_for_metrics = sample_mask.to(pred_ct.device)
                    
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
                        print(f"  Generation{mask_str}: PSNR={gen_metrics.get('psnr', 0.0):.3f}, SSIM={gen_metrics.get('ssim', 0.0):.5f}{mae_str}")
                    
                    # Log training sample images to TensorBoard
                    img_grid = vutils.make_grid(
                        torch.cat([mr_samples.cpu(), pred_ct.cpu(), ct_samples.cpu()], dim=0),
                        nrow=cfg["num_val_samples"],
                        normalize=True,
                        scale_each=True
                    )
                    # Get TensorBoard writer from Accelerate tracker
                    tb_tracker = accelerator.get_tracker("tensorboard")
                    if tb_tracker is not None:
                        tb_tracker.writer.add_image("Gen/MRI_Pred_GT", img_grid, epoch)
                    
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
        
        # Validation
        if val_loader is not None and (epoch + 1) % cfg["val_every"] == 0:
            model.eval()
            val_loss_accum = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    mr = batch["mr"]
                    ct = batch["ct"]
                    # Get body mask for masked loss if available
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
                    val_loss_accum += loss.item()
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
            # All ranks must participate in DataLoader iteration to avoid NCCL timeout
            val_sample_batch = next(iter(val_loader))
            mr_val_samples = val_sample_batch["mr"][:cfg["num_val_samples"]]
            ct_val_samples = val_sample_batch["ct"][:cfg["num_val_samples"]]
            # Get body mask for masked metrics if available
            val_sample_mask = val_sample_batch.get("mask", None)
            if val_sample_mask is not None:
                val_sample_mask = val_sample_mask[:cfg["num_val_samples"]]
            
            # Only main process does generation and logging
            if accelerator.is_main_process and psnr_metric is not None:
                unwrapped_model = accelerator.unwrap_model(model)
                
                # Apply EMA weights for better generation quality
                if ema is not None:
                    ema.apply_shadow()
                
                # Generate samples from validation MRI
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
                )
                
                if cfg.get("use_tta", False):
                    pred_val_ct = edm_sample_tta(
                        unwrapped_model, mr_val_samples,
                        tta_flips=cfg.get("tta_flips", ["lr", "ud", "lr_ud"]),
                        mask=val_sample_mask,
                        **val_sample_kwargs,
                    )
                else:
                    pred_val_ct = edm_sample(
                        unwrapped_model, mr_val_samples,
                        mask=val_sample_mask,
                        **val_sample_kwargs,
                    )
                
                #pred_val_ct = torch.clamp(pred_val_ct, -3.0, 6.0)
                pred_val_ct_01 = (pred_val_ct + 1.0) / 2.0
                ct_val_samples_01 = (ct_val_samples + 1.0) / 2.0
                
                # Prepare mask for validation metrics
                val_mask_for_metrics = None
                if val_sample_mask is not None and cfg.get("use_body_mask", False):
                    val_mask_for_metrics = val_sample_mask.to(pred_val_ct.device)
                
                # Compute validation generation metrics with optional masking
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
                print(f"  Val Generation{val_mask_str}: PSNR={val_gen_metrics.get('psnr', 0.0):.3f}, SSIM={val_gen_metrics.get('ssim', 0.0):.5f}{val_mae_str}")
                
                # Log validation sample images to TensorBoard
                val_img_grid = vutils.make_grid(
                    torch.cat([mr_val_samples.cpu(), pred_val_ct.cpu(), ct_val_samples.cpu()], dim=0),
                    nrow=cfg["num_val_samples"],
                    normalize=True,
                    scale_each=True
                )
                tb_tracker = accelerator.get_tracker("tensorboard")
                if tb_tracker is not None:
                    tb_tracker.writer.add_image("Val_Gen/MRI_Pred_GT", val_img_grid, epoch)
                
                # Restore original weights after EMA sampling
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
        
        # Save checkpoint
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


if __name__ == '__main__':
    train(config)
