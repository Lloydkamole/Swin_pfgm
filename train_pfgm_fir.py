"""
PFGM-FIR Training Script.

Training script for SwinFIR-based PFGM++ model for MRI-to-CT synthesis.
Based on the structure of train_swin_pfgmpp_2d.py from pfgmpp.

Key features:
- PFGM++ perturbation and loss
- SwinFIR-based UNet architecture
- MRI conditioning via input concatenation
- MONAI-based NIfTI data loading (all slices)
- TensorBoard logging with metrics (PSNR, SSIM, MAE)
- Checkpoint saving and resuming
- Training and validation generation visualization

Usage:
    python train_pfgm_fir.py
"""

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
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils

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
# CONFIGURATION - Modify these values directly
# =============================================================================

config = {
    # -------------------------------------------------------------------------
    # Data Settings
    # -------------------------------------------------------------------------
    "data_dir": "/home/lloyd/Documents/AR PFGM/train_pfgm",  # Path to NIfTI data
    "mr_pattern": "mr.nii",              # Pattern to match MR files
    "ct_pattern": "ct.nii",              # Pattern to match CT files
    "train_split": 0.9,                  # Train/val split ratio
    "image_size": 256,                   # Input image size
    
    # -------------------------------------------------------------------------
    # Model Architecture - Size Control
    # -------------------------------------------------------------------------
    # Preset config: "small", "base", "large", or "custom"
    # If "custom", the parameters below are used directly
    # If preset, these are overridden by preset values
    "model_config": "custom",
    
    # Custom architecture parameters (used when model_config="custom")
    "base_channels": 64,                 # Base channel count (reduce for smaller model: 32, 48, 64)
    "channel_mults": [1, 2, 4],       # Channel multipliers per stage (e.g., [1,2,4] for 3 stages)
    "num_res_blocks": 2,                 # ResBlocks per stage (1-3, fewer = smaller)
    "attention_resolutions": [1, 2, 4],     # Which channel mults get Swin attention (e.g., [4,8] or [8])
    "num_heads": 4,                      # Attention heads (2, 4, 8)
    "mlp_ratio": 4.0,                    # MLP expansion ratio in Swin blocks (2.0-4.0)
    
    # Other architecture settings
    "radius_emb_dim": 256,               # Dimension of radius/sigma embedding (128, 256)
    "window_size": 8,                    # Swin attention window size (4, 8, 16)
    "use_sfb": True,                     # Use Spatial-Frequency Block (SwinFIR)
    "dropout": 0.0,                      # Dropout rate
    "drop_path": 0.1,                    # Stochastic depth rate
    
    # -------------------------------------------------------------------------
    # EDM / Diffusion Parameters
    # -------------------------------------------------------------------------
    "sigma_min": 0.002,                  # Minimum sigma for sampling
    "sigma_max": 10.0,                   # Maximum sigma for sampling
    "sigma_data": 0.5,                   # Data standard deviation (for preconditioning)
    "P_mean": -1.2,                      # Log-normal mean for sigma sampling
    "P_std": 1.2,                        # Log-normal std for sigma sampling
    "rho": 7.0,                          # Time step discretization exponent
    "loss_type": "l2",                   # Loss: "l1", "l2", or "huber"
    
    # Classifier-Free Guidance (CFG)
    "cond_drop_prob": 0.0,               # Probability of dropping MRI conditioning (0.0 = no CFG)
    "cfg_scale": 1.0,                    # Guidance scale at inference (1.0 = no guidance)
    
    # -------------------------------------------------------------------------
    # Training Parameters
    # -------------------------------------------------------------------------
    "num_epochs": 100,                    # Total training epochs
    "batch_size": 5,                     # Batch size (reduce if OOM)
    "lr": 1e-4,                          # Learning rate
    "weight_decay": 0.01,                # AdamW weight decay
    "grad_clip": None,                    # Gradient clipping (0 or None to disable)
    "use_mixed_precision": False,        # Mixed precision training
    
    # -------------------------------------------------------------------------
    # Sampling / Generation
    # -------------------------------------------------------------------------
    "num_sample_steps": 50,              # ODE integration steps for sampling
    "num_val_samples": 4,                # Number of samples to visualize
    
    # -------------------------------------------------------------------------
    # Logging & Checkpointing
    # -------------------------------------------------------------------------
    "output_dir": "./outputs",           # Output directory
    "log_every": 1,                      # Log every N batches
    "val_every": 1,                      # Validate every N epochs
    "save_every": 5,                     # Save checkpoint every N epochs
    "sample_every": 1,                   # Generate samples every N epochs
    "resume": None,                      # Path to checkpoint to resume from
    
    # -------------------------------------------------------------------------
    # Gradient & ODE Monitoring
    # -------------------------------------------------------------------------
    "log_gradients": True,               # Log gradient flow statistics
    "log_ode_trajectory": True,          # Log ODE trajectory norms during sampling
    "grad_log_interval": 10,             # Log gradient stats every N steps
    "grad_log_num_params": 10,           # Number of parameter-level grads to log
    
    # -------------------------------------------------------------------------
    # System
    # -------------------------------------------------------------------------
    "seed": 42,                          # Random seed
    "num_workers": 3,                    # DataLoader workers
    "device": "cuda:3" if torch.cuda.is_available() else "cpu",
}

# =============================================================================
# Model Size Presets Reference
# =============================================================================
# "small":  base_channels=48,  channel_mults=[1,2,4],   num_res_blocks=1, num_heads=4  (~15M params)
# "base":   base_channels=64,  channel_mults=[1,2,4,8], num_res_blocks=2, num_heads=8  (~100M params)
# "large":  base_channels=96,  channel_mults=[1,2,4,8], num_res_blocks=3, num_heads=8  (~200M+ params)
#
# To reduce model size, try:
# - Reduce base_channels: 64 -> 48 -> 32
# - Fewer stages: [1,2,4,8] -> [1,2,4] -> [1,2]
# - Fewer blocks: num_res_blocks=2 -> 1
# - Fewer heads: num_heads=8 -> 4 -> 2
# - Smaller MLP: mlp_ratio=4.0 -> 2.0
# - Less attention: attention_resolutions=[4,8] -> [8] -> []
# =============================================================================


# =============================================================================
# Utility Functions
# =============================================================================

def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_data_files(data_path: str, mr_pattern: str = "mr.nii", ct_pattern: str = "ct.nii"):
    """Collect MR and CT file pairs from directory structure."""
    data_files = []
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    for patient_dir in sorted(os.listdir(data_path)):
        patient_path = os.path.join(data_path, patient_dir)
        if not os.path.isdir(patient_path):
            continue
        
        mr_file = None
        ct_file = None
        for filename in os.listdir(patient_path):
            if mr_pattern in filename:
                mr_file = os.path.join(patient_path, filename)
            if ct_pattern in filename:
                ct_file = os.path.join(patient_path, filename)
        
        if mr_file and ct_file:
            data_files.append({"mr": mr_file, "ct": ct_file, "id": patient_dir})
    
    return data_files


# =============================================================================
# PFGM++ / EDM Utilities
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


def log_ode_trajectory(writer: SummaryWriter, stats: dict, epoch: int):
    """
    Log ODE trajectory statistics for monitoring sampling quality.
    
    Args:
        writer: TensorBoard writer
        stats: Dictionary of trajectory stats from edm_sample
        epoch: Current epoch number
    """
    # Log trajectory norms
    writer.add_scalar("ODE/initial_norm", stats['initial_norm'], epoch)
    writer.add_scalar("ODE/final_norm", stats['final_norm'], epoch)
    writer.add_scalar("ODE/mean_denoised_norm", stats['mean_denoised_norm'], epoch)
    writer.add_scalar("ODE/mean_score_norm", stats['mean_score_norm'], epoch)
    writer.add_scalar("ODE/mean_update_norm", stats['mean_update_norm'], epoch)
    
    # Log trajectory progression
    for i, norm in enumerate(stats.get('traj_norms', [])):
        writer.add_scalar(f"ODE_traj/step_{i}_norm", norm, epoch)


def edm_precond(
    model: nn.Module,
    x: torch.Tensor,
    mri: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """
    Apply EDM preconditioning to model output.
    
    The model outputs D(x; sigma) which directly predicts the denoised image.
    EDM uses skip connection scaling: D = c_skip * x + c_out * F(c_in * x; c_noise)
    
    c_noise = ln(sigma) / 4 normalizes the conditioning input.
    """
    # Ensure all tensors are float32 for cuDNN compatibility
    x = x.float()
    mri = mri.float()
    sigma = sigma.float()
    
    # c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
    # c_out = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
    # c_in = 1 / sqrt(sigma^2 + sigma_data^2)
    # c_noise = ln(sigma) / 4 (normalized conditioning)
    
    sigma_4d = sigma.view(-1, 1, 1, 1)
    c_skip = sigma_data ** 2 / (sigma_4d ** 2 + sigma_data ** 2)
    c_out = sigma_4d * sigma_data / (sigma_4d ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma_4d ** 2 + sigma_data ** 2).sqrt()
    
    # c_noise: normalized sigma for conditioning (EDM standard)
    # This maps sigma range to roughly [-1.5, 1.1] for better embedding
    c_noise = sigma.log() / 4
    
    # Scale input
    x_scaled = c_in * x
    
    # Model predicts the "denoising" component with normalized sigma conditioning
    F_x = model(x_scaled, mri, c_noise)
    
    # Combine with skip connection
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
    """Sample sigma values according to EDM log-normal distribution, clamped to valid range."""
    rnd_normal = torch.randn(batch_size, device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    # Clamp to valid sampling range to ensure training covers the full range
    sigma = sigma.clamp(min=sigma_min, max=sigma_max)
    return sigma


def edm_loss(
    model: nn.Module,
    x_clean: torch.Tensor,
    mri: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
    loss_type: str = 'l2',
    cond_drop_prob: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute EDM-style loss with optional classifier-free guidance training.
    
    Loss = weight * ||D(x + n; sigma) - x||^2
    
    Where:
    - D is the denoiser network
    - n is Gaussian noise with std=sigma
    - weight = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
    
    Args:
        cond_drop_prob: Probability of dropping MRI conditioning (replacing with zeros).
                        Set to 0.0 to disable CFG training.
    
    Returns:
        loss: Scalar loss value
        metrics: Dictionary of metrics for logging
    """
    # Add noise
    n = torch.randn_like(x_clean) * sigma.view(-1, 1, 1, 1)
    x_noisy = x_clean + n
    
    # Classifier-free guidance: randomly drop conditioning
    if cond_drop_prob > 0.0 and model.training:
        # Create mask for which samples drop conditioning
        drop_mask = torch.rand(mri.shape[0], device=mri.device) < cond_drop_prob
        # Zero out MRI for dropped samples
        mri = mri.clone()
        mri[drop_mask] = 0.0
    
    # Get denoised prediction with preconditioning
    D_x = edm_precond(model, x_noisy, mri, sigma, sigma_data)
    
    # Compute weight (EDM weighting)
    # This gives equal contribution across log-sigma scale
    sigma_v = sigma.view(-1, 1, 1, 1)
    weight = (sigma_v ** 2 + sigma_data ** 2) / ((sigma_v * sigma_data) ** 2)
    
    # Optional: cap the weight to prevent low-sigma samples from dominating
    # This helps the model learn better at high sigma levels
    weight = weight.clamp(max=100.0)
    
    # Compute loss
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
    
    return loss, metrics


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
) -> torch.Tensor:
    """
    Generate CT images from MRI using EDM sampling (Algorithm 2 from EDM paper).
    
    This is a deterministic ODE sampler with optional stochasticity.
    
    Args:
        return_trajectory_stats: If True, returns (samples, stats_dict) with ODE trajectory info
        cfg_scale: Classifier-free guidance scale. 1.0 = no guidance, >1.0 = stronger conditioning.
                   Requires model trained with cond_drop_prob > 0.
    """
    B, C, H, W = mri.shape
    device = mri.device
    
    # Time step discretization
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    
    # Initialize from pure noise at sigma_max
    x_next = torch.randn(B, 1, H, W, device=device, dtype=torch.float64) * t_steps[0]
    
    # Track ODE trajectory statistics
    traj_norms = [x_next.norm().item()]
    denoised_norms = []
    score_norms = []
    update_norms = []
    
    # Main sampling loop
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Increase noise temporarily (stochasticity)
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        
        # Euler step
        sigma_batch = torch.full((B,), t_hat.item(), device=device, dtype=torch.float32)
        denoised_cond = edm_precond(model, x_hat.float(), mri, sigma_batch, sigma_data).double()
        
        # Classifier-free guidance
        if cfg_scale != 1.0:
            mri_uncond = torch.zeros_like(mri)
            denoised_uncond = edm_precond(model, x_hat.float(), mri_uncond, sigma_batch, sigma_data).double()
            denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
        else:
            denoised = denoised_cond
        
        d_cur = (x_hat - denoised) / t_hat  # Score function
        update = (t_next - t_hat) * d_cur
        x_next = x_hat + update
        
        # Apply 2nd order correction (Heun's method)
        if i < num_steps - 1 and t_next > 0:
            sigma_batch = torch.full((B,), t_next.item(), device=device, dtype=torch.float32)
            denoised_cond_next = edm_precond(model, x_next.float(), mri, sigma_batch, sigma_data).double()
            
            # CFG for Heun correction
            if cfg_scale != 1.0:
                denoised_uncond_next = edm_precond(model, x_next.float(), mri_uncond, sigma_batch, sigma_data).double()
                denoised_next = denoised_uncond_next + cfg_scale * (denoised_cond_next - denoised_uncond_next)
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
# Training Loop
# =============================================================================

def training_loop(cfg: dict, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
    """Main training loop."""
    device = torch.device(cfg["device"])
    
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["output_dir"]) / f"{timestamp}_pfgm_fir"
    log_dir = run_dir / "logs"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PFGM-FIR Training Pipeline")
    print(f"Run directory: {run_dir}")
    print(f"{'='*60}\n")
    
    # Build model
    print("Building model...")
    
    # Check if using custom config or preset
    if cfg["model_config"] == "custom":
        # Use custom architecture parameters from config
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
            use_sfb=cfg["use_sfb"],
            dropout=cfg["dropout"],
            drop_path_rate=cfg["drop_path"],
        )
        print(f"  Using CUSTOM architecture:")
        print(f"    base_channels: {cfg['base_channels']}")
        print(f"    channel_mults: {cfg['channel_mults']}")
        print(f"    num_res_blocks: {cfg['num_res_blocks']}")
        print(f"    attention_resolutions: {cfg['attention_resolutions']}")
        print(f"    num_heads: {cfg['num_heads']}")
    else:
        # Use preset config
        model = create_swin_fir_unet(
            config=cfg["model_config"],
            in_channels=2,
            out_channels=1,
            radius_emb_dim=cfg["radius_emb_dim"],
            window_size=cfg["window_size"],
            use_sfb=cfg["use_sfb"],
            dropout=cfg["dropout"],
            drop_path_rate=cfg["drop_path"],
        )
        print(f"  Using PRESET config: {cfg['model_config']}")
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model built:")
    print(f"  Total parameters: {total_params/1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
    )
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = min(1000, int(total_steps * 0.05))
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision
    scaler = GradScaler() if cfg["use_mixed_precision"] else None
    if scaler:
        print("✓ Mixed precision training enabled")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("Config", str(cfg))
    
    # Metrics
    if TORCHMETRICS_AVAILABLE:
        psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    else:
        psnr_metric = ssim_metric = mae_metric = None
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if cfg["resume"]:
        checkpoint_path = Path(cfg["resume"])
        if checkpoint_path.exists():
            print(f"Resuming from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"\nStarting training for {cfg['num_epochs']} epochs...")
    print(f"Training batches per epoch: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches per epoch: {len(val_loader)}")
    print(f"{'='*60}\n")
    
    global_step = start_epoch * len(train_loader)
    
    # ==================== Training Loop ====================
    for epoch in range(start_epoch, cfg["num_epochs"]):
        epoch_start = time.time()
        model.train()
        
        epoch_loss = 0.0
        num_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['num_epochs']} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            mr = batch["mr"].to(device, non_blocking=True)
            ct = batch["ct"].to(device, non_blocking=True)
            
            # Debug shapes on first batch
            if epoch == 0 and batch_idx == 0:
                print(f"\n[Debug] Tensor shapes - MR: {mr.shape}, CT: {ct.shape}\n")
            
            batch_size = mr.shape[0]
            
            # Sample sigma from log-normal distribution (clamped to sampling range)
            sigma = sample_sigma(
                batch_size=batch_size,
                P_mean=cfg["P_mean"],
                P_std=cfg["P_std"],
                sigma_min=cfg["sigma_min"],
                sigma_max=cfg["sigma_max"],
                device=device,
            )
            
            optimizer.zero_grad()
            
            # Forward pass
            with autocast(enabled=cfg["use_mixed_precision"]):
                loss, metrics = edm_loss(
                    model, ct, mr, sigma,
                    sigma_data=cfg["sigma_data"],
                    loss_type=cfg["loss_type"],
                    cond_drop_prob=cfg.get("cond_drop_prob", 0.0),
                )
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
                if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
            
            # Check gradient flow periodically
            if cfg.get("log_gradients", False) and batch_idx % (cfg["log_every"] * 10) == 0:
                grad_info = check_gradient_flow(model.named_parameters())
                
                # Log gradient statistics
                total_grad_norm = 0.0
                num_layers_with_grad = 0
                has_nan_grads = False
                
                for name, info in grad_info.items():
                    if info['has_nan']:
                        has_nan_grads = True
                        print(f"  [WARNING] NaN gradient in {name}")
                    total_grad_norm += info['mean']
                    num_layers_with_grad += 1
                
                if num_layers_with_grad > 0:
                    avg_grad_norm = total_grad_norm / num_layers_with_grad
                    writer.add_scalar("Gradient/avg_norm", avg_grad_norm, global_step)
                    
                if has_nan_grads:
                    writer.add_scalar("Gradient/has_nan", 1.0, global_step)
                    print(f"  [WARNING] NaN gradients detected at step {global_step}")
                else:
                    writer.add_scalar("Gradient/has_nan", 0.0, global_step)
                
                # Log field norm (prediction magnitude)
                writer.add_scalar("Field/pred_std", metrics['pred_std'], global_step)
                writer.add_scalar("Field/target_std", metrics['target_std'], global_step)
            
            scheduler.step()
            
            # Accumulate loss
            loss_val = loss.item()
            epoch_loss += loss_val * batch_size
            num_samples += batch_size
            
            # Progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                "loss": f"{loss_val:.3e}",
                "lr": f"{current_lr:.2e}"
            })
            
            # Logging
            if batch_idx % cfg["log_every"] == 0:
                writer.add_scalar("Train/Loss_step", loss_val, global_step)
                writer.add_scalar("Train/LR", current_lr, global_step)
                writer.add_scalar("Train/MSE", metrics['mse'], global_step)
                writer.add_scalar("Train/sigma_mean", metrics['sigma_mean'], global_step)
                writer.add_scalar("Train/pred_std", metrics['pred_std'], global_step)
            
            global_step += 1
        
        # Epoch statistics
        avg_train_loss = epoch_loss / num_samples
        epoch_time = time.time() - epoch_start
        
        writer.add_scalar("Train/Loss_epoch", avg_train_loss, epoch)
        writer.add_scalar("System/Epoch_time", epoch_time, epoch)
        if torch.cuda.is_available():
            writer.add_scalar("System/GPU_memory_MB", 
                            torch.cuda.max_memory_allocated(device)/1024**2, epoch)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Loss: {avg_train_loss:.3e}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # ==================== Sample Generation ====================
        if (epoch + 1) % cfg["sample_every"] == 0:
            print("  Generating samples...")
            model.eval()
            
            with torch.no_grad():
                sample_batch = next(iter(train_loader))
                mr_samples = sample_batch["mr"][:cfg["num_val_samples"]].to(device)
                ct_samples = sample_batch["ct"][:cfg["num_val_samples"]].to(device)
                
                # Generate using EDM sampler with trajectory tracking
                log_trajectory = cfg.get("log_ode_trajectory", False)
                result = edm_sample(
                    model, mr_samples,
                    num_steps=cfg["num_sample_steps"],
                    sigma_min=cfg["sigma_min"],
                    sigma_max=cfg["sigma_max"],
                    sigma_data=cfg["sigma_data"],
                    rho=cfg["rho"],
                    return_trajectory_stats=log_trajectory,
                    cfg_scale=cfg.get("cfg_scale", 1.0),
                )
                
                if log_trajectory:
                    pred_ct, traj_stats = result
                    log_ode_trajectory(writer, traj_stats, epoch)
                    print(f"  ODE: init_norm={traj_stats['initial_norm']:.2f}, "
                          f"final_norm={traj_stats['final_norm']:.2f}, "
                          f"mean_score={traj_stats['mean_score_norm']:.4f}")
                else:
                    pred_ct = result
                
                # Clamp to valid range
                pred_ct = torch.clamp(pred_ct, -1.0, 1.0)
                
                # Convert to [0, 1] for metrics
                pred_ct_01 = (pred_ct + 1.0) / 2.0
                ct_samples_01 = (ct_samples + 1.0) / 2.0
                
                # Compute metrics
                if TORCHMETRICS_AVAILABLE:
                    mae_val = mae_metric(pred_ct_01, ct_samples_01)
                    psnr_val = psnr_metric(pred_ct_01, ct_samples_01)
                    ssim_val = ssim_metric(pred_ct_01, ct_samples_01)
                    
                    writer.add_scalar("Gen/MAE", mae_val, epoch)
                    writer.add_scalar("Gen/PSNR", psnr_val, epoch)
                    writer.add_scalar("Gen/SSIM", ssim_val, epoch)
                    
                    print(f"  ✓ Generation: MAE={mae_val:.5f}, PSNR={psnr_val:.3f}, SSIM={ssim_val:.5f}")
                
                # Save images
                img_grid = vutils.make_grid(
                    torch.cat([mr_samples, pred_ct, ct_samples], dim=0),
                    nrow=cfg["num_val_samples"],
                    normalize=True,
                    scale_each=True
                )
                writer.add_image("Gen/MRI_Pred_GT", img_grid, epoch)
        
        # ==================== Validation ====================
        if val_loader is not None and (epoch + 1) % cfg["val_every"] == 0:
            print("  Running validation...")
            model.eval()
            
            val_loss_accum = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    mr = batch["mr"].to(device)
                    ct = batch["ct"].to(device)
                    
                    sigma = sample_sigma(
                        batch_size=ct.shape[0],
                        P_mean=cfg["P_mean"],
                        P_std=cfg["P_std"],
                        sigma_min=cfg["sigma_min"],
                        sigma_max=cfg["sigma_max"],
                        device=device,
                    )
                    
                    loss, _ = edm_loss(
                        model, ct, mr, sigma,
                        sigma_data=cfg["sigma_data"],
                        loss_type=cfg["loss_type"],
                        cond_drop_prob=0.0,  # No CFG dropout during validation
                    )
                    val_loss_accum += loss.item() * ct.shape[0]
                    val_samples += ct.shape[0]
            
            avg_val_loss = val_loss_accum / val_samples
            writer.add_scalar("Val/Loss", avg_val_loss, epoch)
            print(f"  ✓ Validation: loss={avg_val_loss:.3e}")
            
            # Validation sample generation
            if TORCHMETRICS_AVAILABLE:
                val_sample_batch = next(iter(val_loader))
                mr_val_samples = val_sample_batch["mr"][:cfg["num_val_samples"]].to(device)
                ct_val_samples = val_sample_batch["ct"][:cfg["num_val_samples"]].to(device)
                
                # Generate CT predictions from validation MRI
                pred_val_ct = edm_sample(
                    model, mr_val_samples,
                    num_steps=cfg["num_sample_steps"],
                    sigma_min=cfg["sigma_min"],
                    sigma_max=cfg["sigma_max"],
                    sigma_data=cfg["sigma_data"],
                    rho=cfg["rho"],
                    cfg_scale=cfg.get("cfg_scale", 1.0),
                )
                
                # Clamp to valid range
                pred_val_ct = torch.clamp(pred_val_ct, -1.0, 1.0)
                
                # Convert to [0, 1] for metrics
                pred_val_ct_01 = (pred_val_ct + 1.0) / 2.0
                ct_val_samples_01 = (ct_val_samples + 1.0) / 2.0
                
                # Compute validation generation metrics
                val_mae = mae_metric(pred_val_ct_01, ct_val_samples_01)
                val_psnr = psnr_metric(pred_val_ct_01, ct_val_samples_01)
                val_ssim = ssim_metric(pred_val_ct_01, ct_val_samples_01)
                
                writer.add_scalar("Val_Gen/MAE", val_mae, epoch)
                writer.add_scalar("Val_Gen/PSNR", val_psnr, epoch)
                writer.add_scalar("Val_Gen/SSIM", val_ssim, epoch)
                
                print(f"  ✓ Val Generation: MAE={val_mae:.5f}, PSNR={val_psnr:.3f}, SSIM={val_ssim:.5f}")
                
                # Save validation images
                val_img_grid = vutils.make_grid(
                    torch.cat([mr_val_samples, pred_val_ct, ct_val_samples], dim=0),
                    nrow=cfg["num_val_samples"],
                    normalize=True,
                    scale_each=True
                )
                writer.add_image("Val_Gen/MRI_Pred_GT", val_img_grid, epoch)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": cfg,
                }
                torch.save(checkpoint, checkpoint_dir / "checkpoint_best.pt")
                print(f"  ✓ New best model saved!")
        
        # ==================== Save Checkpoint ====================
        if (epoch + 1) % cfg["save_every"] == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "config": cfg,
            }
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch{epoch+1:04d}.pt")
            torch.save(checkpoint, checkpoint_dir / "checkpoint_latest.pt")
            print(f"  ✓ Checkpoint saved")
        
        print()
    
    writer.close()
    print(f"\n{'='*60}")
    print(f"Training complete! Results saved to: {run_dir}")
    if best_val_loss != float('inf'):
        print(f"Best validation loss: {best_val_loss:.3e}")
    print(f"{'='*60}\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    # Print environment diagnostics
    print(f"\n{'='*60}")
    print("Environment Diagnostics:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Disable cuDNN due to CUDA version mismatch
        # PyTorch built with CUDA 11.6, system has CUDA 12.4
        # This causes "Unable to find valid cuDNN algorithm" errors
        torch.backends.cudnn.enabled = False
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled} (disabled for compatibility)")
    print(f"{'='*60}\n")
    
    seed_everything(config["seed"])
    
    # Setup data
    print("Setting up datasets...")
    all_files = collect_data_files(
        config["data_dir"],
        mr_pattern=config["mr_pattern"],
        ct_pattern=config["ct_pattern"]
    )
    
    if len(all_files) == 0:
        raise FileNotFoundError(f"No data found at {config['data_dir']}")
    
    print(f"Found {len(all_files)} patient volumes")
    
    # Train/val split
    random.shuffle(all_files)
    split_idx = max(1, int(len(all_files) * config["train_split"]))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:] if len(all_files) > split_idx else None
    
    print(f"Split: {len(train_files)} train, {len(val_files) if val_files else 0} val patients")
    
    # Create datasets
    train_transforms = get_monai_nifti_transforms(mode="train", target_size=config["image_size"])
    train_dataset = MonaiNiftiSliceDataset(
        data=train_files,
        transform=train_transforms,
        target_size=config["image_size"]
    )
    
    val_dataset = None
    if val_files:
        val_transforms = get_monai_nifti_transforms(mode="val", target_size=config["image_size"])
        val_dataset = MonaiNiftiSliceDataset(
            data=val_files,
            transform=val_transforms,
            target_size=config["image_size"]
        )
    
    print(f"Dataset sizes: {len(train_dataset)} train slices, "
          f"{len(val_dataset) if val_dataset else 0} val slices")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=False
        )
    
    # Start training
    training_loop(config, train_loader, val_loader)


if __name__ == '__main__':
    main()
