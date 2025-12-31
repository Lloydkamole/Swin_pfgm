"""
PFGM-FIR Training Script with EDM2 Magnitude-Preserving Architecture.

Multi-GPU training with Hugging Face Accelerate and EDM2 improvements.

Key differences from train_accelerate.py:
- Uses MPSwinFIRUNet instead of SwinFIRUNet
- Implements magnitude-preserving training principles
- Controlled learning rate decay per EDM2 paper

Usage:
    # Single GPU:
    python train_mp_accelerate.py
    
    # Multi-GPU with accelerate:
    accelerate launch --num_processes 2 --gpu_ids 0,1 train_mp_accelerate.py
    
    # Or with config file:
    accelerate launch --config_file accelerate_config.yaml train_mp_accelerate.py
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

# Import MP model
from models.mp_network import create_mp_swinfir_unet
from models.mp_layers import normalize

# Compatibility fix for Pillow >= 10
try:
    from PIL import Image as _PILImage
    if not hasattr(_PILImage, "ANTIALIAS"):
        _PILImage.ANTIALIAS = _PILImage.Resampling.LANCZOS
except Exception:
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

config = {
    # Data Settings
    "data_dir": "/home/lloyd/Documents/AR PFGM/pfgm_train",
    "mr_pattern": "mr.nii",
    "ct_pattern": "ct.nii",
    "train_split": 0.9,
    "image_size": 256,
    
    # Model Architecture (EDM2-style)
    "model_config": "custom",
    "base_channels": 64,
    "channel_mults": [1, 2, 4, 8],
    "num_res_blocks": 2,
    "attention_resolutions": [2, 4, 8],
    "num_heads": 4,
    "mlp_ratio": 4.0,
    "window_size": 4,
    "use_sfb": True,
    "dropout": 0.0,
    "drop_path": 0.1,
    
    # EDM Parameters
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "sigma_data": 0.5,
    "P_mean": -1.2,
    "P_std": 1.2,
    "rho": 7.0,
    "loss_type": "l2",
    
    # Classifier-Free Guidance (CFG)
    "cond_drop_prob": 0.2,
    "cfg_scale": 2.0,
    
    # Masked Loss
    "use_masked_loss": True,
    "masked_loss_ratio": 0.5,
    "mask_threshold": -0.7,
    
    # Training Parameters (EDM2-style)
    "num_epochs": 100,
    "batch_size": 5,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "grad_clip": None,
    "warmup_steps": 500,
    "lr_decay_end": 0.01,  # Controlled LR decay per EDM2 (decay to 1% of peak)
    
    # Sampling / Generation
    "num_sample_steps": 50,
    "num_val_samples": 4,
    
    # Logging & Checkpointing
    "output_dir": "./outputs_mp_accelerate",
    "log_every": 5,
    "val_every": 1,
    "save_every": 5,
    "sample_every": 1,
    "resume": None,
    
    # Gradient & ODE Monitoring
    "log_gradients": True,
    "log_ode_trajectory": True,
    "grad_log_interval": 100,
    "log_weight_norms": True,  # EDM2: monitor magnitude preservation
    
    # System
    "seed": 42,
    "num_workers": 4,
    "mixed_precision": "no",  # "no", "fp16", or "bf16"
}


# =============================================================================
# Utility Functions
# =============================================================================

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


def check_weight_magnitude(named_parameters):
    """
    Check weight magnitude preservation (EDM2-specific monitoring).
    
    Returns statistics on weight norms to verify magnitude preservation.
    """
    weight_info = {}
    for name, param in named_parameters:
        if param.requires_grad and 'weight' in name:
            # Check if weights are approximately unit norm
            # (per output channel for convolutions)
            weight_norm = torch.linalg.vector_norm(
                param.view(param.size(0), -1), 
                dim=1
            ).mean().item()
            
            weight_info[name] = {
                'mean_norm': weight_norm,
                'expected': 1.0,
                'deviation': abs(weight_norm - 1.0),
            }
    return weight_info


# =============================================================================
# EDM Functions (adapted for MP model)
# =============================================================================

def edm_precond_mp(
    model: nn.Module,
    x: torch.Tensor,
    mri: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """
    Apply EDM preconditioning to MP model output.
    
    Note: MP model uses radius (sigma) directly, not log(sigma)/4.
    """
    x = x.float()
    mri = mri.float()
    sigma = sigma.float()
    
    sigma_4d = sigma.view(-1, 1, 1, 1)
    c_skip = sigma_data ** 2 / (sigma_4d ** 2 + sigma_data ** 2)
    c_out = sigma_4d * sigma_data / (sigma_4d ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma_4d ** 2 + sigma_data ** 2).sqrt()
    
    # Concatenate inputs: [noisy_ct, mri_condition]
    x_scaled = c_in * x
    x_input = torch.cat([x_scaled, mri], dim=1)
    
    # Model takes (input, radius) - radius used directly in embedding
    F_x = model(x_input, sigma)
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


def edm_loss_mp(
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
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute EDM-style loss for MP model.
    
    Includes optional classifier-free guidance training and masked loss.
    """
    n = torch.randn_like(x_clean) * sigma.view(-1, 1, 1, 1)
    x_noisy = x_clean + n
    
    # Classifier-free guidance: randomly drop conditioning
    if cond_drop_prob > 0.0 and model.training:
        drop_mask = torch.rand(mri.shape[0], device=mri.device) < cond_drop_prob
        mri = mri.clone()
        mri[drop_mask] = 0.0
    
    D_x = edm_precond_mp(model, x_noisy, mri, sigma, sigma_data)
    
    sigma_v = sigma.view(-1, 1, 1, 1)
    weight = (sigma_v ** 2 + sigma_data ** 2) / ((sigma_v * sigma_data) ** 2)
    weight = weight.clamp(max=100.0)
    
    # Generate body mask if using masked loss
    body_mask = None
    use_mask_this_batch = False
    if use_masked_loss and model.training:
        if torch.rand(1).item() < masked_loss_ratio:
            use_mask_this_batch = True
            body_mask = (x_clean > mask_threshold).float()
            weight = weight * body_mask
    
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
    
    if use_mask_this_batch and body_mask is not None:
        metrics['mask_ratio'] = body_mask.mean().item()
        metrics['masked'] = 1.0
    else:
        metrics['masked'] = 0.0
    
    return loss, metrics


@torch.no_grad()
def edm_sample_mp(
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
    Generate CT images from MRI using EDM sampling with MP model.
    
    Args:
        return_trajectory_stats: If True, returns (samples, stats_dict)
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
    """
    B, C, H, W = mri.shape
    device = mri.device
    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    
    x_next = torch.randn(B, 1, H, W, device=device, dtype=torch.float64) * t_steps[0]
    
    # Track ODE trajectory statistics
    traj_norms = [x_next.norm().item()]
    denoised_norms = []
    score_norms = []
    update_norms = []
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Increase noise temporarily (stochasticity)
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        
        sigma_batch = torch.full((B,), t_hat.item(), device=device, dtype=torch.float32)
        denoised_cond = edm_precond_mp(model, x_hat.float(), mri, sigma_batch, sigma_data).double()
        
        # Classifier-free guidance
        if cfg_scale != 1.0:
            mri_uncond = torch.zeros_like(mri)
            denoised_uncond = edm_precond_mp(model, x_hat.float(), mri_uncond, sigma_batch, sigma_data).double()
            denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
        else:
            denoised = denoised_cond
        
        d_cur = (x_hat - denoised) / t_hat
        update = (t_next - t_hat) * d_cur
        x_next = x_hat + update
        
        if i < num_steps - 1 and t_next > 0:
            sigma_batch = torch.full((B,), t_next.item(), device=device, dtype=torch.float32)
            denoised_cond_next = edm_precond_mp(model, x_next.float(), mri, sigma_batch, sigma_data).double()
            
            if cfg_scale != 1.0:
                denoised_uncond_next = edm_precond_mp(model, x_next.float(), mri_uncond, sigma_batch, sigma_data).double()
                denoised_next = denoised_uncond_next + cfg_scale * (denoised_cond_next - denoised_uncond_next)
            else:
                denoised_next = denoised_cond_next
            
            d_prime = (x_next - denoised_next) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        # Track statistics
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
    """Main training function with Accelerate and MP model."""
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg["output_dir"]) / f"{timestamp}_pfgm_fir_mp"
    
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
        print(f"PFGM-FIR Training with EDM2 Magnitude-Preserving Architecture")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Mixed precision: {cfg['mixed_precision']}")
        print(f"{'='*60}\n")
    
    # Build MP model
    model = create_mp_swinfir_unet(
        img_size=cfg["image_size"],
        in_channels=2,
        out_channels=1,
        base_channels=cfg["base_channels"],
        channel_mults=tuple(cfg["channel_mults"]),
        num_res_blocks=cfg["num_res_blocks"],
        attention_resolutions=tuple(cfg["attention_resolutions"]),
        num_heads=cfg["num_heads"],
        window_size=cfg["window_size"],
        use_sfb=cfg["use_sfb"],
        dropout=cfg["dropout"],
        drop_path_rate=cfg["drop_path"],
    )
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params/1e6:.2f}M ({trainable_params/1e6:.2f}M trainable)")
        print(f"  Architecture config:")
        print(f"    base_channels: {cfg['base_channels']}")
        print(f"    channel_mults: {cfg['channel_mults']}")
        print(f"    num_res_blocks: {cfg['num_res_blocks']}")
        print(f"    attention_resolutions: {cfg['attention_resolutions']}")
        print(f"    EDM2 features: magnitude-preserving, no GroupNorm, weight normalization")
    
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
    
    # Data loading
    if accelerator.is_main_process:
        print("Loading data files...")
    
    all_files = collect_data_files(
        cfg["data_dir"],
        mr_pattern=cfg["mr_pattern"],
        ct_pattern=cfg["ct_pattern"]
    )
    
    random.shuffle(all_files)
    split_idx = max(1, int(len(all_files) * cfg["train_split"]))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:] if len(all_files) > split_idx else None
    
    if accelerator.is_main_process:
        print(f"Found {len(train_files)} train, {len(val_files) if val_files else 0} val patients")
        print("Creating datasets...")
    
    train_transforms = get_monai_nifti_transforms(mode="train", target_size=cfg["image_size"])
    train_dataset = MonaiNiftiSliceDataset(
        data=train_files,
        transform=train_transforms,
        target_size=cfg["image_size"]
    )
    
    if accelerator.is_main_process:
        print(f"Train dataset: {len(train_dataset)} slices")
    
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
        val_transforms = get_monai_nifti_transforms(mode="val", target_size=cfg["image_size"])
        val_dataset = MonaiNiftiSliceDataset(
            data=val_files,
            transform=val_transforms,
            target_size=cfg["image_size"]
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
    
    # Learning rate scheduler with controlled decay (EDM2 style)
    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = cfg.get("warmup_steps", 500)
    lr_decay_end = cfg.get("lr_decay_end", 0.01)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Controlled decay: from 1.0 to lr_decay_end (e.g., 0.01 = 1% of peak LR)
        return max(lr_decay_end, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    # For non-FSDP, create scheduler before prepare
    scheduler = None
    if not is_fsdp:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Prepare with Accelerator
    if accelerator.is_main_process:
        print("Preparing model and data with Accelerator...")
    
    if is_fsdp:
        # FSDP path: prepare model first, then create optimizer
        model = accelerator.prepare(model)
        optimizer = AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.95),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        optimizer, scheduler, train_loader = accelerator.prepare(optimizer, scheduler, train_loader)
        if val_loader:
            val_loader = accelerator.prepare(val_loader)
    else:
        # Standard path
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
        if val_loader:
            val_loader = accelerator.prepare(val_loader)
    
    # Initialize TensorBoard
    accelerator.init_trackers("pfgm_fir_mp")
    
    # Metrics
    if TORCHMETRICS_AVAILABLE and accelerator.is_main_process:
        psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(accelerator.device)
        ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(accelerator.device)
        mae_metric = torchmetrics.MeanAbsoluteError().to(accelerator.device)
    else:
        psnr_metric = ssim_metric = mae_metric = None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if cfg["resume"]:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {cfg['resume']}")
        checkpoint = torch.load(cfg["resume"], map_location=accelerator.device)
        accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        if accelerator.is_main_process:
            print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training loop
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60 + "\n")
    
    for epoch in range(start_epoch, cfg["num_epochs"]):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        
        if accelerator.is_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")
        else:
            pbar = train_loader
        
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                # Get data
                mri = batch["mr"]
                ct = batch["ct"]
                
                # Sample noise level
                sigma = sample_sigma(
                    ct.shape[0],
                    P_mean=cfg["P_mean"],
                    P_std=cfg["P_std"],
                    sigma_min=cfg["sigma_min"],
                    sigma_max=cfg["sigma_max"],
                    device=ct.device,
                )
                
                # Compute loss
                loss, metrics = edm_loss_mp(
                    model,
                    ct,
                    mri,
                    sigma,
                    sigma_data=cfg["sigma_data"],
                    loss_type=cfg["loss_type"],
                    cond_drop_prob=cfg["cond_drop_prob"],
                    use_masked_loss=cfg["use_masked_loss"],
                    masked_loss_ratio=cfg["masked_loss_ratio"],
                    mask_threshold=cfg["mask_threshold"],
                )
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if cfg["grad_clip"] is not None:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Gather loss across all processes for distributed training
            gathered_loss = accelerator.gather(loss.detach()).mean()
            
            # Accumulate metrics
            epoch_loss += gathered_loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            
            # Logging
            if global_step % cfg["log_every"] == 0:
                current_lr = scheduler.get_last_lr()[0]
                
                log_dict = {
                    "train/loss": gathered_loss.item(),
                    "train/lr": current_lr,
                    "train/epoch": epoch,
                }
                log_dict.update({f"train/{k}": v for k, v in metrics.items()})
                
                accelerator.log(log_dict, step=global_step)
                
                # Update progress bar (only main process)
                if accelerator.is_main_process:
                    pbar.set_postfix({
                        'loss': f"{gathered_loss.item():.3e}",
                        'lr': f"{current_lr:.2e}",
                        'sigma': f"{metrics['sigma_mean']:.3f}",
                    })
            
            # Gradient flow monitoring
            if cfg["log_gradients"] and global_step % cfg["grad_log_interval"] == 0 and accelerator.is_main_process:
                grad_info = check_gradient_flow(accelerator.unwrap_model(model).named_parameters())
                # Log only mean gradients for key layers
                grad_log = {}
                for name, info in grad_info.items():
                    if any(x in name for x in ['input_conv', 'output_conv', 'bottleneck']):
                        grad_log[f"gradients/{name}_mean"] = info['mean']
                
                if grad_log:
                    accelerator.log(grad_log, step=global_step)
            
            # Weight norm monitoring (EDM2-specific)
            if cfg.get("log_weight_norms", False) and global_step % cfg["grad_log_interval"] == 0 and accelerator.is_main_process:
                weight_info = check_weight_magnitude(accelerator.unwrap_model(model).named_parameters())
                # Log deviation from unit norm for key layers
                weight_log = {}
                for name, info in weight_info.items():
                    if any(x in name for x in ['input_conv', 'output_conv']):
                        weight_log[f"weight_norms/{name}_deviation"] = info['deviation']
                
                if weight_log:
                    accelerator.log(weight_log, step=global_step)
            
            global_step += 1
        
        # End of epoch
        avg_loss = epoch_loss / len(train_loader)
        avg_metrics = {k: v / len(train_loader) for k, v in epoch_metrics.items()}
        
        # Log epoch stats
        epoch_time = time.time() - epoch_start if 'epoch_start' in locals() else 0
        accelerator.log({
            "train/epoch_loss": avg_loss,
            "system/epoch_time": epoch_time,
        }, step=epoch)
        
        # Log GPU memory if available
        if torch.cuda.is_available() and accelerator.is_main_process:
            gpu_mem_mb = torch.cuda.max_memory_allocated(accelerator.device) / 1024**2
            accelerator.log({"system/gpu_memory_MB": gpu_mem_mb}, step=epoch)
        
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Avg Loss: {avg_loss:.3e}")
            print(f"  Avg MSE: {avg_metrics.get('mse', 0):.4f}")
            print(f"  Sigma Mean: {avg_metrics.get('sigma_mean', 0):.3f}")
            if 'masked' in avg_metrics:
                print(f"  Masked Ratio: {avg_metrics.get('masked', 0):.2%}")
        
        # Training sample generation
        if (epoch + 1) % cfg["sample_every"] == 0:
            model.eval()
            
            # Get training batch for sampling
            train_sample_batch = next(iter(train_loader))
            mri_train_samples = train_sample_batch["mr"][:cfg["num_val_samples"]]
            ct_train_samples = train_sample_batch["ct"][:cfg["num_val_samples"]]
            
            # Only main process does generation and logging
            if accelerator.is_main_process:
                with torch.no_grad():
                    # Get unwrapped model for sampling
                    unwrapped_model = accelerator.unwrap_model(model)
                    
                    # Generate with optional ODE trajectory logging
                    if cfg.get("log_ode_trajectory", False):
                        pred_train_ct, ode_stats = edm_sample_mp(
                            unwrapped_model, mri_train_samples,
                            num_steps=cfg["num_sample_steps"],
                            sigma_min=cfg["sigma_min"],
                            sigma_max=cfg["sigma_max"],
                            sigma_data=cfg["sigma_data"],
                            rho=cfg["rho"],
                            return_trajectory_stats=True,
                            cfg_scale=cfg.get("cfg_scale", 1.0),
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
                        pred_train_ct = edm_sample_mp(
                            unwrapped_model, mri_train_samples,
                            num_steps=cfg["num_sample_steps"],
                            sigma_min=cfg["sigma_min"],
                            sigma_max=cfg["sigma_max"],
                            sigma_data=cfg["sigma_data"],
                            rho=cfg["rho"],
                            cfg_scale=cfg.get("cfg_scale", 1.0),
                        )
                    
                    pred_train_ct = torch.clamp(pred_train_ct, -1.0, 1.0)
                    pred_train_ct_01 = (pred_train_ct + 1.0) / 2.0
                    ct_train_samples_01 = (ct_train_samples + 1.0) / 2.0
                    
                    if psnr_metric is not None:
                        psnr_val = psnr_metric(pred_train_ct_01, ct_train_samples_01)
                        ssim_val = ssim_metric(pred_train_ct_01, ct_train_samples_01)
                        mae_val = mae_metric(pred_train_ct_01, ct_train_samples_01) if mae_metric else None
                        
                        log_dict = {
                            "gen/psnr": psnr_val.item(),
                            "gen/ssim": ssim_val.item(),
                        }
                        if mae_val is not None:
                            log_dict["gen/mae"] = mae_val.item()
                        
                        accelerator.log(log_dict, step=epoch)
                        mae_str = f", MAE={mae_val:.5f}" if mae_val else ""
                        print(f"  Generation: PSNR={psnr_val:.3f}, SSIM={ssim_val:.5f}{mae_str}")
                    
                    # Log training sample images to TensorBoard
                    train_img_grid = vutils.make_grid(
                        torch.cat([mri_train_samples.cpu(), pred_train_ct.cpu(), ct_train_samples.cpu()], dim=0),
                        nrow=cfg["num_val_samples"],
                        normalize=True,
                        scale_each=True
                    )
                    tb_tracker = accelerator.get_tracker("tensorboard")
                    if tb_tracker is not None:
                        tb_tracker.writer.add_image("Gen/MRI_Pred_GT", train_img_grid, epoch)
            
            model.train()
        
        # Validation
        if val_loader and (epoch + 1) % cfg["val_every"] == 0:
            model.eval()
            val_loss = 0.0
            val_metrics = {}
            
            with torch.no_grad():
                for batch in val_loader:
                    mri = batch["mr"]
                    ct = batch["ct"]
                    
                    sigma = sample_sigma(
                        ct.shape[0],
                        P_mean=cfg["P_mean"],
                        P_std=cfg["P_std"],
                        sigma_min=cfg["sigma_min"],
                        sigma_max=cfg["sigma_max"],
                        device=ct.device,
                    )
                    
                    loss, metrics = edm_loss_mp(
                        model,
                        ct,
                        mri,
                        sigma,
                        sigma_data=cfg["sigma_data"],
                        loss_type=cfg["loss_type"],
                        cond_drop_prob=0.0,  # No dropout during validation
                        use_masked_loss=False,  # Full image for validation
                    )
                    
                    # Gather loss across all processes
                    gathered_loss = accelerator.gather(loss.detach()).mean()
                    val_loss += gathered_loss.item()
                    for k, v in metrics.items():
                        val_metrics[k] = val_metrics.get(k, 0.0) + v
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}
            
            if accelerator.is_main_process:
                print(f"\nValidation:")
                print(f"  Val Loss: {avg_val_loss:.3e}")
                print(f"  Val MSE: {avg_val_metrics.get('mse', 0):.4f}")
                
                val_log = {
                    "val/loss": avg_val_loss,
                }
                val_log.update({f"val/{k}": v for k, v in avg_val_metrics.items()})
                accelerator.log(val_log, step=global_step)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if accelerator.is_main_process:
                    best_checkpoint_path = output_dir / "checkpoints" / "best.pt"
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'config': cfg,
                        'val_loss': best_val_loss,
                    }
                    torch.save(checkpoint, best_checkpoint_path)
                    print(f"  New best model saved! Val loss: {best_val_loss:.4f}")
            
            model.train()
        
        # Sample generation (validation)
        if val_loader and (epoch + 1) % cfg["sample_every"] == 0:
            model.eval()
            
            with torch.no_grad():
                # Get a batch for sampling
                sample_batch = next(iter(val_loader))
                mri_sample = sample_batch["mr"][:cfg["num_val_samples"]]
                ct_sample = sample_batch["ct"][:cfg["num_val_samples"]]
                
                if accelerator.is_main_process:
                    # Get unwrapped model for sampling
                    unwrapped_model = accelerator.unwrap_model(model)
                    
                    # Generate with ODE trajectory stats
                    ct_pred, traj_stats = edm_sample_mp(
                        unwrapped_model,
                        mri_sample,
                        num_steps=cfg["num_sample_steps"],
                        sigma_min=cfg["sigma_min"],
                        sigma_max=cfg["sigma_max"],
                        sigma_data=cfg["sigma_data"],
                        rho=cfg["rho"],
                        return_trajectory_stats=True,
                        cfg_scale=cfg["cfg_scale"],
                    )
                    
                    # Log ODE trajectory stats
                    if cfg["log_ode_trajectory"]:
                        ode_log = {
                            "ode/initial_norm": traj_stats['initial_norm'],
                            "ode/final_norm": traj_stats['final_norm'],
                            "ode/mean_denoised_norm": traj_stats['mean_denoised_norm'],
                            "ode/mean_score_norm": traj_stats['mean_score_norm'],
                            "ode/mean_update_norm": traj_stats['mean_update_norm'],
                        }
                        accelerator.log(ode_log, step=epoch)
                    
                    # Compute generation metrics
                    ct_pred = torch.clamp(ct_pred, -1.0, 1.0)
                    ct_pred_01 = (ct_pred + 1.0) / 2.0
                    ct_sample_01 = (ct_sample + 1.0) / 2.0
                    
                    if psnr_metric is not None:
                        psnr_val = psnr_metric(ct_pred_01, ct_sample_01)
                        ssim_val = ssim_metric(ct_pred_01, ct_sample_01)
                        mae_val = mae_metric(ct_pred_01, ct_sample_01) if mae_metric else None
                        
                        log_dict = {
                            "val_gen/psnr": psnr_val.item(),
                            "val_gen/ssim": ssim_val.item(),
                        }
                        if mae_val is not None:
                            log_dict["val_gen/mae"] = mae_val.item()
                        
                        accelerator.log(log_dict, step=epoch)
                        mae_str = f", MAE={mae_val:.5f}" if mae_val else ""
                        print(f"  Val Generation: PSNR={psnr_val:.3f}, SSIM={ssim_val:.5f}{mae_str}")
                    
                    # Log validation sample images to TensorBoard
                    val_img_grid = vutils.make_grid(
                        torch.cat([mri_sample.cpu(), ct_pred.cpu(), ct_sample.cpu()], dim=0),
                        nrow=cfg["num_val_samples"],
                        normalize=True,
                        scale_each=True
                    )
                    tb_tracker = accelerator.get_tracker("tensorboard")
                    if tb_tracker is not None:
                        tb_tracker.writer.add_image("Val_Gen/MRI_Pred_GT", val_img_grid, epoch)
            
            model.train()
        
        # Save checkpoint
        if (epoch + 1) % cfg["save_every"] == 0 and accelerator.is_main_process:
            checkpoint_path = output_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1}.pt"
            
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': cfg,
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # End of training
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        # Save final model
        final_path = output_dir / "checkpoints" / "final_model.pt"
        torch.save({
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'config': cfg,
        }, final_path)
        print(f"Saved final model: {final_path}")
    
    accelerator.end_training()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    train(config)
