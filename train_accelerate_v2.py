"""
PFGM-FIR v2 Training Script with Cross-Attention and Flash Attention.

Multi-GPU training with Hugging Face Accelerate for the new architecture that uses:
- Separate MRI encoder (not concatenated with CT)
- Multi-scale cross-attention in decoder
- Flash Attention for efficient computation

Usage:
    # Single GPU:
    python train_accelerate_v2.py
    
    # Multi-GPU with accelerate:
    accelerate launch --num_processes 2 --gpu_ids 0,1 train_accelerate_v2.py
    
    # Or with config file:
    accelerate launch --config_file accelerate_config.yaml train_accelerate_v2.py
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

# Import v2 model with cross-attention
from models.network_v2 import create_swin_fir_unet_cross_attn

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
    "data_dir": "/home/lloyd/Documents/AR PFGM/train_pfgm",
    "mr_pattern": "mr.nii",
    "ct_pattern": "ct.nii",
    "train_split": 0.9,
    "image_size": 256,
    
    # Model Architecture (v2 with cross-attention)
    "model_config": "custom",
    "base_channels": 64,
    "channel_mults": [1, 2, 4, 8],
    "num_res_blocks": 2,
    "attention_resolutions": [2, 4, 8],      # Swin self-attention at these levels
    "cross_attn_resolutions": [2, 4, 8],     # Cross-attention at these decoder levels
    "num_heads": 4,
    "cross_attn_heads": 8,                   # Cross-attention heads
    "mlp_ratio": 4.0,
    "radius_emb_dim": 256,
    "window_size": 4,
    "use_sfb": True,
    "dropout": 0.0,
    "drop_path": 0.1,
    "mri_encoder_blocks": 2,                 # Blocks per stage in MRI encoder
    
    # EDM Parameters
    "sigma_min": 0.002,
    "sigma_max": 10.0,
    "sigma_data": 0.5,
    "P_mean": -1.2,
    "P_std": 1.2,
    "rho": 7.0,
    "loss_type": "l2",
    
    # Training Parameters
    "num_epochs": 100,
    "batch_size": 5,  # Per GPU batch size
    "lr": 1e-4,
    "weight_decay": 0.01,
    "grad_clip": None,  # Enable gradient clipping for stability
    "warmup_steps": 500,
    
    # Sampling / Generation
    "num_sample_steps": 50,
    "num_val_samples": 4,
    
    # Logging & Checkpointing
    "output_dir": "./outputs_accelerate_v2",
    "log_every": 5,
    "val_every": 1,
    "save_every": 5,
    "sample_every": 1,
    "resume": None,
    
    # Gradient & ODE Monitoring
    "log_gradients": True,
    "log_ode_trajectory": True,
    "grad_log_interval": 100,
    
    # System
    "seed": 42,
    "num_workers": 4,
    "mixed_precision": "bf16",  # Use bf16 for Flash Attention efficiency
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
                'has_inf': torch.isinf(param.grad).any().item(),
            }
    return grad_info


def check_numerical_stability(tensor_dict: dict, prefix: str = "") -> dict:
    """Check tensors for NaN/Inf values and compute statistics."""
    stats = {}
    for name, t in tensor_dict.items():
        if t is None:
            continue
        key = f"{prefix}/{name}" if prefix else name
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        stats[f"{key}/has_nan"] = float(has_nan)
        stats[f"{key}/has_inf"] = float(has_inf)
        stats[f"{key}/mean"] = t.float().mean().item() if not (has_nan or has_inf) else float('nan')
        stats[f"{key}/std"] = t.float().std().item() if not (has_nan or has_inf) else float('nan')
        stats[f"{key}/min"] = t.float().min().item() if not (has_nan or has_inf) else float('nan')
        stats[f"{key}/max"] = t.float().max().item() if not (has_nan or has_inf) else float('nan')
    return stats


def check_model_weights(model, prefix: str = "") -> dict:
    """Check model weights for NaN/Inf and extreme values."""
    stats = {}
    for name, param in model.named_parameters():
        has_nan = torch.isnan(param.data).any().item()
        has_inf = torch.isinf(param.data).any().item()
        if has_nan or has_inf:
            key = f"{prefix}/{name}" if prefix else name
            stats[f"{key}/has_nan"] = float(has_nan)
            stats[f"{key}/has_inf"] = float(has_inf)
    return stats


# =============================================================================
# EDM Functions (adapted for v2 model signature)
# =============================================================================

def edm_precond(
    model: nn.Module,
    x: torch.Tensor,
    mri: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """
    Apply EDM preconditioning to model output.
    
    Note: v2 model takes (x, mri, r) separately - no concatenation needed.
    """
    x = x.float()
    mri = mri.float()
    sigma = sigma.float()
    
    sigma_4d = sigma.view(-1, 1, 1, 1)
    c_skip = sigma_data ** 2 / (sigma_4d ** 2 + sigma_data ** 2)
    c_out = sigma_4d * sigma_data / (sigma_4d ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma_4d ** 2 + sigma_data ** 2).sqrt()
    c_noise = sigma.log() / 4
    
    x_scaled = c_in * x
    
    # v2 model signature: model(noisy_ct, mri, radius)
    # MRI is processed separately by the MRI encoder inside the model
    F_x = model(x_scaled, mri, c_noise)
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


def edm_loss(
    model: nn.Module,
    x_clean: torch.Tensor,
    mri: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
    loss_type: str = 'l2',
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute EDM-style loss."""
    n = torch.randn_like(x_clean) * sigma.view(-1, 1, 1, 1)
    x_noisy = x_clean + n
    
    D_x = edm_precond(model, x_noisy, mri, sigma, sigma_data)
    
    sigma_v = sigma.view(-1, 1, 1, 1)
    weight = (sigma_v ** 2 + sigma_data ** 2) / ((sigma_v * sigma_data) ** 2)
    weight = weight.clamp(max=100.0)
    
    if loss_type == 'l1':
        loss = (weight * (D_x - x_clean).abs()).mean()
    elif loss_type == 'l2':
        loss = (weight * (D_x - x_clean) ** 2).mean()
    elif loss_type == 'huber':
        loss = (weight * F.smooth_l1_loss(D_x, x_clean, reduction='none')).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
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
) -> torch.Tensor:
    """
    Generate CT images from MRI using EDM sampling (Algorithm 2 from EDM paper).
    """
    B, C, H, W = mri.shape
    device = mri.device
    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    
    x_next = torch.randn(B, 1, H, W, device=device, dtype=torch.float64) * t_steps[0]
    
    traj_norms = [x_next.norm().item()]
    denoised_norms = []
    score_norms = []
    update_norms = []
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        
        sigma_batch = torch.full((B,), t_hat.item(), device=device, dtype=torch.float32)
        denoised = edm_precond(model, x_hat.float(), mri, sigma_batch, sigma_data).double()
        d_cur = (x_hat - denoised) / t_hat
        update = (t_next - t_hat) * d_cur
        x_next = x_hat + update
        
        if i < num_steps - 1 and t_next > 0:
            sigma_batch = torch.full((B,), t_next.item(), device=device, dtype=torch.float32)
            denoised_next = edm_precond(model, x_next.float(), mri, sigma_batch, sigma_data).double()
            d_prime = (x_next - denoised_next) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
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
    """Main training function with Accelerate for v2 model."""
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg["output_dir"]) / f"{timestamp}_pfgm_fir_v2"
    
    # Initialize Accelerator
    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
    )
    
    # DDP kwargs - find_unused_parameters needed because some params may not
    # participate in loss on every iteration (e.g., zero-initialized residual convs)
    ddp_kwargs = {
        "find_unused_parameters": True,
        "static_graph": False,  # Can't use static_graph with find_unused_parameters
    }
    
    accelerator = Accelerator(
        mixed_precision=cfg["mixed_precision"],
        gradient_accumulation_steps=1,
        project_config=project_config,
        log_with="tensorboard",
        kwargs_handlers=[
            __import__('accelerate', fromlist=['DistributedDataParallelKwargs']).DistributedDataParallelKwargs(**ddp_kwargs)
        ],
    )
    
    # Set seed
    set_seed(cfg["seed"])
    
    # Only main process prints
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"PFGM-FIR v2 Training (Cross-Attention + Flash Attention)")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Mixed precision: {cfg['mixed_precision']}")
        print(f"{'='*60}\n")
    
    # Build v2 model with cross-attention
    if cfg["model_config"] == "custom":
        model = create_swin_fir_unet_cross_attn(
            config="custom",
            in_channels=1,  # Just noisy CT (MRI goes through separate encoder)
            out_channels=1,
            base_channels=cfg["base_channels"],
            channel_mults=cfg["channel_mults"],
            num_res_blocks=cfg["num_res_blocks"],
            attention_resolutions=cfg["attention_resolutions"],
            cross_attn_resolutions=cfg["cross_attn_resolutions"],
            num_heads=cfg["num_heads"],
            cross_attn_heads=cfg["cross_attn_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            radius_emb_dim=cfg["radius_emb_dim"],
            window_size=cfg["window_size"],
            use_sfb=cfg["use_sfb"],
            dropout=cfg["dropout"],
            drop_path_rate=cfg["drop_path"],
            mri_encoder_blocks=cfg["mri_encoder_blocks"],
            max_resolution=cfg["image_size"],
        )
    else:
        model = create_swin_fir_unet_cross_attn(
            config=cfg["model_config"],
            in_channels=1,
            out_channels=1,
        )
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        mri_encoder_params = sum(p.numel() for p in model.mri_encoder.parameters())
        print(f"Total model parameters: {total_params/1e6:.2f}M")
        print(f"  MRI encoder parameters: {mri_encoder_params/1e6:.2f}M")
        print(f"  CT UNet parameters: {(total_params - mri_encoder_params)/1e6:.2f}M")
        print(f"\n  Architecture config:")
        print(f"    base_channels: {cfg['base_channels']}")
        print(f"    channel_mults: {cfg['channel_mults']}")
        print(f"    num_res_blocks: {cfg['num_res_blocks']}")
        print(f"    attention_resolutions: {cfg['attention_resolutions']}")
        print(f"    cross_attn_resolutions: {cfg['cross_attn_resolutions']}")
        print(f"    num_heads: {cfg['num_heads']}")
        print(f"    cross_attn_heads: {cfg['cross_attn_heads']}")
        print(f"    window_size: {cfg['window_size']}")
        print(f"    mri_encoder_blocks: {cfg['mri_encoder_blocks']}")
    
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
        print("\nLoading data files...")
    
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
    
    # Learning rate scheduler
    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = cfg.get("warmup_steps", 500)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = None
    if not is_fsdp:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Prepare with Accelerate
    if accelerator.is_main_process:
        print("Preparing model and data with Accelerator...")
    
    if is_fsdp:
        model = accelerator.prepare(model)
        optimizer = AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.95),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        optimizer, train_loader, scheduler = accelerator.prepare(
            optimizer, train_loader, scheduler
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    
    if accelerator.is_main_process:
        print("Model and data prepared!")
    
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    
    # Initialize tracking
    trackable_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)):
            trackable_cfg[k] = str(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            trackable_cfg[k] = v if v is not None else "None"
        else:
            trackable_cfg[k] = str(v)
    accelerator.init_trackers("pfgm_fir_v2", config=trackable_cfg)
    
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
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg['num_epochs']}",
            disable=not accelerator.is_main_process
        )
        
        for batch_idx, batch in enumerate(pbar):
            mr = batch["mr"]
            ct = batch["ct"]
            
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
                    loss_type=cfg["loss_type"]
                )
                
                accelerator.backward(loss)
                
                # Check gradient flow BEFORE optimizer step (while gradients are available)
                grad_log_dict = None
                if cfg.get("log_gradients", False) and batch_idx % cfg.get("grad_log_interval", 100) == 0:
                    unwrapped = accelerator.unwrap_model(model)
                    grad_info = check_gradient_flow(unwrapped.named_parameters())
                    if grad_info:
                        grad_means = [v['mean'] for v in grad_info.values()]
                        grad_maxs = [v['max'] for v in grad_info.values()]
                        has_nan = any(v['has_nan'] for v in grad_info.values())
                        has_inf = any(v.get('has_inf', False) for v in grad_info.values())
                        
                        # Per-module gradient analysis
                        module_grads = {}
                        for name, info in grad_info.items():
                            parts = name.split('.')
                            module = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
                            if module not in module_grads:
                                module_grads[module] = {'means': [], 'maxs': []}
                            module_grads[module]['means'].append(info['mean'])
                            module_grads[module]['maxs'].append(info['max'])
                        
                        vanishing_threshold = 1e-7
                        exploding_threshold = 1e3
                        vanishing_modules = []
                        exploding_modules = []
                        
                        for module, grads in module_grads.items():
                            avg_grad = np.mean(grads['means'])
                            max_grad = np.max(grads['maxs'])
                            if avg_grad < vanishing_threshold:
                                vanishing_modules.append((module, avg_grad))
                            if max_grad > exploding_threshold:
                                exploding_modules.append((module, max_grad))
                        
                        grad_log_dict = {
                            "grad/mean_avg": float(np.mean(grad_means)),
                            "grad/mean_max": float(np.max(grad_means)),
                            "grad/mean_min": float(np.min(grad_means)),
                            "grad/max_avg": float(np.mean(grad_maxs)),
                            "grad/max_max": float(np.max(grad_maxs)),
                            "grad/has_nan": float(has_nan),
                            "grad/has_inf": float(has_inf),
                            "grad/num_vanishing_modules": len(vanishing_modules),
                            "grad/num_exploding_modules": len(exploding_modules),
                        }
                        
                        for module in ['mri_encoder', 'encoder_stages', 'decoder_stages', 'bottleneck', 'cross_attn']:
                            matching = {k: v for k, v in module_grads.items() if module in k}
                            if matching:
                                all_means = [m for g in matching.values() for m in g['means']]
                                all_maxs = [m for g in matching.values() for m in g['maxs']]
                                grad_log_dict[f"grad/{module}_mean"] = float(np.mean(all_means))
                                grad_log_dict[f"grad/{module}_max"] = float(np.max(all_maxs))
                        
                        if vanishing_modules and accelerator.is_main_process:
                            print(f"\n⚠️ Vanishing gradients in: {[m[0] for m in vanishing_modules[:3]]}")
                        if exploding_modules and accelerator.is_main_process:
                            print(f"\n⚠️ Exploding gradients in: {[m[0] for m in exploding_modules[:3]]}")
                        if has_nan and accelerator.is_main_process:
                            nan_params = [k for k, v in grad_info.items() if v['has_nan']]
                            print(f"\n⚠️ NaN gradients in: {nan_params[:5]}")
                        if has_inf and accelerator.is_main_process:
                            inf_params = [k for k, v in grad_info.items() if v.get('has_inf', False)]
                            print(f"\n⚠️ Inf gradients in: {inf_params[:5]}")
                
                if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Log gradients after zero_grad is okay - we captured them before
                if grad_log_dict is not None:
                    accelerator.log(grad_log_dict, step=global_step)
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.3e}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            if batch_idx % cfg["log_every"] == 0:
                accelerator.log({
                    "train/loss": loss.item(),
                    "train/mse": metrics['mse'],
                    "train/pred_std": metrics['pred_std'],
                    "train/target_std": metrics['target_std'],
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/sigma_mean": metrics['sigma_mean'],
                    "train/sigma_max": metrics['sigma_max'],
                    "train/sigma_min": metrics['sigma_min'],
                }, step=global_step)
            
            # Check for NaN/Inf in loss (numerical explosion)
            if torch.isnan(loss) or torch.isinf(loss):
                if accelerator.is_main_process:
                    print(f"\n⚠️ WARNING: NaN/Inf detected in loss at step {global_step}!")
                    print(f"  Loss: {loss.item()}")
                    print(f"  Sigma range: [{sigma.min().item():.4f}, {sigma.max().item():.4f}]")
                accelerator.log({"train/nan_inf_detected": 1.0}, step=global_step)
            
            global_step += 1
        
        # Epoch stats
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start
        
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
        
        # Sample generation
        if (epoch + 1) % cfg["sample_every"] == 0:
            model.eval()
            sample_batch = next(iter(train_loader))
            
            if accelerator.is_main_process:
                with torch.no_grad():
                    mr_samples = sample_batch["mr"][:cfg["num_val_samples"]]
                    ct_samples = sample_batch["ct"][:cfg["num_val_samples"]]
                
                unwrapped_model = accelerator.unwrap_model(model)
                
                if cfg.get("log_ode_trajectory", False):
                    pred_ct, ode_stats = edm_sample(
                        unwrapped_model, mr_samples,
                        num_steps=cfg["num_sample_steps"],
                        sigma_min=cfg["sigma_min"],
                        sigma_max=cfg["sigma_max"],
                        sigma_data=cfg["sigma_data"],
                        rho=cfg["rho"],
                        return_trajectory_stats=True,
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
                        num_steps=cfg["num_sample_steps"],
                        sigma_min=cfg["sigma_min"],
                        sigma_max=cfg["sigma_max"],
                        sigma_data=cfg["sigma_data"],
                        rho=cfg["rho"],
                    )
                
                pred_ct = torch.clamp(pred_ct, -1.0, 1.0)
                pred_ct_01 = (pred_ct + 1.0) / 2.0
                ct_samples_01 = (ct_samples + 1.0) / 2.0
                
                if psnr_metric is not None:
                    psnr_val = psnr_metric(pred_ct_01, ct_samples_01)
                    ssim_val = ssim_metric(pred_ct_01, ct_samples_01)
                    mae_val = mae_metric(pred_ct_01, ct_samples_01) if mae_metric else None
                    
                    log_dict = {
                        "gen/psnr": psnr_val.item(),
                        "gen/ssim": ssim_val.item(),
                    }
                    if mae_val is not None:
                        log_dict["gen/mae"] = mae_val.item()
                    
                    accelerator.log(log_dict, step=epoch)
                    mae_str = f", MAE={mae_val:.5f}" if mae_val else ""
                    print(f"  Generation: PSNR={psnr_val:.3f}, SSIM={ssim_val:.5f}{mae_str}")
                
                # Log sample images to TensorBoard
                img_grid = vutils.make_grid(
                    torch.cat([mr_samples.cpu(), pred_ct.cpu(), ct_samples.cpu()], dim=0),
                    nrow=cfg["num_val_samples"],
                    normalize=True,
                    scale_each=True
                )
                tb_tracker = accelerator.get_tracker("tensorboard")
                if tb_tracker is not None:
                    tb_tracker.writer.add_image("Gen/MRI_Pred_GT", img_grid, epoch)
            
            model.train()
            accelerator.wait_for_everyone()
        
        # Validation
        if val_loader is not None and (epoch + 1) % cfg["val_every"] == 0:
            model.eval()
            val_loss_accum = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    mr = batch["mr"]
                    ct = batch["ct"]
                    
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
                        loss_type=cfg["loss_type"]
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
            
            # Validation sample generation
            val_sample_batch = next(iter(val_loader))
            
            if accelerator.is_main_process and psnr_metric is not None:
                mr_val_samples = val_sample_batch["mr"][:cfg["num_val_samples"]]
                ct_val_samples = val_sample_batch["ct"][:cfg["num_val_samples"]]
                
                unwrapped_model = accelerator.unwrap_model(model)
                
                pred_val_ct = edm_sample(
                    unwrapped_model, mr_val_samples,
                    num_steps=cfg["num_sample_steps"],
                    sigma_min=cfg["sigma_min"],
                    sigma_max=cfg["sigma_max"],
                    sigma_data=cfg["sigma_data"],
                    rho=cfg["rho"],
                )
                
                pred_val_ct = torch.clamp(pred_val_ct, -1.0, 1.0)
                pred_val_ct_01 = (pred_val_ct + 1.0) / 2.0
                ct_val_samples_01 = (ct_val_samples + 1.0) / 2.0
                
                val_psnr = psnr_metric(pred_val_ct_01, ct_val_samples_01)
                val_ssim = ssim_metric(pred_val_ct_01, ct_val_samples_01)
                val_mae = mae_metric(pred_val_ct_01, ct_val_samples_01) if mae_metric else None
                
                val_gen_log = {
                    "val_gen/psnr": val_psnr.item(),
                    "val_gen/ssim": val_ssim.item(),
                }
                if val_mae is not None:
                    val_gen_log["val_gen/mae"] = val_mae.item()
                
                accelerator.log(val_gen_log, step=epoch)
                
                val_mae_str = f", MAE={val_mae:.5f}" if val_mae else ""
                print(f"  Val Generation: PSNR={val_psnr:.3f}, SSIM={val_ssim:.5f}{val_mae_str}")
                
                val_img_grid = vutils.make_grid(
                    torch.cat([mr_val_samples.cpu(), pred_val_ct.cpu(), ct_val_samples.cpu()], dim=0),
                    nrow=cfg["num_val_samples"],
                    normalize=True,
                    scale_each=True
                )
                tb_tracker = accelerator.get_tracker("tensorboard")
                if tb_tracker is not None:
                    tb_tracker.writer.add_image("Val_Gen/MRI_Pred_GT", val_img_grid, epoch)
            
            model.train()
            accelerator.wait_for_everyone()
        
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
