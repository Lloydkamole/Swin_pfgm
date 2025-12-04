"""
PFGM-FIR Training Script with PyTorch Lightning + FSDP.

Training script for SwinFIR-based PFGM++ model for MRI-to-CT synthesis
using PyTorch Lightning with Fully Sharded Data Parallel (FSDP) for
training larger models across multiple GPUs.

Key features:
- PyTorch Lightning for clean training loop
- FSDP for memory-efficient distributed training
- Automatic mixed precision
- Checkpoint saving with FSDP state dict handling
- TensorBoard logging

Usage:
    python train_lightning_fsdp.py
"""

import os
import sys
import math
import random
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
import functools

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

# Import model components
from models import create_swin_fir_unet
from models.fir_blocks import FIRResBlock, SFB

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
    # -------------------------------------------------------------------------
    # Data Settings
    # -------------------------------------------------------------------------
    "data_dir": "/home/lloyd/Documents/AR PFGM/train_pfgm",
    "mr_pattern": "mr.nii",
    "ct_pattern": "ct.nii",
    "train_split": 0.9,
    "image_size": 256,
    
    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    "model_config": "custom",
    "base_channels": 64,
    "channel_mults": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attention_resolutions": [2, 3, 4],
    "num_heads": 4,
    "mlp_ratio": 2.0,
    "radius_emb_dim": 256,
    "window_size": 8,
    "use_sfb": True,
    "dropout": 0.0,
    "drop_path": 0.1,
    
    # -------------------------------------------------------------------------
    # EDM / Diffusion Parameters
    # -------------------------------------------------------------------------
    "sigma_min": 0.002,
    "sigma_max": 5.0,
    "sigma_data": 0.5,
    "P_mean": -1.2,
    "P_std": 1.2,
    "rho": 7.0,
    "loss_type": "l2",
    
    # -------------------------------------------------------------------------
    # Training Parameters
    # -------------------------------------------------------------------------
    "num_epochs": 20,
    "batch_size": 4,  # Per GPU batch size
    "lr": 1e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "accumulate_grad_batches": 1,
    
    # -------------------------------------------------------------------------
    # FSDP Settings
    # -------------------------------------------------------------------------
    "devices": [0, 1],  # GPU indices to use
    "precision": "bf16-mixed",  # "32", "16-mixed", "bf16-mixed"
    "fsdp_min_num_params": 1e6,  # Min params for FSDP wrapping
    
    # -------------------------------------------------------------------------
    # Sampling / Generation
    # -------------------------------------------------------------------------
    "num_sample_steps": 50,
    "num_val_samples": 4,
    
    # -------------------------------------------------------------------------
    # Logging & Checkpointing
    # -------------------------------------------------------------------------
    "output_dir": "./outputs_lightning",
    "log_every_n_steps": 10,
    "val_check_interval": 1.0,  # Validate every epoch
    "save_every_n_epochs": 5,
    
    # -------------------------------------------------------------------------
    # System
    # -------------------------------------------------------------------------
    "seed": 42,
    "num_workers": 4,
}


# =============================================================================
# Data Module
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


class PFGMDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for PFGM training."""
    
    def __init__(
        self,
        data_dir: str,
        mr_pattern: str = "mr.nii",
        ct_pattern: str = "ct.nii",
        train_split: float = 0.9,
        image_size: int = 256,
        batch_size: int = 4,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.mr_pattern = mr_pattern
        self.ct_pattern = ct_pattern
        self.train_split = train_split
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        all_files = collect_data_files(
            self.data_dir,
            mr_pattern=self.mr_pattern,
            ct_pattern=self.ct_pattern
        )
        
        if len(all_files) == 0:
            raise FileNotFoundError(f"No data found at {self.data_dir}")
        
        # Shuffle with seed for reproducibility
        random.seed(self.seed)
        random.shuffle(all_files)
        
        # Split
        split_idx = max(1, int(len(all_files) * self.train_split))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:] if len(all_files) > split_idx else []
        
        print(f"Data split: {len(train_files)} train, {len(val_files)} val patients")
        
        # Create datasets
        train_transforms = get_monai_nifti_transforms(mode="train", target_size=self.image_size)
        self.train_dataset = MonaiNiftiSliceDataset(
            data=train_files,
            transform=train_transforms,
            target_size=self.image_size
        )
        
        if val_files:
            val_transforms = get_monai_nifti_transforms(mode="val", target_size=self.image_size)
            self.val_dataset = MonaiNiftiSliceDataset(
                data=val_files,
                transform=val_transforms,
                target_size=self.image_size
            )
        
        print(f"Dataset sizes: {len(self.train_dataset)} train slices, "
              f"{len(self.val_dataset) if self.val_dataset else 0} val slices")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )


# =============================================================================
# Lightning Module
# =============================================================================

class PFGMFIRLightningModule(LightningModule):
    """PyTorch Lightning Module for PFGM-FIR training."""
    
    def __init__(
        self,
        # Model config
        model_config: str = "custom",
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [2, 3, 4],
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        radius_emb_dim: int = 256,
        window_size: int = 8,
        use_sfb: bool = True,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        # EDM config
        sigma_min: float = 0.002,
        sigma_max: float = 5.0,
        sigma_data: float = 0.5,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        rho: float = 7.0,
        loss_type: str = "l2",
        # Training config
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 20,
        warmup_steps: int = 1000,
        # Sampling config
        num_sample_steps: int = 50,
        num_val_samples: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Build model
        if model_config == "custom":
            self.model = create_swin_fir_unet(
                config="custom",
                in_channels=2,
                out_channels=1,
                base_channels=base_channels,
                channel_mults=channel_mults,
                num_res_blocks=num_res_blocks,
                attention_resolutions=attention_resolutions,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                radius_emb_dim=radius_emb_dim,
                window_size=window_size,
                use_sfb=use_sfb,
                dropout=dropout,
                drop_path_rate=drop_path,
            )
        else:
            self.model = create_swin_fir_unet(
                config=model_config,
                in_channels=2,
                out_channels=1,
                radius_emb_dim=radius_emb_dim,
                window_size=window_size,
                use_sfb=use_sfb,
                dropout=dropout,
                drop_path_rate=drop_path,
            )
        
        # Metrics
        if TORCHMETRICS_AVAILABLE:
            self.psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
            self.ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
            self.mae_metric = torchmetrics.MeanAbsoluteError()
        
        # For logging
        self.validation_step_outputs = []
    
    def forward(self, x: torch.Tensor, mri: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x, mri, c_noise)
    
    def edm_precond(
        self,
        x: torch.Tensor,
        mri: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Apply EDM preconditioning."""
        sigma_data = self.hparams.sigma_data
        
        x = x.float()
        mri = mri.float()
        sigma = sigma.float()
        
        sigma_4d = sigma.view(-1, 1, 1, 1)
        c_skip = sigma_data ** 2 / (sigma_4d ** 2 + sigma_data ** 2)
        c_out = sigma_4d * sigma_data / (sigma_4d ** 2 + sigma_data ** 2).sqrt()
        c_in = 1.0 / (sigma_4d ** 2 + sigma_data ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        x_scaled = c_in * x
        F_x = self.model(x_scaled, mri, c_noise)
        D_x = c_skip * x + c_out * F_x
        
        return D_x
    
    def sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample sigma from log-normal distribution."""
        rnd_normal = torch.randn(batch_size, device=device)
        sigma = (rnd_normal * self.hparams.P_std + self.hparams.P_mean).exp()
        sigma = sigma.clamp(min=self.hparams.sigma_min, max=self.hparams.sigma_max)
        return sigma
    
    def compute_loss(
        self,
        x_clean: torch.Tensor,
        mri: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute EDM loss."""
        sigma_data = self.hparams.sigma_data
        
        # Add noise
        n = torch.randn_like(x_clean) * sigma.view(-1, 1, 1, 1)
        x_noisy = x_clean + n
        
        # Get denoised prediction
        D_x = self.edm_precond(x_noisy, mri, sigma)
        
        # Compute weight
        sigma_v = sigma.view(-1, 1, 1, 1)
        weight = (sigma_v ** 2 + sigma_data ** 2) / ((sigma_v * sigma_data) ** 2)
        weight = weight.clamp(max=100.0)
        
        # Compute loss
        if self.hparams.loss_type == 'l1':
            loss = (weight * (D_x - x_clean).abs()).mean()
        elif self.hparams.loss_type == 'l2':
            loss = (weight * (D_x - x_clean) ** 2).mean()
        elif self.hparams.loss_type == 'huber':
            loss = (weight * F.smooth_l1_loss(D_x, x_clean, reduction='none')).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.hparams.loss_type}")
        
        # Metrics
        with torch.no_grad():
            mse = F.mse_loss(D_x, x_clean).item()
        
        metrics = {
            'mse': mse,
            'sigma_mean': sigma.mean().item(),
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def edm_sample(self, mri: torch.Tensor) -> torch.Tensor:
        """Generate CT from MRI using EDM sampling."""
        B, C, H, W = mri.shape
        device = mri.device
        
        num_steps = self.hparams.num_sample_steps
        sigma_min = self.hparams.sigma_min
        sigma_max = self.hparams.sigma_max
        rho = self.hparams.rho
        
        # Time step discretization
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * 
                   (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        
        # Initialize from noise
        x_next = torch.randn(B, 1, H, W, device=device, dtype=torch.float64) * t_steps[0]
        
        # Sampling loop
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            
            # Euler step
            sigma_batch = torch.full((B,), t_cur.item(), device=device, dtype=torch.float32)
            denoised = self.edm_precond(x_cur.float(), mri, sigma_batch).double()
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur
            
            # 2nd order correction (Heun)
            if i < num_steps - 1 and t_next > 0:
                sigma_batch = torch.full((B,), t_next.item(), device=device, dtype=torch.float32)
                denoised_next = self.edm_precond(x_next.float(), mri, sigma_batch).double()
                d_prime = (x_next - denoised_next) / t_next
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_next.float()
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        mr = batch["mr"]
        ct = batch["ct"]
        
        sigma = self.sample_sigma(ct.shape[0], ct.device)
        loss, metrics = self.compute_loss(ct, mr, sigma)
        
        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mse", metrics['mse'], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/sigma_mean", metrics['sigma_mean'], on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        mr = batch["mr"]
        ct = batch["ct"]
        
        sigma = self.sample_sigma(ct.shape[0], ct.device)
        loss, metrics = self.compute_loss(ct, mr, sigma)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Generate samples on first batch
        if batch_idx == 0 and self.global_rank == 0:
            num_samples = min(self.hparams.num_val_samples, mr.shape[0])
            mr_samples = mr[:num_samples]
            ct_samples = ct[:num_samples]
            
            pred_ct = self.edm_sample(mr_samples)
            pred_ct = torch.clamp(pred_ct, -1.0, 1.0)
            
            # Metrics
            if TORCHMETRICS_AVAILABLE:
                pred_01 = (pred_ct + 1.0) / 2.0
                ct_01 = (ct_samples + 1.0) / 2.0
                
                psnr = self.psnr_metric(pred_01, ct_01)
                ssim = self.ssim_metric(pred_01, ct_01)
                mae = self.mae_metric(pred_01, ct_01)
                
                self.log("val/psnr", psnr, sync_dist=True)
                self.log("val/ssim", ssim, sync_dist=True)
                self.log("val/mae", mae, sync_dist=True)
            
            # Log images
            if self.logger is not None:
                img_grid = vutils.make_grid(
                    torch.cat([mr_samples, pred_ct, ct_samples], dim=0),
                    nrow=num_samples,
                    normalize=True,
                    scale_each=True
                )
                self.logger.experiment.add_image(
                    "val/MRI_Pred_GT", img_grid, self.current_epoch
                )
        
        return {"val_loss": loss}
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Cosine annealing with warmup
        def lr_lambda(current_step):
            warmup_steps = self.hparams.warmup_steps
            total_steps = self.trainer.estimated_stepping_batches
            
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    print(f"\n{'='*60}")
    print("PFGM-FIR Training with PyTorch Lightning + FSDP")
    print(f"{'='*60}\n")
    
    # Environment info
    print("Environment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Lightning: {pl.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    print()
    
    # Seed
    pl.seed_everything(config["seed"], workers=True)
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["output_dir"]) / f"{timestamp}_pfgm_fir_fsdp"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}\n")
    
    # Data module
    data_module = PFGMDataModule(
        data_dir=config["data_dir"],
        mr_pattern=config["mr_pattern"],
        ct_pattern=config["ct_pattern"],
        train_split=config["train_split"],
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )
    
    # Model
    model = PFGMFIRLightningModule(
        model_config=config["model_config"],
        base_channels=config["base_channels"],
        channel_mults=config["channel_mults"],
        num_res_blocks=config["num_res_blocks"],
        attention_resolutions=config["attention_resolutions"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        radius_emb_dim=config["radius_emb_dim"],
        window_size=config["window_size"],
        use_sfb=config["use_sfb"],
        dropout=config["dropout"],
        drop_path=config["drop_path"],
        sigma_min=config["sigma_min"],
        sigma_max=config["sigma_max"],
        sigma_data=config["sigma_data"],
        P_mean=config["P_mean"],
        P_std=config["P_std"],
        rho=config["rho"],
        loss_type=config["loss_type"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        num_epochs=config["num_epochs"],
        num_sample_steps=config["num_sample_steps"],
        num_val_samples=config["num_val_samples"],
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M\n")
    
    # FSDP Strategy with auto-wrapping policy
    # Wrap FIRResBlock and SFB modules for optimal sharding
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=int(config["fsdp_min_num_params"]),
    )
    
    fsdp_strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing_policy={FIRResBlock},  # Save memory via activation checkpointing
        state_dict_type="full",  # For easier checkpoint loading
        limit_all_gathers=True,  # Reduce memory spikes
        cpu_offload=False,  # Keep on GPU for speed
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="epoch{epoch:02d}-val_loss{val/loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=str(run_dir),
        name="logs",
        version="",
    )
    
    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=config["devices"],
        strategy=fsdp_strategy,
        precision=config["precision"],
        max_epochs=config["num_epochs"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        gradient_clip_val=config["grad_clip"],
        log_every_n_steps=config["log_every_n_steps"],
        val_check_interval=config["val_check_interval"],
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=2,
    )
    
    # Train
    print(f"{'='*60}")
    print("Starting training...")
    print(f"  GPUs: {config['devices']}")
    print(f"  Precision: {config['precision']}")
    print(f"  Batch size per GPU: {config['batch_size']}")
    print(f"  Effective batch size: {config['batch_size'] * len(config['devices']) * config['accumulate_grad_batches']}")
    print(f"{'='*60}\n")
    
    trainer.fit(model, data_module)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
