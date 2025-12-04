#!/usr/bin/env python
"""Quick test of DDP training (simpler than FSDP)."""
import sys
sys.path.insert(0, '.')

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
import tempfile

from train_lightning_fsdp import PFGMFIRLightningModule, PFGMDataModule

# Set precision for tensor cores
torch.set_float32_matmul_precision('medium')

def main():
    print('='*60)
    print('Testing DDP Training with 2 GPUs')
    print('='*60)

    # Create data module
    data_module = PFGMDataModule(
        data_dir='/home/lloyd/Documents/AR PFGM/train_pfgm',
        mr_pattern='mr.nii',
        ct_pattern='ct.nii',
        train_split=0.9,
        image_size=256,
        batch_size=2,
        num_workers=2,
        seed=42,
    )

    # Create small model for quick test
    model = PFGMFIRLightningModule(
        model_config='custom',
        base_channels=32,
        channel_mults=[1, 2, 4],
        num_res_blocks=1,
        attention_resolutions=[4],
        num_heads=2,
        mlp_ratio=2.0,
        radius_emb_dim=128,
        window_size=8,
        use_sfb=True,
        dropout=0.0,
        drop_path=0.1,
        sigma_min=0.002,
        sigma_max=5.0,
        sigma_data=0.5,
        P_mean=-1.2,
        P_std=1.2,
        rho=7.0,
        loss_type='l2',
        lr=1e-4,
        weight_decay=0.01,
        num_epochs=1,
        warmup_steps=10,
        num_sample_steps=10,
        num_val_samples=2,
    )

    print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')

    # DDP Strategy (simpler, no sharding)
    ddp_strategy = DDPStrategy(
        find_unused_parameters=False,
    )

    # Create temp dir for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            accelerator='gpu',
            devices=[0, 1],
            strategy=ddp_strategy,
            precision='16-mixed',  # FP16 instead of BF16
            max_epochs=1,
            max_steps=10,
            gradient_clip_val=1.0,
            log_every_n_steps=5,
            val_check_interval=10,
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            num_sanity_val_steps=0,  # Skip sanity check
        )
        
        print()
        print('Starting DDP training test...')
        trainer.fit(model, data_module)

    print()
    print('='*60)
    print('DDP Training test PASSED!')
    print('='*60)


if __name__ == '__main__':
    main()
