#!/usr/bin/env python
"""Quick test of FSDP training."""
import sys
sys.path.insert(0, '.')

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
import tempfile

from train_lightning_fsdp import PFGMFIRLightningModule, PFGMDataModule
from models.fir_blocks import FIRResBlock

# Set precision for tensor cores
torch.set_float32_matmul_precision('medium')

def main():
    print('='*60)
    print('Testing FSDP Training with 2 GPUs')
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

    # FSDP Strategy
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=100000,
    )

    fsdp_strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        state_dict_type='full',
        limit_all_gathers=True,
        cpu_offload=False,
    )

    # Create temp dir for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TensorBoardLogger(save_dir=tmpdir, name='test')
        
        trainer = Trainer(
            accelerator='gpu',
            devices=[0, 1],
            strategy=fsdp_strategy,
            precision='bf16-mixed',
            max_epochs=1,
            max_steps=20,
            gradient_clip_val=1.0,
            log_every_n_steps=5,
            val_check_interval=10,
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=False,
            num_sanity_val_steps=1,
        )
        
        print()
        print('Starting FSDP training test...')
        trainer.fit(model, data_module)

    print()
    print('='*60)
    print('FSDP Training test PASSED!')
    print('='*60)


if __name__ == '__main__':
    main()
