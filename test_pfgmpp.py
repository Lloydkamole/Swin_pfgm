#!/usr/bin/env python3
"""
Quick test script for PFGM++ implementation.
Tests perturbation, prior sampling, and basic forward pass.
"""

import sys
import torch
import numpy as np

# Add parent to path
sys.path.insert(0, '.')

from train_accelerate import (
    pfgmpp_perturbation,
    pfgmpp_sample_prior,
    pfgmpp_target,
    sample_sigma,
    edm_precond,
    edm_loss,
    edm_sample,
)


def test_pfgmpp_perturbation():
    """Test PFGM++ perturbation (inverse-beta)."""
    print("=" * 60)
    print("Testing PFGM++ Perturbation")
    print("=" * 60)
    
    B, C, H, W = 4, 1, 64, 64
    D = 128
    
    # Create dummy clean images
    images = torch.randn(B, C, H, W)
    sigma = sample_sigma(B, device=images.device)
    
    print(f"Input shape: {images.shape}")
    print(f"Sigma shape: {sigma.shape}, range: [{sigma.min():.4f}, {sigma.max():.4f}]")
    
    # Apply PFGM++ perturbation
    perturbed, used_sigma = pfgmpp_perturbation(images, sigma, D=D)
    
    print(f"Perturbed shape: {perturbed.shape}")
    print(f"Perturbation norm (mean): {(perturbed - images).view(B, -1).norm(dim=1).mean():.4f}")
    
    # Check that perturbation is non-trivial
    diff = (perturbed - images).abs().mean()
    assert diff > 0.01, f"Perturbation too small: {diff}"
    print(f"✓ Perturbation is non-trivial (mean abs diff: {diff:.4f})")
    
    print()


def test_pfgmpp_prior():
    """Test PFGM++ prior sampling."""
    print("=" * 60)
    print("Testing PFGM++ Prior Sampling")
    print("=" * 60)
    
    B, C, H, W = 4, 1, 64, 64
    D = 128
    sigma_max = 80.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sample from PFGM++ prior
    latents = pfgmpp_sample_prior(
        batch_size=B,
        channels=C,
        height=H,
        width=W,
        sigma_max=sigma_max,
        D=D,
        device=device,
    )
    
    print(f"Latent shape: {latents.shape}")
    print(f"Latent dtype: {latents.dtype}")
    print(f"Latent norm (per sample): {latents.view(B, -1).norm(dim=1)}")
    print(f"Latent mean: {latents.mean():.4f}, std: {latents.std():.4f}")
    
    # Expected norm should be around r_max * sqrt(inverse_beta)
    # For large N, this is approximately r_max = sigma_max * sqrt(D)
    r_max = sigma_max * np.sqrt(D)
    print(f"Expected r_max: {r_max:.2f}")
    
    print("✓ Prior sampling works")
    print()


def test_pfgmpp_loss():
    """Test PFGM++ loss computation."""
    print("=" * 60)
    print("Testing PFGM++ Loss")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, C, H, W = 2, 1, 64, 64
    D = 128
    
    # Import and create the real model
    from models import create_swin_fir_unet
    
    model = create_swin_fir_unet(
        config="custom",
        in_channels=2,
        out_channels=1,
        base_channels=32,  # Small for testing
        channel_mults=[1, 2],
        num_res_blocks=1,
        attention_resolutions=[2],
        num_heads=2,
        radius_emb_dim=64,
        window_size=8,
    ).to(device)
    
    # Create dummy data
    ct = torch.randn(B, C, H, W, device=device)
    mr = torch.randn(B, C, H, W, device=device)
    sigma = sample_sigma(B, device=device)
    
    print(f"CT shape: {ct.shape}")
    print(f"MR shape: {mr.shape}")
    print(f"Sigma: {sigma}")
    
    # Test with pfgmpp=False (EDM mode)
    loss_edm, metrics_edm = edm_loss(
        model, ct, mr, sigma,
        sigma_data=0.5,
        pfgmpp=False,
        D=D,
    )
    print(f"EDM loss: {loss_edm.item():.4f}")
    
    # Test with pfgmpp=True (PFGM++ mode)
    loss_pfgm, metrics_pfgm = edm_loss(
        model, ct, mr, sigma,
        sigma_data=0.5,
        pfgmpp=True,
        D=D,
    )
    print(f"PFGM++ loss: {loss_pfgm.item():.4f}")
    
    print("✓ Loss computation works for both modes")
    print()


def test_pfgmpp_sample():
    """Test PFGM++ sampling (short run)."""
    print("=" * 60)
    print("Testing PFGM++ Sampling")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, C, H, W = 1, 1, 64, 64
    D = 128
    
    # Import and create the real model
    from models import create_swin_fir_unet
    
    model = create_swin_fir_unet(
        config="custom",
        in_channels=2,
        out_channels=1,
        base_channels=32,  # Small for testing
        channel_mults=[1, 2],
        num_res_blocks=1,
        attention_resolutions=[2],
        num_heads=2,
        radius_emb_dim=64,
        window_size=8,
    ).to(device)
    
    # Create dummy MRI conditioning
    mri = torch.randn(B, C, H, W, device=device)
    
    print(f"MRI shape: {mri.shape}")
    
    # Test sampling with pfgmpp=False (EDM)
    print("Testing EDM sampling (5 steps)...")
    result_edm = edm_sample(
        model, mri,
        num_steps=5,
        sigma_min=0.002,
        sigma_max=80.0,
        pfgmpp=False,
        D=D,
    )
    print(f"EDM result shape: {result_edm.shape}")
    print(f"EDM result range: [{result_edm.min():.2f}, {result_edm.max():.2f}]")
    
    # Test sampling with pfgmpp=True (PFGM++)
    print("Testing PFGM++ sampling (5 steps)...")
    result_pfgm = edm_sample(
        model, mri,
        num_steps=5,
        sigma_min=0.002,
        sigma_max=80.0,
        pfgmpp=True,
        D=D,
    )
    print(f"PFGM++ result shape: {result_pfgm.shape}")
    print(f"PFGM++ result range: [{result_pfgm.min():.2f}, {result_pfgm.max():.2f}]")
    
    print("✓ Sampling works for both modes")
    print()


def main():
    print("\n" + "=" * 60)
    print("PFGM++ Implementation Tests")
    print("=" * 60 + "\n")
    
    try:
        test_pfgmpp_perturbation()
        test_pfgmpp_prior()
        test_pfgmpp_loss()
        test_pfgmpp_sample()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
