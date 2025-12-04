"""
Test script for PFGM-FIR models.

Run this to verify the models work correctly:
    python test_models.py
"""

import torch
import torch.nn as nn
import sys

def test_embeddings():
    """Test radius embedding module."""
    print("Testing RadiusEmbedding...")
    from models.embeddings import RadiusEmbedding
    
    embed = RadiusEmbedding(embed_dim=256)
    
    # Test with batch of radius values
    r = torch.rand(4)
    emb = embed(r)
    
    assert emb.shape == (4, 256), f"Expected (4, 256), got {emb.shape}"
    print(f"  RadiusEmbedding output shape: {emb.shape} ✓")
    
    return True


def test_swin_attention():
    """Test Swin Transformer block."""
    print("Testing SwinTransformerBlock...")
    from models.swin_attention import SwinTransformerBlock
    
    block = SwinTransformerBlock(
        dim=96,
        input_resolution=(64, 64),
        num_heads=4,
        window_size=8,
        shift_size=0,
    )
    
    # Input: [B, H*W, C]
    x = torch.randn(2, 64*64, 96)
    out = block(x, x_size=(64, 64))
    
    assert out.shape == (2, 64*64, 96), f"Expected (2, 4096, 96), got {out.shape}"
    print(f"  SwinTransformerBlock output shape: {out.shape} ✓")
    
    # Test shifted window
    block_shifted = SwinTransformerBlock(
        dim=96,
        input_resolution=(64, 64),
        num_heads=4,
        window_size=8,
        shift_size=4,
    )
    out_shifted = block_shifted(x, x_size=(64, 64))
    assert out_shifted.shape == (2, 64*64, 96)
    print(f"  Shifted window attention: {out_shifted.shape} ✓")
    
    return True


def test_fir_blocks():
    """Test FIR building blocks."""
    print("Testing FIR blocks...")
    from models.fir_blocks import FourierUnit, SFB, FIRResBlock
    
    # Test FourierUnit
    fu = FourierUnit(64)
    x = torch.randn(2, 64, 64, 64)
    out = fu(x)
    assert out.shape == x.shape, f"FourierUnit: Expected {x.shape}, got {out.shape}"
    print(f"  FourierUnit output shape: {out.shape} ✓")
    
    # Test SFB
    sfb = SFB(64)
    x = torch.randn(2, 64, 64, 64)
    out = sfb(x)
    assert out.shape == x.shape, f"SFB: Expected {x.shape}, got {out.shape}"
    print(f"  SFB output shape: {out.shape} ✓")
    
    # Test FIRResBlock
    block = FIRResBlock(
        in_channels=64,
        out_channels=64,
        radius_emb_dim=256,
        use_swin=False,
        use_sfb=True,
    )
    x = torch.randn(2, 64, 64, 64)
    r_emb = torch.randn(2, 256)
    out = block(x, r_emb)
    assert out.shape == (2, 64, 64, 64), f"FIRResBlock: Expected (2, 64, 64, 64), got {out.shape}"
    print(f"  FIRResBlock output shape: {out.shape} ✓")
    
    # Test FIRResBlock with Swin attention
    block_swin = FIRResBlock(
        in_channels=64,
        out_channels=64,
        radius_emb_dim=256,
        use_swin=True,
        use_sfb=True,
        input_resolution=(64, 64),
        num_heads=4,
        window_size=8,
    )
    out_swin = block_swin(x, r_emb, x_size=(64, 64))
    assert out_swin.shape == (2, 64, 64, 64)
    print(f"  FIRResBlock with Swin: {out_swin.shape} ✓")
    
    return True


def test_full_network():
    """Test complete SwinFIRUNet."""
    print("Testing SwinFIRUNet...")
    from models.network import SwinFIRUNet, create_swin_fir_unet
    
    # Create small model for testing
    model = create_swin_fir_unet(
        config='small',
        in_channels=2,
        out_channels=1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params / 1e6:.2f}M")
    
    # Test forward pass
    batch_size = 2
    img_size = 64
    
    noisy_ct = torch.randn(batch_size, 1, img_size, img_size)
    mri = torch.randn(batch_size, 1, img_size, img_size)
    r = torch.rand(batch_size)
    
    out = model(noisy_ct, mri, r)
    
    assert out.shape == (batch_size, 1, img_size, img_size), \
        f"Expected ({batch_size}, 1, {img_size}, {img_size}), got {out.shape}"
    print(f"  SwinFIRUNet output shape: {out.shape} ✓")
    
    # Test with different image sizes
    for size in [128, 256]:
        noisy_ct = torch.randn(1, 1, size, size)
        mri = torch.randn(1, 1, size, size)
        r = torch.rand(1)
        out = model(noisy_ct, mri, r)
        assert out.shape == (1, 1, size, size), f"Size {size}: Expected (1, 1, {size}, {size}), got {out.shape}"
        print(f"  Image size {size}×{size}: {out.shape} ✓")
    
    return True


def test_pfgm_functions():
    """Test PFGM++ perturbation and loss."""
    print("Testing PFGM++ functions...")
    
    # Import from training script
    sys.path.insert(0, '.')
    from train_pfgm_fir import pfgm_perturbation, sample_radius, pfgm_loss, pfgm_sample
    from models.network import create_swin_fir_unet
    
    # Test perturbation
    x = torch.randn(2, 1, 64, 64)
    r = torch.tensor([0.1, 0.5])
    x_noisy, u = pfgm_perturbation(x, r, D=128)
    
    assert x_noisy.shape == x.shape
    assert u.shape == x.shape
    print(f"  Perturbation output: {x_noisy.shape} ✓")
    
    # Check that u is normalized
    u_norm = u.flatten(1).norm(dim=1)
    assert torch.allclose(u_norm, torch.ones_like(u_norm), atol=1e-5), "u should be unit vectors"
    print("  Perturbation direction normalized ✓")
    
    # Test radius sampling
    r_samples = sample_radius(100, sigma_max=2.0, sigma_min=0.002, D=128)
    assert r_samples.shape == (100,)
    assert r_samples.min() >= 0
    print(f"  Sampled radius range: [{r_samples.min():.4f}, {r_samples.max():.4f}] ✓")
    
    # Test loss computation
    model = create_swin_fir_unet(config='small')
    x_clean = torch.randn(2, 1, 64, 64)
    mri = torch.randn(2, 1, 64, 64)
    r = sample_radius(2, device=x_clean.device)
    
    loss = pfgm_loss(model, x_clean, mri, r, D=128)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    print(f"  Loss value: {loss.item():.4f} ✓")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("PFGM-FIR Model Tests")
    print("=" * 60)
    
    tests = [
        ("Embeddings", test_embeddings),
        ("Swin Attention", test_swin_attention),
        ("FIR Blocks", test_fir_blocks),
        ("Full Network", test_full_network),
        ("PFGM++ Functions", test_pfgm_functions),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            print(f"\n{name}:")
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed! ✓")
    else:
        print("\nSome tests failed. ✗")
        sys.exit(1)


if __name__ == '__main__':
    main()
