"""
Comprehensive Training Diagnostic for 2.5D Model

This script checks:
1. Gradient flow through all layers
2. Weight initialization and distribution
3. Loss stability over iterations
4. Activation magnitudes
5. Model output statistics
"""

import torch
import torch.nn as nn
import numpy as np
from models.network_25d import create_swin_fir_unet_25d
from loader_25d import MonaiNifti25DDataset, get_monai_nifti_transforms_25d
from torch.utils.data import DataLoader
import os

# Import EDM functions from train script
import sys
sys.path.insert(0, os.path.dirname(__file__))

def sample_sigma(batch_size, P_mean=-1.2, P_std=1.2, sigma_min=0.002, sigma_max=80.0, device=None):
    rnd_normal = torch.randn(batch_size, device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    sigma = sigma.clamp(min=sigma_min, max=sigma_max)
    return sigma

def edm_precond(model, x, mri, sigma, sigma_data=0.5):
    x = x.float()
    mri = mri.float()
    sigma = sigma.float()
    
    sigma_5d = sigma.view(-1, 1, 1, 1, 1)
    c_skip = sigma_data ** 2 / (sigma_5d ** 2 + sigma_data ** 2)
    c_out = sigma_5d * sigma_data / (sigma_5d ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma_5d ** 2 + sigma_data ** 2).sqrt()
    c_noise = sigma.log() / 4
    
    x_scaled = c_in * x
    F_x = model(x_scaled, mri, c_noise)
    D_x = c_skip * x + c_out * F_x
    
    return D_x

def edm_loss(model, x_clean, mri, sigma, sigma_data=0.5):
    n = torch.randn_like(x_clean) * sigma.view(-1, 1, 1, 1, 1)
    x_noisy = x_clean + n
    
    D_x = edm_precond(model, x_noisy, mri, sigma, sigma_data)
    
    sigma_v = sigma.view(-1, 1, 1, 1, 1)
    weight = (sigma_v ** 2 + sigma_data ** 2) / ((sigma_v * sigma_data) ** 2)
    weight = weight.clamp(max=100.0)
    
    loss = (weight * (D_x - x_clean) ** 2).mean()
    
    return loss, D_x

def collect_data_files(data_path, mr_pattern="mr.nii", ct_pattern="ct.nii"):
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


def check_weight_initialization(model):
    """Check weight initialization statistics."""
    print("\n" + "="*80)
    print("WEIGHT INITIALIZATION CHECK")
    print("="*80)
    
    weight_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) >= 2:
            weight_stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item(),
                'shape': tuple(param.shape),
            }
    
    # Group by layer type
    conv_weights = {k: v for k, v in weight_stats.items() if 'conv' in k.lower() and 'weight' in k}
    norm_weights = {k: v for k, v in weight_stats.items() if ('norm' in k.lower() or 'bn' in k.lower()) and 'weight' in k}
    attn_weights = {k: v for k, v in weight_stats.items() if 'attn' in k.lower() and 'weight' in k}
    
    print(f"\nConvolution Layers: {len(conv_weights)}")
    if conv_weights:
        conv_stds = [v['std'] for v in conv_weights.values()]
        print(f"  Std range: [{min(conv_stds):.6f}, {max(conv_stds):.6f}]")
        print(f"  Avg std: {np.mean(conv_stds):.6f}")
        
        # Check for very small or very large initializations
        problematic_conv = [k for k, v in conv_weights.items() if v['std'] < 0.001 or v['std'] > 1.0]
        if problematic_conv:
            print(f"  ⚠️  WARNING: {len(problematic_conv)} layers with unusual std:")
            for k in problematic_conv[:5]:
                print(f"    {k}: std={conv_weights[k]['std']:.6f}")
    
    print(f"\nNormalization Layers: {len(norm_weights)}")
    if norm_weights:
        norm_means = [v['mean'] for v in norm_weights.values()]
        print(f"  Mean range: [{min(norm_means):.6f}, {max(norm_means):.6f}]")
        print(f"  Avg mean: {np.mean(norm_means):.6f}")
    
    print(f"\nAttention Layers: {len(attn_weights)}")
    if attn_weights:
        attn_stds = [v['std'] for v in attn_weights.values()]
        print(f"  Std range: [{min(attn_stds):.6f}, {max(attn_stds):.6f}]")
        print(f"  Avg std: {np.mean(attn_stds):.6f}")
    
    return weight_stats


def check_gradient_flow(model, loss):
    """Check gradient flow after backward pass."""
    print("\n" + "="*80)
    print("GRADIENT FLOW CHECK")
    print("="*80)
    
    loss.backward()
    
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_stats[name] = {
                'mean': param.grad.abs().mean().item(),
                'max': param.grad.abs().max().item(),
                'std': param.grad.std().item(),
                'has_nan': torch.isnan(param.grad).any().item(),
                'has_inf': torch.isinf(param.grad).any().item(),
            }
    
    # Check for gradient issues
    zero_grads = [k for k, v in grad_stats.items() if v['mean'] < 1e-10]
    large_grads = [k for k, v in grad_stats.items() if v['max'] > 100]
    nan_grads = [k for k, v in grad_stats.items() if v['has_nan']]
    inf_grads = [k for k, v in grad_stats.items() if v['has_inf']]
    
    print(f"\nTotal parameters with gradients: {len(grad_stats)}")
    print(f"Parameters with near-zero gradients (<1e-10): {len(zero_grads)}")
    print(f"Parameters with large gradients (>100): {len(large_grads)}")
    print(f"Parameters with NaN gradients: {len(nan_grads)}")
    print(f"Parameters with Inf gradients: {len(inf_grads)}")
    
    if zero_grads:
        print(f"\n⚠️  Potential vanishing gradients in:")
        for k in zero_grads[:5]:
            print(f"  {k}: mean={grad_stats[k]['mean']:.2e}, max={grad_stats[k]['max']:.2e}")
    
    if large_grads:
        print(f"\n⚠️  Potential exploding gradients in:")
        for k in large_grads[:5]:
            print(f"  {k}: mean={grad_stats[k]['mean']:.2e}, max={grad_stats[k]['max']:.2e}")
    
    if nan_grads or inf_grads:
        print(f"\n❌ CRITICAL: NaN/Inf gradients detected!")
        for k in (nan_grads + inf_grads)[:5]:
            print(f"  {k}")
    
    # Group gradients by layer
    grad_means = [v['mean'] for v in grad_stats.values()]
    grad_maxs = [v['max'] for v in grad_stats.values()]
    
    print(f"\nOverall gradient statistics:")
    print(f"  Mean gradient magnitude: {np.mean(grad_means):.2e}")
    print(f"  Max gradient magnitude: {np.max(grad_maxs):.2e}")
    print(f"  Gradient mean range: [{min(grad_means):.2e}, {max(grad_means):.2e}]")
    
    return grad_stats


def check_activation_magnitudes(model, x, mri, c_noise):
    """Check activation magnitudes through forward pass."""
    print("\n" + "="*80)
    print("ACTIVATION MAGNITUDE CHECK")
    print("="*80)
    
    activations = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = {
                    'mean': output.abs().mean().item(),
                    'std': output.std().item(),
                    'max': output.abs().max().item(),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item(),
                }
        return hook
    
    # Register hooks on key layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.GroupNorm, nn.LayerNorm)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        output = model(x, mri, c_noise)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print(f"\nCaptured activations from {len(activations)} layers")
    
    # Check for issues
    nan_acts = [k for k, v in activations.items() if v['has_nan']]
    inf_acts = [k for k, v in activations.items() if v['has_inf']]
    zero_acts = [k for k, v in activations.items() if v['mean'] < 1e-10]
    large_acts = [k for k, v in activations.items() if v['max'] > 1000]
    
    if nan_acts or inf_acts:
        print(f"\n❌ CRITICAL: NaN/Inf activations detected in {len(nan_acts + inf_acts)} layers!")
        for k in (nan_acts + inf_acts)[:5]:
            print(f"  {k}")
    
    if zero_acts:
        print(f"\n⚠️  {len(zero_acts)} layers with near-zero activations:")
        for k in zero_acts[:5]:
            print(f"  {k}: mean={activations[k]['mean']:.2e}")
    
    if large_acts:
        print(f"\n⚠️  {len(large_acts)} layers with very large activations:")
        for k in large_acts[:5]:
            print(f"  {k}: max={activations[k]['max']:.2e}")
    
    act_means = [v['mean'] for v in activations.values()]
    print(f"\nOverall activation statistics:")
    print(f"  Mean activation: {np.mean(act_means):.2e}")
    print(f"  Activation range: [{min(act_means):.2e}, {max(act_means):.2e}]")
    
    return activations


def check_loss_stability(model, dataloader, device, num_steps=20):
    """Check loss stability over multiple training steps."""
    print("\n" + "="*80)
    print("LOSS STABILITY CHECK")
    print("="*80)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    losses = []
    mses = []
    pred_stds = []
    target_stds = []
    
    model.train()
    
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        mr = batch["mr"].to(device)
        ct = batch["ct"].to(device)
        
        sigma = sample_sigma(ct.shape[0], device=device)
        
        optimizer.zero_grad()
        loss, D_x = edm_loss(model, ct, mr, sigma)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ CRITICAL: Loss is NaN/Inf at step {step}!")
            break
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            mse = torch.nn.functional.mse_loss(D_x, ct).item()
            pred_std = D_x.std().item()
            target_std = ct.std().item()
        
        losses.append(loss.item())
        mses.append(mse)
        pred_stds.append(pred_std)
        target_stds.append(target_std)
        
        if step % 5 == 0:
            print(f"Step {step:3d}: loss={loss.item():.4f}, mse={mse:.4f}, "
                  f"pred_std={pred_std:.3f}, target_std={target_std:.3f}")
    
    # Analyze stability
    print(f"\nLoss statistics over {len(losses)} steps:")
    print(f"  Mean: {np.mean(losses):.4f}")
    print(f"  Std: {np.std(losses):.4f}")
    print(f"  Min: {np.min(losses):.4f}")
    print(f"  Max: {np.max(losses):.4f}")
    print(f"  Range: {np.max(losses) - np.min(losses):.4f}")
    
    # Check if loss is decreasing
    if len(losses) >= 10:
        early_mean = np.mean(losses[:5])
        late_mean = np.mean(losses[-5:])
        print(f"\n  First 5 steps avg: {early_mean:.4f}")
        print(f"  Last 5 steps avg: {late_mean:.4f}")
        print(f"  Change: {late_mean - early_mean:.4f} ({(late_mean/early_mean - 1)*100:.1f}%)")
        
        if late_mean >= early_mean:
            print(f"  ⚠️  WARNING: Loss not decreasing!")
    
    # Check prediction vs target std
    print(f"\nPrediction std: {np.mean(pred_stds):.3f} ± {np.std(pred_stds):.3f}")
    print(f"Target std: {np.mean(target_stds):.3f} ± {np.std(target_stds):.3f}")
    
    ratio = np.mean(pred_stds) / np.mean(target_stds)
    print(f"Pred/Target std ratio: {ratio:.3f}")
    
    if ratio < 0.5 or ratio > 2.0:
        print(f"  ⚠️  WARNING: Prediction std significantly different from target!")
    
    return losses, mses, pred_stds, target_stds


def main():
    print("\n" + "="*80)
    print("2.5D MODEL TRAINING DIAGNOSTIC")
    print("="*80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nCreating model...")
    model = create_swin_fir_unet_25d(
        in_channels=2,
        out_channels=1,
        num_slices=5,
        base_channels=64,
        channel_mults=(1, 2, 4),
        attention_resolutions=(2, 4),
        num_res_blocks=2,
        radius_emb_dim=256,
        dropout=0.0,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")
    
    # Check weight initialization
    weight_stats = check_weight_initialization(model)
    
    # Load data
    print("\nLoading data...")
    data_path = "/home/lloyd/Documents/AR PFGM/pfgm_train"
    all_files = collect_data_files(data_path)
    print(f"Found {len(all_files)} patients")
    
    train_transforms = get_monai_nifti_transforms_25d(mode="train")
    dataset = MonaiNifti25DDataset(
        data=all_files[:10],  # Use first 10 for quick test
        transform=train_transforms,
        target_size=256,
        num_slices=5,
        stride=3,
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} volumes")
    
    # Get a batch
    batch = next(iter(dataloader))
    mr = batch["mr"].to(device)
    ct = batch["ct"].to(device)
    
    print(f"\nBatch shapes:")
    print(f"  MR: {mr.shape}")
    print(f"  CT: {ct.shape}")
    print(f"  MR range: [{mr.min().item():.3f}, {mr.max().item():.3f}]")
    print(f"  CT range: [{ct.min().item():.3f}, {ct.max().item():.3f}]")
    
    # Test forward pass
    print("\nTesting forward pass...")
    sigma = sample_sigma(ct.shape[0], device=device)
    c_noise = sigma.log() / 4
    
    with torch.no_grad():
        x_scaled = ct * 0.5  # Simple scaling
        output = model(x_scaled, mr, c_noise)
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print(f"  Output mean: {output.mean().item():.3f}")
        print(f"  Output std: {output.std().item():.3f}")
    
    # Check activation magnitudes
    activations = check_activation_magnitudes(model, x_scaled, mr, c_noise)
    
    # Test backward pass
    print("\nTesting backward pass...")
    loss, D_x = edm_loss(model, ct, mr, sigma)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  D_x range: [{D_x.min().item():.3f}, {D_x.max().item():.3f}]")
    
    # Check gradient flow
    grad_stats = check_gradient_flow(model, loss)
    
    # Check loss stability
    losses, mses, pred_stds, target_stds = check_loss_stability(model, dataloader, device)
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    issues = []
    
    # Check for critical issues
    if any(torch.isnan(p.data).any() for p in model.parameters()):
        issues.append("❌ CRITICAL: NaN in model weights")
    
    grad_means = [v['mean'] for v in grad_stats.values() if not v['has_nan']]
    if grad_means and np.mean(grad_means) < 1e-8:
        issues.append("⚠️  WARNING: Very small gradients (potential vanishing)")
    
    if grad_means and np.max([v['max'] for v in grad_stats.values()]) > 100:
        issues.append("⚠️  WARNING: Very large gradients (potential explosion)")
    
    if len(losses) >= 10:
        if np.std(losses) / np.mean(losses) > 0.5:
            issues.append("⚠️  WARNING: High loss variance (unstable training)")
        
        if np.mean(losses[-5:]) >= np.mean(losses[:5]):
            issues.append("⚠️  WARNING: Loss not decreasing")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No critical issues detected!")
        print("   Model appears ready for training.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
