"""
Volume Inference Script for PFGM-FIR Model.

Processes entire MRI volumes slice-by-slice to generate corresponding CT volumes.

IMPORTANT: Enhanced inference features:
- Model is unwrapped after loading (accelerator.unwrap_model)
- Random seed is set for reproducibility
- Generated CT is clamped to [-3.0, 3.0] range (matches training)
- CFG++ with manifold constraints (rescaling to match conditional statistics)
- Dynamic & adaptive CFG scaling (decreases from max to min during sampling)
- Only raw z-score normalized output is saved

CFG++ Implementation:
- Rescales CFG output to match conditional prediction statistics
- Prevents oversaturation and maintains better manifold adherence
- Configurable rescale factor (0.0 = standard CFG, 1.0 = full rescale)

Dynamic CFG Scaling:
- Starts with high guidance (cfg_scale_max) for strong conditioning
- Linearly decreases to low guidance (cfg_scale_min) for better details
- Improves quality by balancing global structure and fine details

Usage:
    # Configure the parameters in the config dictionary below, then run:
    conda run -n edm python inference_volume.py
    
    # For single file mode, set:
    #   mode = "single"
    #   input_mri = "/path/to/mri.nii"
    #   output_ct = "/path/to/output_ct.nii"
    
    # For batch directory mode, set:
    #   mode = "directory"
    #   input_dir = "/path/to/mri_volumes"
    #   output_dir = "/path/to/output_cts"
    #   pattern = "mr.nii"
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from tqdm import tqdm

# MONAI for consistent data loading with training
from monai.transforms import LoadImage, EnsureChannelFirst, Orientation, Compose

# Hugging Face Accelerate
from accelerate import Accelerator

# Import model
from models import create_swin_fir_unet


# Set random seed for reproducibility
def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


# =============================================================================
# CONFIGURATION
# =============================================================================

config = {
    # Mode: "single" for one file, "directory" for batch processing
    "mode": "single",  # "single" or "directory"
    
    # Checkpoint
    "checkpoint": "./outputs_accelerate/20260112_000305_pfgm_fir/checkpoints/best",
    
    # Single file mode (used when mode="single")
    # "input_mri": "/mnt/nvme/lloyd/DL project CT-MRI/pfgm_test_brain/1BA336/mr.nii_processed_patch.nii.gz",  # CHANGE THIS
    # "output_ct": "/mnt/nvme/lloyd/DL project CT-MRI/pfgm_test_brain/1BA336/generated_ct.nii.gz",
    
    # Directory mode (used when mode="directory")
    "input_dir": "/mnt/nvme/lloyd/DL project CT-MRI/pfgm_train_brain",
    "output_dir": "./inference_output",
    "pattern": "mr.nii",  # Filename pattern to match MRI files
    
    # Model Architecture (must match training config)
    "base_channels": 128,
    "channel_mults": [1, 2, 4, 8],
    "num_res_blocks": 1,
    "attention_resolutions": [],
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "radius_emb_dim": 256,
    "window_size": 4,
    "num_swin_blocks": 4,
    "num_sfb_blocks": 1,
    "num_bottleneck_blocks": 4,
    "use_sfb": True,
    "dropout": 0.0,
    "drop_path": 0.0,
    
    # Sampling Parameters
    "num_steps": 50,
    "cfg_scale": 2.0,
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "sigma_data": 1.0,
    "rho": 7.0,
    "pfgmpp": True,
    "D": 128,
    
    # CFG++ and Dynamic Scaling
    "use_cfg_plus_plus": True,  # Use CFG++ manifold constraints
    "cfg_rescale_factor": 0.7,  # Rescale factor for CFG++ (0.0 = no rescale, 1.0 = full rescale)
    "use_dynamic_cfg": True,  # Use dynamic CFG scaling
    "cfg_scale_min": 1.0,  # Minimum CFG scale (at end of sampling)
    "cfg_scale_max": 3.0,  # Maximum CFG scale (at start of sampling)
    
    # Processing Parameters
    "image_size": 256,
    "batch_size": 40,  # Number of slices to process at once
    "device": "cuda:1",  # None for auto-detect, or "cuda" / "cpu"
    "seed": 42,  # Random seed for reproducibility
}


# =============================================================================
# EDM Functions (from train_accelerate.py)
# =============================================================================

def edm_precond(
    model: nn.Module,
    x: torch.Tensor,
    mri: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
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
    F_x = model(x_scaled, mri, c_noise)
    D_x = c_skip * x + c_out * F_x
    
    return D_x


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
    """Sample from PFGM++ prior distribution for sampling initialization."""
    N = channels * height * width
    r_max = sigma_max * np.sqrt(D)
    
    samples_norm = np.random.beta(a=N / 2., b=D / 2., size=batch_size).astype(np.float64)
    samples_norm = torch.from_numpy(samples_norm).to(device)
    inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
    
    radius = r_max * torch.sqrt(inverse_beta + 1e-8)
    
    gaussian = torch.randn(batch_size, N, device=device, dtype=torch.float64)
    unit_gaussian = gaussian / (torch.norm(gaussian, p=2, dim=1, keepdim=True) + 1e-8)
    
    latent = unit_gaussian * radius.view(batch_size, 1)
    latent = latent.view(batch_size, channels, height, width).to(dtype)
    
    return latent


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
    cfg_scale: float = 1.0,
    pfgmpp: bool = False,
    D: int = 128,
    # CFG++ and Dynamic Scaling options
    use_cfg_plus_plus: bool = False,
    cfg_rescale_factor: float = 0.7,
    use_dynamic_cfg: bool = False,
    cfg_scale_min: float = 1.0,
    cfg_scale_max: float = 3.0,
) -> torch.Tensor:
    """
    Generate CT images from MRI using EDM/PFGM++ sampling.
    
    Enhanced with:
    - CFG++ (manifold constraints): Rescales CFG output to match conditional statistics
    - Dynamic CFG scaling: Linearly decreases guidance from cfg_scale_max to cfg_scale_min
    
    Args:
        model: Denoising model
        mri: Conditioning MRI images [B, 1, H, W]
        num_steps: Number of sampling steps
        sigma_min/max: Noise level range
        sigma_data: Data standard deviation for EDM preconditioning
        rho: Schedule parameter
        S_churn/S_min/S_max/S_noise: Stochasticity parameters
        cfg_scale: Base CFG scale (used if dynamic_cfg=False)
        pfgmpp: Use PFGM++ prior
        D: PFGM++ augmented dimension
        use_cfg_plus_plus: Enable CFG++ manifold constraints
        cfg_rescale_factor: Rescale factor for CFG++ (0=standard, 1=full rescale)
        use_dynamic_cfg: Enable dynamic CFG scaling
        cfg_scale_min: Minimum CFG scale (at end of sampling)
        cfg_scale_max: Maximum CFG scale (at start of sampling)
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
        denoised_cond = edm_precond(model, x_hat.float(), mri, sigma_batch, sigma_data).double()
        
        # Classifier-free guidance (standard or CFG++)
        if current_cfg_scale != 1.0:
            denoised_uncond = edm_precond(model, x_hat.float(), mri_uncond, sigma_batch, sigma_data).double()
            
            if use_cfg_plus_plus:
                denoised = apply_cfg_plus_plus(denoised_cond, denoised_uncond, 
                                                current_cfg_scale, cfg_rescale_factor)
            else:
                denoised = denoised_uncond + current_cfg_scale * (denoised_cond - denoised_uncond)
        else:
            denoised = denoised_cond
        
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        # Heun's 2nd order correction
        if i < num_steps - 1 and t_next > 0:
            sigma_batch = torch.full((B,), t_next.item(), device=device, dtype=torch.float32)
            denoised_cond_next = edm_precond(model, x_next.float(), mri, sigma_batch, sigma_data).double()
            
            # CFG for Heun correction
            if current_cfg_scale != 1.0:
                denoised_uncond_next = edm_precond(model, x_next.float(), mri_uncond, sigma_batch, sigma_data).double()
                
                if use_cfg_plus_plus:
                    denoised_next = apply_cfg_plus_plus(denoised_cond_next, denoised_uncond_next,
                                                         current_cfg_scale, cfg_rescale_factor)
                else:
                    denoised_next = denoised_uncond_next + current_cfg_scale * (denoised_cond_next - denoised_uncond_next)
            else:
                denoised_next = denoised_cond_next
            
            d_prime = (x_next - denoised_next) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    
    return x_next.float()


# =============================================================================
# Volume Processing Functions
# =============================================================================

def load_model(checkpoint_path: str, cfg: Dict, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    print(f"Target device: {device}")
    
    # Build model
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
    )
    
    # Try loading checkpoint - check for safetensors first (accelerator default), then pytorch_model.bin
    safetensors_file = Path(checkpoint_path) / "model.safetensors"
    pytorch_file = Path(checkpoint_path) / "pytorch_model.bin"
    
    if safetensors_file.exists():
        print(f"Loading weights from {safetensors_file}")
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_file))
        
        # Diagnostic: Check state_dict keys
        sample_keys = list(state_dict.keys())[:5]
        print(f"Checkpoint keys (first 5): {sample_keys}")
        print(f"Total keys in checkpoint: {len(state_dict.keys())}")
        
        # Check if keys have wrapper prefix (module., _orig_mod., etc)
        has_wrapper = any(k.startswith('module.') or k.startswith('_orig_mod.') for k in state_dict.keys())
        print(f"Checkpoint has wrapper prefix: {has_wrapper}")
        
        # If wrapped, unwrap the state dict
        if has_wrapper:
            print("Unwrapping checkpoint keys...")
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '').replace('_orig_mod.', '')
                new_state_dict[new_key] = v
            state_dict = new_state_dict
        
        # Check model keys
        model_keys = list(model.state_dict().keys())[:5]
        print(f"Model keys (first 5): {model_keys}")
        print(f"Total keys in model: {len(model.state_dict().keys())}")
        
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"WARNING: Missing keys ({len(missing_keys)}): {missing_keys[:5]}")
            if unexpected_keys:
                print(f"WARNING: Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}")
            if not missing_keys and not unexpected_keys:
                print("✓ All keys matched perfectly!")
            print("Loaded weights from model.safetensors")
        except Exception as e:
            print(f"ERROR loading state_dict: {e}")
            raise
    elif pytorch_file.exists():
        print(f"Loading weights from {pytorch_file}")
        state_dict = torch.load(pytorch_file, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        print("Loaded weights from pytorch_model.bin")
    else:
        # Fallback to accelerator method
        print("No direct model file found, using accelerator.load_state()")
        accelerator = Accelerator()
        model = accelerator.prepare(model)
        accelerator.load_state(checkpoint_path)
        model = accelerator.unwrap_model(model)
        print("Loaded via accelerator and unwrapped")
    
    # Move ALL model parameters and buffers to specified device
    model = model.to(device)
    
    # Set to eval mode BEFORE checking
    model.eval()
    
    # Force disable dropout and other training-specific layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
            module.train(False)
    
    # Verify model is on correct device
    device_issues = []
    for name, param in model.named_parameters():
        if param.device != device:
            device_issues.append(name)
            param.data = param.data.to(device)
    
    if device_issues:
        print(f"Warning: Moved {len(device_issues)} parameters to {device}")
    
    # Check if model has reasonable weights (not all zeros or NaN)
    total_params = 0
    zero_params = 0
    nan_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if torch.all(param == 0):
            zero_params += 1
        if torch.any(torch.isnan(param)):
            nan_params += 1
            print(f"Warning: Parameter {name} contains NaN!")
    
    print(f"Model loaded successfully on {device}!")
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"Parameter check: {total_params} total, {zero_params} all-zero, {nan_params} with NaN")
    
    # Verify model is in eval mode
    print(f"Model training mode: {model.training}")
    
    # Quick sanity check: test forward pass
    with torch.no_grad():
        test_x = torch.randn(1, 1, 256, 256, device=device)
        test_mri = torch.randn(1, 1, 256, 256, device=device)
        test_sigma = torch.tensor([1.0], device=device)
        try:
            test_out = edm_precond(model, test_x, test_mri, test_sigma, sigma_data=1.0)
            print(f"Test forward pass: output shape {test_out.shape}, range [{test_out.min():.3f}, {test_out.max():.3f}]")
            if torch.all(test_out == 0):
                print("ERROR: Model outputs all zeros!")
            elif torch.any(torch.isnan(test_out)):
                print("ERROR: Model outputs contain NaN!")
        except Exception as e:
            print(f"ERROR in test forward pass: {e}")
    
    return model


def normalize_volume(volume: np.ndarray, method: str = 'none') -> np.ndarray:
    """
    Normalize MRI volume.
    
    IMPORTANT: The training data is already z-score normalized in the NIfTI files.
    Using method='none' skips normalization to match training behavior.
    
    Args:
        volume: Input volume [D, H, W]
        method: 'none' (default, data already normalized), 'z_score', or 'min_max'
    
    Returns:
        Normalized volume
    """
    if method == 'none':
        # Data is already normalized - use as-is (matches training)
        return volume
    elif method == 'z_score':
        mean = volume.mean()
        std = volume.std()
        normalized = (volume - mean) / (std + 1e-8)
    elif method == 'min_max':
        vmin = volume.min()
        vmax = volume.max()
        normalized = (volume - vmin) / (vmax - vmin + 1e-8)
        normalized = normalized * 2 - 1  # Scale to [-1, 1]
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def denormalize_volume(volume: np.ndarray, method: str = 'z_score') -> np.ndarray:
    """
    Denormalize generated CT volume.
    
    For display/saving purposes. The model outputs in normalized space.
    """
    # CT is already in normalized space, we keep it as is
    # or apply inverse normalization if needed
    return volume


def process_volume(
    mri_path: str,
    model: nn.Module,
    output_path: str,
    cfg: Dict,
    device: torch.device = None,
) -> None:
    """
    Process entire MRI volume to generate CT volume.
    
    Args:
        mri_path: Path to input MRI NIfTI file
        model: Loaded model
        output_path: Path to save output CT NIfTI file
        cfg: Configuration dictionary
        device: Device to use
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nProcessing volume: {mri_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Load MRI volume using MONAI transforms (MUST match training!)
    # Training uses: LoadImaged -> EnsureChannelFirstd -> Orientationd(axcodes='RAS')
    # This ensures the same orientation/axis order as training
    load_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes='RAS'),  # Critical: match training orientation
    ])
    
    mri_vol = load_transforms(mri_path)  # [C, H, W, D] tensor
    
    # Also load with nibabel to get affine/header for saving
    mri_nifti = nib.load(mri_path)
    affine = mri_nifti.affine
    header = mri_nifti.header
    
    # Convert to numpy, remove channel dim
    mri_data = mri_vol.numpy()[0]  # [H, W, D]
    
    print(f"MRI volume shape: {mri_data.shape}")
    print(f"MRI data range: [{mri_data.min():.2f}, {mri_data.max():.2f}]")
    print(f"MRI mean: {mri_data.mean():.3f}, std: {mri_data.std():.3f}")
    print(f"MRI orientation: RAS (reoriented to match training)")
    
    # Use data as-is (already z-score normalized in the NIfTI files)
    # This matches the training data loader behavior
    mri_normalized = normalize_volume(mri_data, method='none')
    
    # Get dimensions (NIfTI format is typically [H, W, D])
    height, width, depth = mri_data.shape
    
    # Prepare output volume
    ct_volume = np.zeros_like(mri_normalized)
    
    # Process slices in batches (along last axis, matching training)
    num_slices = depth
    batch_size = cfg["batch_size"]
    image_size = cfg["image_size"]
    num_batches = (num_slices + batch_size - 1) // batch_size
    
    print(f"Processing {num_slices} slices in {num_batches} batches (batch_size={batch_size})...")
    print("DEBUG MODE: Processing only first batch to save time")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating CT slices"):
            # Get batch slice indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_slices)
            current_batch_size = end_idx - start_idx
            
            # Extract MRI slices (along last axis, matching training)
            mri_slices = []
            for i in range(start_idx, end_idx):
                mri_slice = mri_normalized[:, :, i]  # [H, W] - slice along last axis
                mri_slices.append(mri_slice)
            
            # Stack and convert to tensor
            mri_batch = np.stack(mri_slices, axis=0)  # [B, H, W]
            mri_batch = torch.from_numpy(mri_batch).float().unsqueeze(1)  # [B, 1, H, W]
            
            # Resize using torch.interpolate (MUST match training!)
            # Training uses: torch.nn.functional.interpolate(mode='bilinear', align_corners=False)
            if mri_batch.shape[2] != image_size or mri_batch.shape[3] != image_size:
                mri_batch = torch.nn.functional.interpolate(
                    mri_batch,
                    size=(image_size, image_size),
                    mode='bilinear',
                    align_corners=False
                )
            
            mri_batch = mri_batch.to(device)
            
            # Generate CT slices
            ct_batch = edm_sample(
                model=model,
                mri=mri_batch,
                num_steps=cfg["num_steps"],
                sigma_min=cfg["sigma_min"],
                sigma_max=cfg["sigma_max"],
                sigma_data=cfg["sigma_data"],
                rho=cfg["rho"],
                cfg_scale=cfg["cfg_scale"],
                pfgmpp=cfg["pfgmpp"],
                D=cfg["D"],
                # CFG++ and Dynamic Scaling
                use_cfg_plus_plus=cfg.get("use_cfg_plus_plus", False),
                cfg_rescale_factor=cfg.get("cfg_rescale_factor", 0.7),
                use_dynamic_cfg=cfg.get("use_dynamic_cfg", False),
                cfg_scale_min=cfg.get("cfg_scale_min", 1.0),
                cfg_scale_max=cfg.get("cfg_scale_max", 3.0),
            )
            
            # Clamp output to reasonable range (matches training)
            ct_batch = torch.clamp(ct_batch, -3.0, 3.0)
            
            # Convert back to numpy and store
            ct_batch = ct_batch.cpu().numpy()  # [B, 1, H, W]
            

            for i, slice_idx in enumerate(range(start_idx, end_idx)):
                ct_slice = ct_batch[i, 0, :, :]  # [H, W]
                
                # Resize back to original size if needed (using torch.interpolate for consistency)
                if ct_slice.shape[0] != height or ct_slice.shape[1] != width:
                    ct_slice_tensor = torch.from_numpy(ct_slice).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    ct_slice_tensor = torch.nn.functional.interpolate(
                        ct_slice_tensor,
                        size=(height, width),
                        mode='bilinear',
                        align_corners=False
                    )
                    ct_slice = ct_slice_tensor.squeeze().numpy()
                
                # Store in output volume (along last axis, matching input)
                ct_volume[:, :, slice_idx] = ct_slice
    
    print(f"CT volume generated! Shape: {ct_volume.shape}")
    print(f"CT data range: [{ct_volume.min():.2f}, {ct_volume.max():.2f}]")
    
    # Save CT volume (normalized z-score space, clamped to [-3, 3])
    ct_nifti = nib.Nifti1Image(ct_volume, affine, header)
    nib.save(ct_nifti, output_path)
    print(f"CT volume saved to: {output_path}")


def process_directory(
    input_dir: str,
    output_dir: str,
    model: nn.Module,
    cfg: Dict,
    device: torch.device = None,
) -> None:
    """
    Process all MRI volumes in a directory.
    
    Args:
        input_dir: Directory containing MRI volumes
        output_dir: Directory to save generated CT volumes
        model: Loaded model
        cfg: Configuration dictionary
        device: Device to use
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pattern = cfg["pattern"]
    
    # Find all MRI files
    mri_files = list(input_path.rglob(f"*{pattern}*"))
    
    if len(mri_files) == 0:
        print(f"No files matching pattern '{pattern}' found in {input_dir}")
        return
    
    print(f"\nFound {len(mri_files)} MRI volumes to process")
    
    for mri_file in mri_files:
        # Determine output filename
        relative_path = mri_file.relative_to(input_path)
        output_file = output_path / str(relative_path).replace(pattern, "ct_generated.nii")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process volume
        try:
            process_volume(
                mri_path=str(mri_file),
                model=model,
                output_path=str(output_file),
                cfg=cfg,
                device=device,
            )
        except Exception as e:
            print(f"Error processing {mri_file}: {e}")
            continue
    
    print(f"\nProcessing complete! All outputs saved to: {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main inference function."""
    cfg = config
    
    # Set random seed for reproducibility
    set_seed(cfg.get("seed", 42))
    
    # Set device
    if cfg["device"]:
        device = torch.device(cfg["device"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print("PFGM-FIR Volume Inference")
    print("="*60)
    print(f"Mode: {cfg['mode']}")
    print(f"Checkpoint: {cfg['checkpoint']}")
    print(f"Device: {device}")
    print(f"Sampling steps: {cfg['num_steps']}")
    print(f"CFG scale: {cfg['cfg_scale']}")
    print(f"CFG++: {cfg.get('use_cfg_plus_plus', False)} (rescale={cfg.get('cfg_rescale_factor', 0.7)})")
    print(f"Dynamic CFG: {cfg.get('use_dynamic_cfg', False)} (max={cfg.get('cfg_scale_max', 3.0)} → min={cfg.get('cfg_scale_min', 1.0)})")
    print(f"PFGMPP: {cfg['pfgmpp']}")
    print(f"Image size: {cfg['image_size']}")
    print(f"Batch size: {cfg['batch_size']}")
    print("="*60 + "\n")
    
    # Load model
    model = load_model(cfg["checkpoint"], cfg, device)
    
    # Process based on mode
    if cfg["mode"] == "single":
        # Single file mode
        print(f"Input MRI: {cfg['input_mri']}")
        print(f"Output CT: {cfg['output_ct']}")
        
        # Create output directory if needed
        output_path = Path(cfg["output_ct"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        process_volume(
            mri_path=cfg["input_mri"],
            model=model,
            output_path=cfg["output_ct"],
            cfg=cfg,
            device=device,
        )
    elif cfg["mode"] == "directory":
        # Directory mode
        print(f"Input directory: {cfg['input_dir']}")
        print(f"Output directory: {cfg['output_dir']}")
        print(f"Pattern: {cfg['pattern']}")
        
        process_directory(
            input_dir=cfg["input_dir"],
            output_dir=cfg["output_dir"],
            model=model,
            cfg=cfg,
            device=device,
        )
    else:
        raise ValueError(f"Unknown mode: {cfg['mode']}. Must be 'single' or 'directory'")
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)


if __name__ == "__main__":
    for k in ["1BB184", "1BC085", "1BC086"]:
        print(f"Processing subject: {k}")
        config["input_mri"] = f"/mnt/nvme/lloyd/DL project CT-MRI/pfgm_test_brain/{k}/mr.nii_processed_patch.nii.gz"
        config["output_ct"] = f"/mnt/nvme/lloyd/DL project CT-MRI/pfgm_test_brain/{k}/generated_ct.nii.gz"
        main()
