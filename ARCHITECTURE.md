# PFGM-FIR: PFGM++ with SwinFIR Blocks for MRI-to-CT Synthesis

## Overview

This project combines:
1. **PFGM++ (Poisson Flow Generative Model)** - For the diffusion-like training framework
2. **SwinFIR blocks** - For the neural network architecture with Fast Fourier Convolution
3. **UNet structure** - From `Diffusion_model_transformer.py` for encoder-decoder with skip connections

## Architecture Design

### High-Level Structure

```
Input: [noisy_CT (1ch) | MRI_condition (1ch)] = 2 channels
       ↓
   Conv3x3 (shallow feature extraction)
       ↓
┌─────────────────────────────────────────────────────────────┐
│                         ENCODER                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Stage 0: FIRResBlock × N + Downsample                   ││
│  │   - Resolution: H × W                                   ││
│  │   - Channels: base_ch                                   ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Stage 1: FIRResBlock × N + Downsample                   ││
│  │   - Resolution: H/2 × W/2                               ││
│  │   - Channels: base_ch × 2                               ││
│  │   - Swin Attention: ENABLED                             ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Stage 2: FIRResBlock × N + Downsample                   ││
│  │   - Resolution: H/4 × W/4                               ││
│  │   - Channels: base_ch × 4                               ││
│  │   - Swin Attention: ENABLED                             ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Stage 3: FIRResBlock × N (no downsample)                ││
│  │   - Resolution: H/8 × W/8                               ││
│  │   - Channels: base_ch × 8                               ││
│  │   - Swin Attention: ENABLED                             ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│                       BOTTLENECK                             │
│   FIRResBlock × 2 with Swin Attention + SFB                 │
│   - Resolution: H/8 × W/8                                   │
│   - Channels: base_ch × 8                                   │
└─────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│                         DECODER                              │
│  (Mirror of encoder with skip connections)                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Stage 3: Concat(skip) + FIRResBlock × N + Upsample      ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Stage 2: Concat(skip) + FIRResBlock × N + Upsample      ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Stage 1: Concat(skip) + FIRResBlock × N + Upsample      ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Stage 0: Concat(skip) + FIRResBlock × N                 ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
       ↓
   Conv3x3 (output projection)
       ↓
Output: predicted vector field (1 channel)
```

---

## Key Components from SwinFIR

### 1. SFB (Spatial-Frequency Block) - The Core Innovation

The SFB block is what gives SwinFIR its "Fast Fourier" capability:

```python
class SFB(nn.Module):
    """Spatial-Frequency Block: combines local (spatial) and global (frequency) features."""
    
    def __init__(self, embed_dim):
        self.S = ResB(embed_dim)           # Spatial branch: standard residual conv
        self.F = SpectralTransform(embed_dim)  # Frequency branch: FFT-based
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 1)
    
    def forward(self, x):
        s = self.S(x)                      # Local features via convolution
        f = self.F(x)                      # Global features via FFT
        out = torch.cat([s, f], dim=1)     # Concatenate
        out = self.fusion(out)             # Fuse with 1x1 conv
        return out
```

**Why this matters:**
- Standard Swin attention is window-limited (e.g., 7×7)
- FFT gives **image-wide receptive field** instantly
- Combining both gives local detail + global context

### 2. SpectralTransform (Fast Fourier Convolution)

```python
class SpectralTransform(nn.Module):
    """Applies convolution in frequency domain for global receptive field."""
    
    def forward(self, x):
        # 1. Transform to frequency domain
        ffted = torch.fft.rfftn(x, dim=(-2, -1))
        
        # 2. Apply convolution in frequency space
        ffted = self.conv_layer(ffted)
        
        # 3. Transform back to spatial domain
        output = torch.fft.irfftn(ffted)
        
        return output
```

**Key insight:** Convolution in frequency domain = multiplication, which is computationally efficient and captures global patterns.

### 3. SwinTransformerBlock with Shifted Windows

```python
class SwinTransformerBlock(nn.Module):
    """Standard Swin block with window attention + shifted window attention."""
    
    # Alternates between:
    # - Window-MSA (shift_size=0): attention within windows
    # - SW-MSA (shift_size=window_size//2): shifted windows for cross-window interaction
```

---

## Modifications for PFGM++

### 1. Radius Conditioning

PFGM++ uses radius `r` instead of timestep `t`. We inject this via:

```python
class RadiusEmbedding(nn.Module):
    """Fourier embedding for radius, similar to timestep embedding in diffusion."""
    
    def forward(self, r):
        # Fourier features: [sin(2πr·f), cos(2πr·f)] for multiple frequencies
        freqs = torch.exp(-math.log(max_freq) * torch.arange(dim//2) / (dim//2))
        args = r[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

Then project to each block via **FiLM conditioning**:
```python
# In FIRResBlock:
scale, shift = self.radius_proj(r_emb).chunk(2, dim=1)
h = h * (1 + scale) + shift  # FiLM: Feature-wise Linear Modulation
```

### 2. MRI Conditioning

Following the Synthetic-CT approach, MRI is concatenated at input:

```python
def forward(self, noisy_ct, radius, mri_condition):
    # Concatenate noisy CT and MRI condition
    x = torch.cat([noisy_ct, mri_condition], dim=1)  # [B, 2, H, W]
    
    # Process through network
    x = self.conv_first(x)  # [B, embed_dim, H, W]
    ...
```

---

## FIRResBlock: The Main Building Block

Each FIRResBlock combines:

1. **Convolution** with radius conditioning
2. **Swin Attention** (optional, for deeper layers)
3. **SFB residual connection** for global receptive field

```python
class FIRResBlock(nn.Module):
    """Residual block with SwinFIR components and radius conditioning."""
    
    def __init__(self, channels, radius_emb_dim, use_swin=False, use_sfb=True):
        # Convolution layers
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        # Radius conditioning (FiLM)
        self.radius_proj = nn.Linear(radius_emb_dim, channels * 2)
        
        # Optional Swin attention
        if use_swin:
            self.swin = SwinTransformerBlock(...)
        
        # SFB for residual connection (global receptive field)
        if use_sfb:
            self.sfb = SFB(channels)
        else:
            self.sfb = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x, r_emb):
        # Main path
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Radius conditioning via FiLM
        scale, shift = self.radius_proj(r_emb).chunk(2, dim=-1)
        scale = scale[..., None, None]
        shift = shift[..., None, None]
        h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # Optional Swin attention
        if self.swin is not None:
            h = self.swin(h)
        
        # Residual with SFB (global features)
        return self.sfb(x) + h
```

---

## Training Pipeline

The training follows the same structure as `train_swin_pfgmpp_2d.py`:

1. **Data**: MRI-CT pairs loaded via MONAI
2. **Perturbation**: PFGM++ perturbation (add noise based on radius `r`)
3. **Loss**: MSE between predicted and target vector field
4. **ODE Sampling**: Euler integration from r_max to 0

```python
# Perturbation
noisy_ct = clean_ct + (r / sqrt(D)) * noise  # PFGM++ forward

# Target
target_field = (noisy_ct - clean_ct) / (r / sqrt(D))  # What model learns

# Loss
pred_field = model(noisy_ct, r, mri_condition)
loss = MSE(pred_field, target_field)

# Sampling (ODE)
for r in linspace(r_max, 0, steps):
    pred_field = model(x, r, mri)
    x = x + dr * pred_field / sqrt(D)  # Euler step
```

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `D` | 64 | Augmented dimension for PFGM++ |
| `sigma_max` | 2.0 | Controls r_max = sigma_max × √D ≈ 16 |
| `base_channels` | 128 | Initial feature channels |
| `channel_mults` | (1, 2, 3, 4) | Channel multipliers per stage |
| `num_res_blocks` | 2 | ResBlocks per stage |
| `window_size` | 8 | Swin attention window size |
| `num_heads` | 4 | Attention heads |
| `use_sfb_stages` | (False, True, True, True) | Which stages use SFB |
| `learning_rate` | 1e-4 | AdamW optimizer |
| `batch_size` | 4 | Per GPU |

---

## File Structure

```
pfgm_fir/
├── ARCHITECTURE.md           # This file
├── README.md                 # Quick start guide
├── requirements.txt          # Dependencies
├── __init__.py
│
├── models/
│   ├── __init__.py
│   ├── fir_network.py        # Main SwinFIR-PFGM++ network
│   ├── fir_blocks.py         # FIRResBlock, SFB, SpectralTransform
│   ├── swin_attention.py     # Swin attention components
│   └── embeddings.py         # Radius embedding
│
├── pfgm/
│   ├── __init__.py
│   ├── perturbation.py       # PFGM++ perturbation & loss
│   └── sampling.py           # ODE sampling
│
├── data/
│   ├── __init__.py
│   └── loader.py             # MONAI-based data loader
│
├── train_pfgm_fir.py         # Main training script
├── inference_pfgm_fir.py     # Inference script
└── outputs/                  # Training outputs (logs, checkpoints)
```

---

## Comparison with Other Architectures

| Feature | Original Swin | SwinFIR | PFGM-FIR (Ours) |
|---------|--------------|---------|-----------------|
| Receptive Field | Window-limited | Global via FFT | Global via FFT |
| Conditioning | None | None | Radius (PFGM++) |
| Task | Classification/SR | Super-Resolution | MRI→CT Synthesis |
| Structure | Hierarchical | Residual body | UNet |
| Skip Connections | No | No | Yes |
| Frequency Branch | No | SFB | SFB |

---

## References

1. **SwinFIR**: Zhang et al., "SwinFIR: Revisiting the SwinIR with Fast Fourier Convolution and Improved Training for Image Super-Resolution" (2022)
2. **PFGM++**: Xu et al., "PFGM++: Unlocking the Potential of Physics-Inspired Generative Models" (2023)
3. **Synthetic-CT Transformer**: The 3D transformer-based denoising diffusion model for MRI-to-CT synthesis
4. **SwinIR**: Liang et al., "SwinIR: Image Restoration Using Swin Transformer" (2021)
