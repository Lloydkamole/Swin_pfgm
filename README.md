# PFGM-FIR: SwinFIR-based PFGM++ for MRI-to-CT Synthesis

A hybrid architecture combining **SwinFIR** (Swin Transformer with Fast Fourier Convolution) and **PFGM++** (Poisson Flow Generative Model) for medical image synthesis.

## Overview

This project synthesizes CT images from MRI scans using a diffusion-based generative model with:

- **PFGM++ framework**: Improved sampling via Poisson Flow with augmented dimensions
- **SwinFIR blocks**: Swin Transformer attention + Spatial-Frequency Block (SFB) for global receptive field
- **UNet architecture**: Encoder-decoder with skip connections

## Key Features

### From SwinFIR
- **SFB (Spatial-Frequency Block)**: Combines local convolution with FFT-based global features
- **FourierUnit**: Convolution in frequency domain for image-wide receptive field
- **Window attention**: Efficient Swin Transformer attention with shifted windows

### From PFGM++
- **Radius conditioning**: Network conditioned on PFGM++ radius parameter (not timestep)
- **Augmented dimension**: D=128 for stable training
- **ODE-based sampling**: Euler integration from noise to clean image

## Project Structure

```
pfgm_fir/
├── ARCHITECTURE.md          # Detailed architecture documentation
├── README.md                 # This file
├── train_pfgm_fir.py        # Training script
├── test_models.py           # Model tests
└── models/
    ├── __init__.py          # Module exports
    ├── embeddings.py        # RadiusEmbedding (Fourier features + MLP)
    ├── swin_attention.py    # WindowAttention, SwinTransformerBlock
    ├── fir_blocks.py        # FourierUnit, SFB, FIRResBlock
    └── network.py           # SwinFIRUNet (main model)
```

## Installation

```bash
# Create environment
conda create -n pfgm_fir python=3.9
conda activate pfgm_fir

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install monai tensorboard numpy scipy
```

## Usage

### Training

```bash
python train_pfgm_fir.py \
    --data_dir /path/to/mri_ct_data \
    --output_dir ./outputs \
    --model_config base \
    --batch_size 4 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --D 128 \
    --sigma_max 2.0
```

### Model Configurations

| Config | Base Channels | Channel Mults | Params |
|--------|--------------|---------------|--------|
| small  | 48           | (1,2,4,4)    | ~15M   |
| base   | 64           | (1,2,4,8)    | ~40M   |
| large  | 96           | (1,2,4,8)    | ~90M   |

### Testing Models

```bash
cd pfgm_fir
python test_models.py
```

## Architecture Details

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation including:
- Complete architecture diagrams
- SFB block explanation
- PFGM++ mathematics
- Implementation notes

## Key Components

### SFB (Spatial-Frequency Block)
```python
# Combines spatial (local) and frequency (global) features
class SFB(nn.Module):
    def forward(self, x):
        s = self.ResB(x)              # Local via convolution
        f = self.SpectralTransform(x)  # Global via FFT
        return self.fusion(cat([s, f]))
```

### FIRResBlock
```python
# Main building block with radius conditioning
class FIRResBlock(nn.Module):
    def forward(self, x, r_emb):
        h = self.conv1(self.norm1(x))
        h = h * (1 + scale) + shift    # FiLM conditioning
        h = self.conv2(self.norm2(h))
        if self.use_swin:
            h = self.swin_blocks(h)
        return self.sfb(skip) + h       # SFB residual
```

### SwinFIRUNet
```python
# Full UNet with skip connections
Input: [noisy_CT, MRI] → 2 channels
↓ Conv
↓ Encoder (FIRResBlock stages + downsample)
↓ Bottleneck (FIRResBlock with attention)
↓ Decoder (upsample + skip concat + FIRResBlock)
↓ Conv
Output: predicted vector field → 1 channel
```

## PFGM++ Training

The model is trained to predict the direction back to clean data:

1. Sample clean CT `x` and MRI condition
2. Sample radius `r` from distribution
3. Perturb: `x_noisy = x + (r/√D) * u` where `u` is random unit vector
4. Model predicts `-u` given `(x_noisy, MRI, r)`
5. Loss: MSE between prediction and target direction

## Sampling

Generate CT from MRI using Euler ODE integration:

```python
x = sample_noise(r_max)
for r in [r_max → r_min]:
    v = model(x, mri, r)
    x = x + (v / √D) * dr
return x
```

## References

- **SwinFIR**: [Spatial-Frequency Integration for Image Restoration](https://github.com/Zdafeng/SwinFIR)
- **PFGM++**: [Poisson Flow Generative Models](https://arxiv.org/abs/2302.02552)
- **Swin Transformer**: [Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- **Diffusion for Medical Imaging**: [3D Transformer-based Denoising Diffusion Model](https://github.com/Synthetic-CT-generation)

## License

MIT License
