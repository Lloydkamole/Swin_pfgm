"""
PFGM-FIR Models: SwinFIR-based architecture for PFGM++ MRI-to-CT synthesis.

This module contains the neural network components for PFGM-FIR:
- SwinFIRUNet: Main UNet model with SwinFIR blocks
- MPSwinFIRUNet: EDM2-style magnitude-preserving UNet
- FIRResBlock: Core building block with Swin attention + SFB
- MPFIRResBlock: Magnitude-preserving version of FIRResBlock
- RadiusEmbedding: PFGM++ radius conditioning
- SFB: Spatial-Frequency Block for global receptive field
- FourierUnit: FFT-based processing
"""

from .embeddings import RadiusEmbedding, FourierEmbedding, TimestepEmbedding, timestep_embedding
from .swin_attention import (
    SwinTransformerBlock,
    AdaLNZeroSwinTransformerBlock,
    WindowAttention,
    window_partition,
    window_reverse,
)
from .deformable_attention import (
    DeformableAttention,
    AdaLNZeroDeformableBlock,
)
from .fir_blocks import (
    FourierUnit,
    SpectralTransform,
    ResB,
    SFB,
    CAB,
    ChannelAttention,
    FIRResBlock,
    Downsample,
    Upsample,
    LayerNorm2d,
    AdaGroupNorm,
)
from .network import SwinFIRUNet, SwinFIRUNetConfig, create_swin_fir_unet

# EDM2-style magnitude-preserving models
from .mp_layers import (
    normalize,
    mp_silu,
    mp_sum,
    mp_cat,
    MPFourier,
    MPConv,
    MPSiLU,
)
from .mp_fir_blocks import (
    PixelNorm,
    MPFourierUnit,
    MPSpectralTransform,
    MPSFB,
    MPFIRResBlock,
    MPDownsample,
    MPUpsample,
)
from .mp_network import MPSwinFIRUNet, create_mp_swinfir_unet

__all__ = [
    # Embeddings
    'RadiusEmbedding',
    'FourierEmbedding',
    'TimestepEmbedding',
    'timestep_embedding',
    # Swin attention
    'SwinTransformerBlock',
    'AdaLNZeroSwinTransformerBlock',
    'WindowAttention',
    'window_partition',
    'window_reverse',
    # Deformable attention
    'DeformableAttention',
    'AdaLNZeroDeformableBlock',
    # FIR blocks
    'FourierUnit',
    'SpectralTransform',
    'ResB',
    'SFB',
    'CAB',
    'ChannelAttention',
    'FIRResBlock',
    'Downsample',
    'Upsample',
    'AdaGroupNorm',
    # Main network
    'SwinFIRUNet',
    'SwinFIRUNetConfig',
    'create_swin_fir_unet',
    # EDM2 magnitude-preserving layers
    'normalize',
    'mp_silu',
    'mp_sum',
    'mp_cat',
    'MPFourier',
    'MPConv',
    'MPSiLU',
    # EDM2 magnitude-preserving blocks
    'PixelNorm',
    'MPFourierUnit',
    'MPSpectralTransform',
    'MPSFB',
    'MPFIRResBlock',
    'MPDownsample',
    'MPUpsample',
    # EDM2 magnitude-preserving network
    'MPSwinFIRUNet',
    'create_mp_swinfir_unet',
]
