"""
Frozen MedSAM ViT-B image encoder, used as a foundation feature extractor for
MRI conditioning. Produces a dense token grid plus a pooled vector that are
later injected into SwinFIRUNet via two zero-initialized paths (see
`models/network.py::SwinFIRUNet(use_foundation_cond=True)`).

Reference:
    MedSAM (Ma, He, et al., 2024). Apache-2.0.
    https://github.com/bowang-lab/MedSAM
    Checkpoint: `medsam_vit_b.pth` (ViT-B image encoder, ~90M params).

The encoder is loaded with `requires_grad=False` and `.eval()`. It is intended
to be invoked once per dataset sample by the precompute script
(`tools/precompute_medsam_features.py`). At training time, features are read
from a disk cache by `loader_medsam.py` so MedSAM never appears in the forward
graph.

Output tokens are returned as float32 in NHWC layout
`[B, 64, 64, 256]`, and the pooled vector as `[B, 256]`. The cache layer is
free to downcast to float16 for storage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# MedSAM/SAM ViT-B input resolution. The image encoder is patch-embedding-
# fixed at 1024×1024 with 16-px patches (64×64 token grid, 256-dim).
MEDSAM_INPUT_SIZE = 1024
MEDSAM_TOKEN_GRID = 64
MEDSAM_TOKEN_DIM = 256


def _import_sam_image_encoder():
    """
    Import the SAM ViT-B image encoder builder lazily so that this module can
    be imported even when `segment_anything` is not installed (e.g. on
    inference machines that only need the cache-consuming code paths).
    """
    try:
        from segment_anything.modeling.image_encoder import ImageEncoderViT
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "MedSAMEncoder requires `segment_anything`. Install with:\n"
            "    pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            f"Original error: {e}"
        ) from e
    return ImageEncoderViT


def _build_sam_vit_b_image_encoder() -> nn.Module:
    """Construct the SAM ViT-B image encoder with the canonical config."""
    ImageEncoderViT = _import_sam_image_encoder()
    # Config copied verbatim from segment_anything/build_sam.py::build_sam_vit_b
    # so the architecture matches MedSAM's released checkpoint.
    return ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=MEDSAM_INPUT_SIZE,
        mlp_ratio=4,
        norm_layer=lambda c: nn.LayerNorm(c, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=MEDSAM_TOKEN_DIM,
    )


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class MedSAMEncoder(nn.Module):
    """
    Frozen MedSAM ViT-B image encoder for MRI feature extraction.

    Args:
        checkpoint_path: Path to `medsam_vit_b.pth` from the MedSAM repo.
        device: Device for the encoder (default: CPU; the precompute script
            moves to GPU as needed).

    Methods:
        encode(mri_2d):
            Args:
                mri_2d: tensor of shape [B, 1, H, W] in [-1, 1] (the loader's
                    output range). Any H, W is accepted; the encoder resizes
                    internally to 1024×1024.
            Returns:
                tokens: [B, 64, 64, 256]  (float32, NHWC)
                pooled: [B, 256]          (mean over the 64×64 grid)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"MedSAM checkpoint not found at {checkpoint_path}. Download "
                "`medsam_vit_b.pth` from https://github.com/bowang-lab/MedSAM "
                "(Apache-2.0)."
            )

        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = _file_sha256(checkpoint_path)

        encoder = _build_sam_vit_b_image_encoder()

        # MedSAM checkpoints distributed by the upstream repo contain the full
        # SAM state_dict (image_encoder + mask_decoder + prompt_encoder). We
        # only need the `image_encoder.*` keys.
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        ie_state = {
            k[len("image_encoder."):]: v
            for k, v in state.items()
            if k.startswith("image_encoder.")
        }
        if not ie_state:
            # Already a bare image-encoder state_dict.
            ie_state = state
        missing, unexpected = encoder.load_state_dict(ie_state, strict=False)
        # SAM image encoder has no extras we care about; warn loudly only if
        # core ViT parameters are missing.
        critical_missing = [k for k in missing if "patch_embed" in k or "blocks" in k or "neck" in k]
        if critical_missing:
            raise RuntimeError(
                f"MedSAM checkpoint is missing critical image_encoder weights: "
                f"{critical_missing[:5]}..."
            )
        if unexpected:
            # Non-fatal: extra keys (mask_decoder etc.) were already filtered.
            pass

        self.encoder = encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # ImageNet-style normalization used by MedSAM (matches SAM).
        # Pixel values are expected in [0, 255] before this normalization in
        # the upstream MedSAM preprocessing, but the SAM ImageEncoderViT does
        # not include normalization itself — it's done externally. We apply
        # the SAM defaults.
        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
            persistent=False,
        )

        if device is not None:
            self.to(device)

    def train(self, mode: bool = True):  # noqa: D401 — keep frozen
        """Override: MedSAM stays in eval mode regardless of caller intent."""
        return super().train(False)

    @staticmethod
    def _to_three_channel(mri_2d: torch.Tensor) -> torch.Tensor:
        """Map single-channel MRI in [-1, 1] to a 3-channel pseudo-RGB in
        [0, 255], which is what SAM/MedSAM preprocessing expects."""
        # [-1, 1] -> [0, 1]
        x = (mri_2d.clamp(-1.0, 1.0) + 1.0) * 0.5
        # 1ch -> 3ch by replication.
        x = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
        # [0, 1] -> [0, 255]
        x = x * 255.0
        return x

    def _preprocess(self, mri_2d: torch.Tensor) -> torch.Tensor:
        x = self._to_three_channel(mri_2d)
        if x.shape[-2:] != (MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE):
            x = F.interpolate(
                x,
                size=(MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    @torch.inference_mode()
    def encode(self, mri_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of 2D MRI slices.

        Args:
            mri_2d: [B, 1, H, W] in [-1, 1].

        Returns:
            tokens: [B, 64, 64, 256] float32 (NHWC).
            pooled: [B, 256] float32 (mean over the 64×64 grid).
        """
        if mri_2d.dim() != 4 or mri_2d.shape[1] != 1:
            raise ValueError(
                f"Expected mri_2d of shape [B, 1, H, W]; got {tuple(mri_2d.shape)}"
            )
        x = self._preprocess(mri_2d.to(self.pixel_mean.device, dtype=torch.float32))
        # SAM ImageEncoderViT returns [B, 256, 64, 64] (NCHW after the neck).
        feat = self.encoder(x)
        if feat.shape[1] != MEDSAM_TOKEN_DIM or feat.shape[-2:] != (
            MEDSAM_TOKEN_GRID,
            MEDSAM_TOKEN_GRID,
        ):
            raise RuntimeError(
                "Unexpected encoder output shape "
                f"{tuple(feat.shape)}; expected [B, {MEDSAM_TOKEN_DIM}, "
                f"{MEDSAM_TOKEN_GRID}, {MEDSAM_TOKEN_GRID}]."
            )
        # NCHW -> NHWC for downstream cross-attention friendliness.
        tokens = feat.permute(0, 2, 3, 1).contiguous().float()
        pooled = tokens.mean(dim=(1, 2))  # [B, 256]
        return tokens, pooled

    def forward(self, mri_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode(mri_2d)
