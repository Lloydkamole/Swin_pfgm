"""
Loader extension that returns cached MedSAM features alongside the MR/CT slice.

The disk cache is produced by `tools/precompute_medsam_features.py`. Each
patient directory must contain a `medsam_features.pt` file with
`{tokens: [D, 64, 64, 256] f16, pooled: [D, 256] f16}` and a sidecar
`medsam_features.json` matching the MR file checksum.

Spatial augmentation note
-------------------------
The MedSAM features are computed once on the original (pre-augment) MR. Spatial
augmentations (flips, affine) applied to the MR/CT slice would therefore desync
from the cached features. This dataset:

* Always applies the SAME horizontal/vertical flip to MR, CT, mask, and the
  token grid + pooled vector (flips of token tensors are well-defined: flip
  along the H/W axis of [D, 64, 64, 256]).
* DROPS the random affine augmentation when MedSAM features are loaded
  (rotation/scale of the 64×64 token grid is not equivariant under MedSAM's
  patch-embedding ViT).
* Keeps MR-only intensity augmentations (Gaussian noise, contrast) — these do
  not touch CT/mask/features.

Original `loader.MonaiNiftiSliceDataset` is left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from loader import MonaiNiftiSliceDataset, get_monai_nifti_transforms  # noqa: F401


CACHE_NAME = "medsam_features.pt"
SIDECAR_NAME = "medsam_features.json"


def _resolve_patient_dir(file_dict: Dict[str, str]) -> Path:
    """Patient dir = directory containing the MR file (matches the cache layout
    used by `tools/precompute_medsam_features.py`)."""
    return Path(file_dict["mr"]).parent


def _validate_cache(patient_dir: Path) -> bool:
    cache = patient_dir / CACHE_NAME
    sidecar = patient_dir / SIDECAR_NAME
    return cache.is_file() and sidecar.is_file()


class MedSAMSliceDataset(MonaiNiftiSliceDataset):
    """
    Drop-in replacement for `MonaiNiftiSliceDataset` that additionally returns
    cached MedSAM features.

    Sample dict adds two keys:
        cond_tokens: [64, 64, 256] float32   (NHWC, per-slice token grid)
        cond_global: [256]          float32   (per-slice pooled vector)

    Args:
        medsam_dir: Unused placeholder kept for API compatibility — caches are
            always read from the patient directory itself (next to the MR file),
            matching the precompute script's layout. Pass any truthy value (or
            None) to disable foundation features for a debug run.
        require_cache: If True (default), missing/invalid caches raise. If
            False, missing caches return zero tensors (useful only for ablation
            of the conditioning path itself).
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        transform,
        target_size: int = 256,
        use_body_mask: bool = False,
        cache_volumes: bool = True,
        mode: str = "train",
        medsam_dir: Optional[str] = "auto",
        require_cache: bool = True,
        token_grid: int = 64,
        token_dim: int = 256,
    ):
        super().__init__(
            data=data,
            transform=transform,
            target_size=target_size,
            use_body_mask=use_body_mask,
            cache_volumes=cache_volumes,
            mode=mode,
        )
        self.medsam_enabled = medsam_dir is not None
        self.require_cache = require_cache
        self.token_grid = token_grid
        self.token_dim = token_dim

        # Per-patient feature cache (lazy-loaded into RAM the first time a
        # slice from that patient is requested). Keyed by file_idx.
        self._feat_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        if self.medsam_enabled:
            missing = []
            for i, file_dict in enumerate(self.file_list):
                pdir = _resolve_patient_dir(file_dict)
                if not _validate_cache(pdir):
                    missing.append(file_dict.get("id", str(pdir)))
            if missing and require_cache:
                raise FileNotFoundError(
                    f"MedSAM cache missing for {len(missing)} patient(s); "
                    f"first few: {missing[:5]}. Run "
                    "`python -m Swin_pfgm.tools.precompute_medsam_features --data_dir ...` "
                    "to populate."
                )
            if missing:
                print(
                    f"[MedSAMSliceDataset] WARNING: {len(missing)} patients "
                    "have no cache; their features will be zero-filled."
                )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _load_features(self, file_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if not self.medsam_enabled:
            return None
        if file_idx in self._feat_cache:
            return self._feat_cache[file_idx]
        pdir = _resolve_patient_dir(self.file_list[file_idx])
        cache = pdir / CACHE_NAME
        if not cache.is_file():
            self._feat_cache[file_idx] = {}  # sentinel: no cache
            return self._feat_cache[file_idx]
        loaded = torch.load(str(cache), map_location="cpu", weights_only=False)
        # Keep float16 in RAM; cast to float32 per-slice in __getitem__.
        feats = {
            "tokens": loaded["tokens"],  # [D, 64, 64, 256] f16
            "pooled": loaded["pooled"],  # [D, 256] f16
        }
        self._feat_cache[file_idx] = feats
        return feats

    def _zero_feats(self) -> Dict[str, torch.Tensor]:
        return {
            "cond_tokens": torch.zeros(
                self.token_grid, self.token_grid, self.token_dim, dtype=torch.float32
            ),
            "cond_global": torch.zeros(self.token_dim, dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Augmentation (override to keep MR/CT in lockstep with token grid)
    # ------------------------------------------------------------------
    def _augment_with_features(
        self,
        sample: Dict[str, torch.Tensor],
        cond_tokens: Optional[torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Apply flips + intensity augs to the slice, mirroring flips on the
        token grid. Affine augmentations are skipped when token grid is present."""
        if self.mode != "train":
            return sample, cond_tokens

        keys = list(sample.keys())

        # --- Horizontal flip (image axis 0 = H) ---
        if torch.rand(1).item() < 0.5:
            for k in keys:
                sample[k] = sample[k].flip(1)
            if cond_tokens is not None:
                # tokens shape [64, 64, 256]; image axis 0 == token axis 0
                cond_tokens = cond_tokens.flip(0)

        # --- Vertical flip (image axis 1 = W) ---
        if torch.rand(1).item() < 0.5:
            for k in keys:
                sample[k] = sample[k].flip(2)
            if cond_tokens is not None:
                cond_tokens = cond_tokens.flip(1)

        # --- Affine: SKIP when token grid present (not equivariant) ---
        if cond_tokens is None and torch.rand(1).item() < 0.2:
            angle = (torch.rand(1).item() - 0.5) * 2 * np.deg2rad(5.0)
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.1
            cos_a, sin_a = np.cos(angle) * scale, np.sin(angle) * scale
            theta = torch.tensor(
                [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], dtype=torch.float32
            ).unsqueeze(0)
            for k in keys:
                img = sample[k].unsqueeze(0)
                grid = F.affine_grid(theta, img.shape, align_corners=False)
                mode = "nearest" if k == "mask" else "bilinear"
                pad = "zeros" if k == "mask" else "border"
                sample[k] = F.grid_sample(
                    img, grid, mode=mode, padding_mode=pad, align_corners=False
                ).squeeze(0)

        # --- MR-only intensity augs ---
        if "mr" in sample and torch.rand(1).item() < 0.15:
            sample["mr"] = sample["mr"] + torch.randn_like(sample["mr"]) * 0.02
        if "mr" in sample and torch.rand(1).item() < 0.15:
            gamma = 0.9 + torch.rand(1).item() * 0.2
            mr = (sample["mr"] + 1.0) / 2.0
            mr = mr.clamp(0, 1).pow(gamma)
            sample["mr"] = mr * 2.0 - 1.0

        return sample, cond_tokens

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_map[idx]
        if not self.cache_volumes:
            # Fall back to parent (no MedSAM augmentation guarantee in this path).
            output = super().__getitem__(idx)
            output.update(self._zero_feats() if self.medsam_enabled else {})
            return output

        cached = self._vol_cache[file_idx]
        mr_slice = cached["mr"][:, :, :, slice_idx].clone()
        ct_slice = cached["ct"][:, :, :, slice_idx].clone()

        mask_slice = None
        if self.use_body_mask and cached["mask"] is not None:
            mask_slice = cached["mask"][:, :, :, slice_idx].clone()
            mask_slice = (mask_slice > 0).float()

        # Load features for this slice
        feats = self._load_features(file_idx)
        cond_tokens = None
        cond_global = None
        if self.medsam_enabled and feats and "tokens" in feats:
            t = feats["tokens"]
            p = feats["pooled"]
            if slice_idx >= t.shape[0]:
                if self.require_cache:
                    raise IndexError(
                        f"slice_idx {slice_idx} out of range for cache "
                        f"(D={t.shape[0]}) in patient "
                        f"{self.file_list[file_idx].get('id', file_idx)}"
                    )
                cond_tokens = torch.zeros(
                    self.token_grid, self.token_grid, self.token_dim,
                    dtype=torch.float32,
                )
                cond_global = torch.zeros(self.token_dim, dtype=torch.float32)
            else:
                cond_tokens = t[slice_idx].float().clone()  # [64,64,256]
                cond_global = p[slice_idx].float().clone()  # [256]

        # Apply (feature-aware) augmentations
        sample = {"mr": mr_slice, "ct": ct_slice}
        if mask_slice is not None:
            sample["mask"] = mask_slice
        sample, cond_tokens = self._augment_with_features(sample, cond_tokens)

        mr_slice = sample["mr"]
        ct_slice = sample["ct"]
        mask_slice = sample.get("mask", mask_slice)

        # Resize to target (matches parent behavior)
        if mr_slice.shape[1] != self.target_size or mr_slice.shape[2] != self.target_size:
            mr_slice = F.interpolate(
                mr_slice.unsqueeze(0), size=(self.target_size, self.target_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            ct_slice = F.interpolate(
                ct_slice.unsqueeze(0), size=(self.target_size, self.target_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            if mask_slice is not None:
                mask_slice = F.interpolate(
                    mask_slice.unsqueeze(0), size=(self.target_size, self.target_size),
                    mode="nearest",
                ).squeeze(0)

        mr_slice = mr_slice.clamp(-1.0, 1.0).float()
        ct_slice = ct_slice.clamp(-1.0, 1.0).float()

        output: Dict[str, torch.Tensor] = {"mr": mr_slice, "ct": ct_slice}
        if self.use_body_mask:
            output["mask"] = (
                mask_slice.float()
                if mask_slice is not None
                else torch.ones(1, self.target_size, self.target_size, dtype=torch.float32)
            )
        if self.medsam_enabled:
            if cond_tokens is None or cond_global is None:
                output.update(self._zero_feats())
            else:
                output["cond_tokens"] = cond_tokens.float().contiguous()
                output["cond_global"] = cond_global.float().contiguous()
        return output
