"""
Precompute frozen-MedSAM features for every MRI slice in a dataset, writing one
cache file per patient.

Usage
-----
    python -m Swin_pfgm.tools.precompute_medsam_features \
        --data_dir /path/to/data \
        --medsam_ckpt /path/to/medsam_vit_b.pth \
        --mr_pattern mr.nii \
        --device cuda

Reads the existing dataset using the SAME load+orient transforms as the
training loader (`loader.MonaiNiftiSliceDataset` / `_get_load_transforms`) so
axial slice indexing matches at training time.

Output (per patient, written next to the MR/CT files):
    <patient_dir>/medsam_features.pt        # float16 dict
        {
            "tokens": [D, 64, 64, 256] float16,    # NHWC
            "pooled": [D, 256]          float16,
        }
    <patient_dir>/medsam_features.json     # sidecar metadata
        {
            "backend": "medsam_vit_b",
            "encoder_sha256": "...",
            "n_slices": D,
            "input_size": 1024,
            "token_grid": 64,
            "token_dim": 256,
            "mr_path": "...",
            "mr_sha256": "...",
            "axial_axis": 3,                 # last spatial dim of [C,H,W,D]
            "version": 1,
        }

The script is idempotent: if a sidecar JSON exists and its `mr_sha256` matches
the current MR file, the patient is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Make sibling-package imports work when invoked as a script.
_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent  # .../Swin_pfgm
_REPO_ROOT = _PKG_ROOT.parent  # .../Swin-pfgm
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Swin_pfgm.loader import _get_load_transforms  # noqa: E402
from Swin_pfgm.models.foundation_encoder import (  # noqa: E402
    MEDSAM_TOKEN_DIM,
    MEDSAM_TOKEN_GRID,
    MedSAMEncoder,
)


CACHE_VERSION = 1
CACHE_NAME = "medsam_features.pt"
SIDECAR_NAME = "medsam_features.json"


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect_patients(
    data_dir: Path,
    mr_pattern: str,
    ct_pattern: str,
) -> List[Dict[str, str]]:
    """Subset of `train_accelerate.collect_data_files` that does not require
    the body mask. Returns dicts with `id`, `mr`, `ct`, `dir` fields."""
    patients = []
    for patient in sorted(os.listdir(data_dir)):
        pdir = data_dir / patient
        if not pdir.is_dir():
            continue
        mr_file = ct_file = None
        for fname in os.listdir(pdir):
            if mr_pattern in fname and mr_file is None:
                mr_file = pdir / fname
            if ct_pattern in fname and ct_file is None:
                ct_file = pdir / fname
        if mr_file and ct_file:
            patients.append(
                {"id": patient, "mr": str(mr_file), "ct": str(ct_file), "dir": str(pdir)}
            )
    return patients


def _is_cached_and_current(patient_dir: Path, mr_path: Path) -> bool:
    sidecar = patient_dir / SIDECAR_NAME
    cache = patient_dir / CACHE_NAME
    if not (sidecar.is_file() and cache.is_file()):
        return False
    try:
        meta = json.loads(sidecar.read_text())
    except Exception:
        return False
    if meta.get("version") != CACHE_VERSION:
        return False
    expected_sha = meta.get("mr_sha256")
    if not expected_sha:
        return False
    return expected_sha == _file_sha256(mr_path)


def _encode_volume(
    encoder: MedSAMEncoder,
    mr_volume: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Encode every axial slice of an MR volume.

    Args:
        mr_volume: [C, H, W, D] float tensor (already in [-1, 1]).
        batch_size: Slices per MedSAM forward pass.

    Returns:
        Dict with `tokens` [D, 64, 64, 256] float16 and `pooled` [D, 256] float16.
    """
    if mr_volume.dim() != 4:
        raise ValueError(f"Expected mr_volume [C,H,W,D]; got {tuple(mr_volume.shape)}")
    C, H, W, D = mr_volume.shape
    if C != 1:
        # Take the first channel — the loader always produces single-channel MR.
        mr_volume = mr_volume[:1]
        C = 1

    tokens_out = torch.empty(
        (D, MEDSAM_TOKEN_GRID, MEDSAM_TOKEN_GRID, MEDSAM_TOKEN_DIM),
        dtype=torch.float16,
    )
    pooled_out = torch.empty((D, MEDSAM_TOKEN_DIM), dtype=torch.float16)

    for start in range(0, D, batch_size):
        end = min(start + batch_size, D)
        # Extract slices and stack along the batch axis.
        # mr_volume[:, :, :, s] has shape [1, H, W]; we want [B, 1, H, W].
        batch = torch.stack(
            [mr_volume[:, :, :, s] for s in range(start, end)], dim=0
        ).to(device=device, dtype=torch.float32, non_blocking=True)
        tokens, pooled = encoder.encode(batch)  # both float32 on device
        tokens_out[start:end] = tokens.detach().to("cpu", dtype=torch.float16)
        pooled_out[start:end] = pooled.detach().to("cpu", dtype=torch.float16)

    return {"tokens": tokens_out, "pooled": pooled_out}


def process_patient(
    patient: Dict[str, str],
    encoder: MedSAMEncoder,
    load_tfm,
    batch_size: int,
    device: torch.device,
    overwrite: bool,
) -> str:
    """Returns "skipped", "wrote", or "error: <reason>"."""
    pdir = Path(patient["dir"])
    mr_path = Path(patient["mr"])

    if not overwrite and _is_cached_and_current(pdir, mr_path):
        return "skipped"

    try:
        loaded = load_tfm({"mr": str(mr_path), "ct": patient["ct"]})
        mr_vol = loaded["mr"]
        if hasattr(mr_vol, "data"):
            mr_vol = mr_vol.data
        mr_vol = mr_vol.clone().float()
    except Exception as e:
        return f"error: load failed: {e}"

    feats = _encode_volume(encoder, mr_vol, batch_size=batch_size, device=device)

    # Atomic write: temp file then rename.
    cache_path = pdir / CACHE_NAME
    sidecar_path = pdir / SIDECAR_NAME
    tmp_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_sidecar = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")

    torch.save(feats, str(tmp_cache))
    sidecar = {
        "backend": "medsam_vit_b",
        "encoder_sha256": encoder.checkpoint_sha256,
        "n_slices": int(feats["tokens"].shape[0]),
        "input_size": 1024,
        "token_grid": MEDSAM_TOKEN_GRID,
        "token_dim": MEDSAM_TOKEN_DIM,
        "mr_path": str(mr_path),
        "mr_sha256": _file_sha256(mr_path),
        "axial_axis": 3,
        "version": CACHE_VERSION,
    }
    tmp_sidecar.write_text(json.dumps(sidecar, indent=2))
    os.replace(tmp_cache, cache_path)
    os.replace(tmp_sidecar, sidecar_path)
    return "wrote"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=Path, required=True,
                   help="Root containing one folder per patient.")
    p.add_argument("--medsam_ckpt", type=Path, required=True,
                   help="Path to medsam_vit_b.pth (Apache-2.0).")
    p.add_argument("--mr_pattern", type=str, default="mr.nii")
    p.add_argument("--ct_pattern", type=str, default="ct.nii")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Slices per MedSAM forward pass. Lower if OOM.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-encode patients even if a current cache exists.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N patients (for testing).")
    args = p.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(f"data_dir not found: {args.data_dir}")

    device = torch.device(args.device)
    print(f"[medsam-cache] device={device}  ckpt={args.medsam_ckpt}")
    encoder = MedSAMEncoder(checkpoint_path=args.medsam_ckpt, device=device)
    print(f"[medsam-cache] encoder sha256: {encoder.checkpoint_sha256[:12]}...")

    load_tfm = _get_load_transforms(use_body_mask=False)

    patients = collect_patients(args.data_dir, args.mr_pattern, args.ct_pattern)
    if args.limit is not None:
        patients = patients[: args.limit]
    print(f"[medsam-cache] {len(patients)} patients found")

    n_wrote = n_skipped = n_err = 0
    t0 = time.time()
    for i, patient in enumerate(patients, 1):
        t_p = time.time()
        result = process_patient(
            patient, encoder, load_tfm,
            batch_size=args.batch_size,
            device=device,
            overwrite=args.overwrite,
        )
        if result == "wrote":
            n_wrote += 1
        elif result == "skipped":
            n_skipped += 1
        else:
            n_err += 1
        elapsed = time.time() - t_p
        print(f"[{i:>4d}/{len(patients)}] {patient['id']:<32s} {result:<10s}  "
              f"({elapsed:.1f}s, total {time.time()-t0:.1f}s)")

    print(f"\n[medsam-cache] done: wrote={n_wrote} skipped={n_skipped} errors={n_err}")


if __name__ == "__main__":
    main()
