"""
Inference for MedSAM-conditioned PFGM-FIR models.

This is a NEW inference script. It does NOT modify `inference.py` or
`inference_volume.py`.

Two feature sources are supported:
    --features cache   (default)
        Read precomputed `medsam_features.pt` from each patient directory
        (produced by `python -m Swin_pfgm.tools.precompute_medsam_features`).
    --features encode
        Encode MRI on the fly using `models.foundation_encoder.MedSAMEncoder`.
        Requires `--medsam_ckpt`. Useful for held-out patients with no cache.

Usage (cache mode):
    python inference_medsam.py \
        --checkpoint outputs_accelerate/<run>_medsam_both/checkpoints/best/model.safetensors \
        --data_dir /path/to/test_data \
        --output_dir inference_output/medsam_both \
        --cond_path both \
        --num_steps 25
"""

from __future__ import annotations

import argparse
import contextvars
import os
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Reuse the EDM sampler from the existing inference module unchanged.
from inference import (
    edm_sample,
    DEFAULT_MODEL_CONFIG,
)
from loader import _get_load_transforms
from models.network import SwinFIRUNet


# Conditioning context — same approach as the trainer.
_cond_tokens_ctx: contextvars.ContextVar[Optional[torch.Tensor]] = contextvars.ContextVar(
    "cond_tokens", default=None
)
_cond_global_ctx: contextvars.ContextVar[Optional[torch.Tensor]] = contextvars.ContextVar(
    "cond_global", default=None
)


def _model_cond_pre_hook(module, args, kwargs):
    t = _cond_tokens_ctx.get()
    g = _cond_global_ctx.get()
    if t is not None and "cond_tokens" not in kwargs:
        kwargs["cond_tokens"] = t
    if g is not None and "cond_global" not in kwargs:
        kwargs["cond_global"] = g
    return args, kwargs


def build_model(cfg: Dict) -> SwinFIRUNet:
    in_channels = 3 if cfg.get("use_body_mask", False) else 2
    return SwinFIRUNet(
        in_channels=in_channels,
        out_channels=1,
        base_channels=cfg["base_channels"],
        channel_mults=tuple(cfg["channel_mults"]),
        num_res_blocks=cfg["num_res_blocks"],
        radius_emb_dim=cfg["radius_emb_dim"],
        attention_resolutions=tuple(cfg["attention_resolutions"]),
        num_heads=cfg["num_heads"],
        window_size=cfg["window_size"],
        mlp_ratio=cfg["mlp_ratio"],
        num_swin_blocks=cfg.get("num_swin_blocks", 2),
        num_sfb_blocks=cfg.get("num_sfb_blocks", 1),
        num_bottleneck_blocks=cfg.get("num_bottleneck_blocks", 2),
        use_sfb=cfg.get("use_sfb", False),
        dropout=cfg["dropout"],
        drop_path_rate=cfg["drop_path"],
        gradient_checkpoint=False,
        attention_type=cfg.get("attention_type", "deformable"),
        deformable_kwargs=cfg.get("deformable_kwargs", {"n_groups": 4, "kv_stride": 1}),
        use_foundation_cond=True,
    )


def load_checkpoint(model: nn.Module, ckpt_path: str) -> None:
    p = Path(ckpt_path)
    if p.is_dir():
        cand = p / "model.safetensors"
        if cand.exists():
            p = cand
        else:
            p = p / "pytorch_model.bin"
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(p))
    else:
        state = torch.load(str(p), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] loaded {p}  missing={len(missing)}  unexpected={len(unexpected)}")


def collect_patients(data_dir: Path, mr_pattern: str, ct_pattern: str) -> List[Dict[str, str]]:
    patients = []
    for p in sorted(os.listdir(data_dir)):
        pdir = data_dir / p
        if not pdir.is_dir():
            continue
        mr_file = ct_file = None
        for fname in os.listdir(pdir):
            if mr_pattern in fname and mr_file is None:
                mr_file = pdir / fname
            if ct_pattern in fname and ct_file is None:
                ct_file = pdir / fname
        if mr_file:
            patients.append({"id": p, "mr": str(mr_file),
                              "ct": str(ct_file) if ct_file else None,
                              "dir": str(pdir)})
    return patients


def load_features_cached(patient_dir: Path) -> Optional[Dict[str, torch.Tensor]]:
    cache = patient_dir / "medsam_features.pt"
    if not cache.is_file():
        return None
    return torch.load(str(cache), map_location="cpu", weights_only=False)


def encode_features_live(encoder, mr_volume: torch.Tensor, batch_size: int,
                          device: torch.device) -> Dict[str, torch.Tensor]:
    """Encode an MR volume [C, H, W, D] -> {tokens, pooled} float16."""
    from models.foundation_encoder import MEDSAM_TOKEN_DIM, MEDSAM_TOKEN_GRID
    C, H, W, D = mr_volume.shape
    if C != 1:
        mr_volume = mr_volume[:1]
    tokens_out = torch.empty(
        (D, MEDSAM_TOKEN_GRID, MEDSAM_TOKEN_GRID, MEDSAM_TOKEN_DIM),
        dtype=torch.float16,
    )
    pooled_out = torch.empty((D, MEDSAM_TOKEN_DIM), dtype=torch.float16)
    for s in range(0, D, batch_size):
        e = min(s + batch_size, D)
        batch = torch.stack([mr_volume[:, :, :, i] for i in range(s, e)], dim=0)
        batch = batch.to(device, dtype=torch.float32, non_blocking=True)
        tk, pl = encoder.encode(batch)
        tokens_out[s:e] = tk.detach().to("cpu", dtype=torch.float16)
        pooled_out[s:e] = pl.detach().to("cpu", dtype=torch.float16)
    return {"tokens": tokens_out, "pooled": pooled_out}


@torch.no_grad()
def infer_volume(
    model: nn.Module,
    mr_volume: torch.Tensor,        # [C, H, W, D] in [-1, 1]
    feats: Optional[Dict[str, torch.Tensor]],
    cond_path: str,
    target_size: int,
    batch_size: int,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    sigma_data: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    """Run per-slice EDM sampling, returning [1, H, W, D] CT in [-1, 1]."""
    C, H, W, D = mr_volume.shape

    # Resize MR slices to model's target size if needed.
    out = torch.empty((1, target_size, target_size, D), dtype=torch.float32)

    send_t = cond_path in ("spatial", "both")
    send_g = cond_path in ("global", "both")

    for start in tqdm(range(0, D, batch_size), desc="slices"):
        end = min(start + batch_size, D)
        # Build MR batch [B, 1, target, target]
        mr_batch = torch.stack(
            [mr_volume[:, :, :, s] for s in range(start, end)], dim=0
        ).float()
        if mr_batch.shape[-2:] != (target_size, target_size):
            mr_batch = F.interpolate(
                mr_batch, size=(target_size, target_size),
                mode="bilinear", align_corners=False,
            )
        mr_batch = mr_batch.clamp(-1.0, 1.0).to(device, non_blocking=True)

        # Per-slice conditioning
        ct_tok = ct_glo = None
        if feats is not None:
            tslices = feats["tokens"][start:end].float().to(device, non_blocking=True)
            pslices = feats["pooled"][start:end].float().to(device, non_blocking=True)
            if send_t:
                ct_tok = tslices
            if send_g:
                ct_glo = pslices

        tok_token = _cond_tokens_ctx.set(ct_tok)
        glo_token = _cond_global_ctx.set(ct_glo)
        try:
            ct_batch = edm_sample(
                model, mr_batch,
                num_steps=num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sigma_data=sigma_data,
                rho=rho,
                cfg_scale=1.0,
            )
        finally:
            _cond_tokens_ctx.reset(tok_token)
            _cond_global_ctx.reset(glo_token)

        ct_batch = ct_batch.clamp(-1.0, 1.0).cpu().float()
        for i, s in enumerate(range(start, end)):
            out[:, :, :, s] = ct_batch[i]
    return out


def save_volume(ct_volume: torch.Tensor, reference_nii_path: str, output_path: str):
    ref = nib.load(reference_nii_path)
    arr = ct_volume.squeeze(0).numpy().astype(np.float32)  # [H, W, D]
    # Resize back to reference shape (H, W) if different.
    H_ref, W_ref = ref.shape[:2]
    if arr.shape[:2] != (H_ref, W_ref):
        # Resample axially-only nearest is fine for inference output;
        # use bilinear via torch.
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(1)  # [D,1,H,W]
        t = F.interpolate(t, size=(H_ref, W_ref), mode="bilinear", align_corners=False)
        arr = t.squeeze(1).permute(1, 2, 0).numpy()
    nib.save(nib.Nifti1Image(arr, ref.affine, ref.header), output_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--cond_path", choices=["none", "global", "spatial", "both"], default="both")
    p.add_argument("--features", choices=["cache", "encode"], default="cache")
    p.add_argument("--medsam_ckpt", type=Path, default=None,
                   help="Required when --features encode.")
    p.add_argument("--mr_pattern", default="mr.nii")
    p.add_argument("--ct_pattern", default="ct.nii")
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--sigma_min", type=float, default=0.01)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--sigma_data", type=float, default=0.45)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--target_size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build & load model
    cfg = dict(DEFAULT_MODEL_CONFIG)
    cfg.setdefault("attention_type", "deformable")
    cfg.setdefault("deformable_kwargs", {"n_groups": 4, "kv_stride": 1})
    cfg.setdefault("num_swin_blocks", 6)
    cfg.setdefault("num_sfb_blocks", 1)
    cfg.setdefault("num_bottleneck_blocks", 2)
    cfg.setdefault("use_body_mask", True)
    model = build_model(cfg).to(device).eval()
    load_checkpoint(model, args.checkpoint)
    model.register_forward_pre_hook(_model_cond_pre_hook, with_kwargs=True)

    # Optional live encoder
    encoder = None
    if args.features == "encode":
        if args.medsam_ckpt is None:
            raise SystemExit("--medsam_ckpt is required when --features encode.")
        from models.foundation_encoder import MedSAMEncoder
        encoder = MedSAMEncoder(checkpoint_path=args.medsam_ckpt, device=device)

    load_tfm = _get_load_transforms(use_body_mask=False)

    patients = collect_patients(args.data_dir, args.mr_pattern, args.ct_pattern)
    if args.limit is not None:
        patients = patients[: args.limit]
    print(f"[infer] {len(patients)} patients")

    for patient in patients:
        print(f"\n=== {patient['id']} ===")
        loaded = load_tfm({"mr": patient["mr"], "ct": patient["ct"] or patient["mr"]})
        mr_vol = loaded["mr"]
        if hasattr(mr_vol, "data"):
            mr_vol = mr_vol.data
        mr_vol = mr_vol.clone().float()

        # Features
        feats = None
        if args.cond_path != "none":
            if args.features == "cache":
                feats = load_features_cached(Path(patient["dir"]))
                if feats is None:
                    print(f"[warn] no cache for {patient['id']}; running unconditioned")
            else:
                feats = encode_features_live(
                    encoder, mr_vol, batch_size=args.batch_size, device=device,
                )

        ct_vol = infer_volume(
            model, mr_vol, feats,
            cond_path=args.cond_path,
            target_size=args.target_size,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            sigma_data=args.sigma_data,
            rho=args.rho,
            device=device,
        )

        out_path = args.output_dir / f"{patient['id']}_pred_ct.nii.gz"
        save_volume(ct_vol, patient["mr"], str(out_path))
        print(f"[wrote] {out_path}")

    print("\n[done]")


if __name__ == "__main__":
    main()
