"""Foreground reductions, acquisition-group splitting, and sampling utilities."""

import csv
import hashlib
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def patch_means(values, mask=None):
    """Return one mean per patch and its validity; empty masks have zero weight.

    The zero for an empty patch remains attached to the computation graph, so
    DDP ranks with no foreground can still participate in backward safely.
    """
    if mask is None:
        return values.flatten(1).mean(1), torch.ones(
            values.shape[0], dtype=torch.bool, device=values.device
        )
    foreground = torch.broadcast_to(mask.to(values.device) > 0.5, values.shape)
    counts = foreground.flatten(1).sum(1)
    sums = torch.where(foreground, values, 0.0).flatten(1).sum(1)
    return sums / counts.clamp_min(1), counts > 0


def valid_patch_mean(values, valid):
    return torch.where(valid, values, 0.0).sum() / valid.sum().clamp_min(1)


def foreground_metrics(pred, target, mask=None):
    """Pooled foreground PSNR/MAE and SSIM with entirely foreground windows.

    Inputs are [B,C,H,W] in [0,1], with data range fixed to 1. SSIM uses the
    standard 11x11 Gaussian window (sigma=1.5), without image padding. Every
    voxel in a contributing window must be foreground. Undefined scores are
    NaN, never a perfect-background score; callers should omit them from logs
    and checkpoint selection. Counts expose how much anatomy was evaluated.
    """
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("Metrics require matching [B,C,H,W] images")
    pred, target = pred.float(), target.float()
    foreground = (torch.ones_like(pred, dtype=torch.bool) if mask is None else
                  torch.broadcast_to(mask.to(pred.device) > 0.5, pred.shape))
    count = int(foreground.sum())
    result = {"psnr": float("nan"), "mae": float("nan"), "ssim": float("nan"),
              "foreground_pixels": count, "ssim_windows": 0}
    if not count:
        return result
    difference = (pred - target)[foreground]
    mse = difference.square().mean()
    result["psnr"] = float(-10 * torch.log10(mse))
    result["mae"] = float(difference.abs().mean())
    kernel_size = 11
    if min(pred.shape[-2:]) < kernel_size:
        return result
    channels = pred.shape[1]
    # Zero outside before convolution: even nonfinite background must not enter
    # otherwise valid windows. The support test below excludes border windows.
    x = torch.where(foreground, pred, 0.0)
    y = torch.where(foreground, target, 0.0)
    axis = torch.arange(kernel_size, device=pred.device, dtype=pred.dtype) - 5
    gaussian = torch.exp(-axis.square() / (2 * 1.5 ** 2))
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(channels, 1, -1, -1)
    moments = F.conv2d(torch.cat([x, y, x.square(), y.square(), x * y]),
                       kernel, groups=channels).chunk(5)
    mx, my, xx, yy, xy = moments
    vx, vy = (xx - mx.square()).clamp_min(0), (yy - my.square()).clamp_min(0)
    covariance = xy - mx * my
    ssim = ((2 * mx * my + 0.01 ** 2) * (2 * covariance + 0.03 ** 2) /
            ((mx.square() + my.square() + 0.01 ** 2) * (vx + vy + 0.03 ** 2)))
    valid_windows = F.avg_pool2d(foreground.float(), kernel_size, stride=1) == 1
    result["ssim_windows"] = int(valid_windows.sum())
    if result["ssim_windows"]:
        result["ssim"] = float(ssim[valid_windows].mean())
    return result


def find_group_manifest(data_root):
    """Follow normalization provenance back to its acquisition-group CSV."""
    root = Path(data_root).resolve()
    visited = set()
    while root not in visited:
        visited.add(root)
        for candidate in [root / "dataset_split_groups.csv",
                          root.parent / "reports" / "dataset_split_groups.csv"]:
            if candidate.is_file():
                return candidate
        settings_path = root / "preprocessing_settings.json"
        if not settings_path.is_file():
            return None
        source = json.loads(settings_path.read_text()).get("input_root")
        if not source:
            raise ValueError(f"Missing input_root in {settings_path}")
        source = Path(source)
        root = (source if source.is_absolute() else root / source).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Cannot follow split-group provenance to {root}")
    raise ValueError(f"Cyclic preprocessing provenance for {data_root}")


def split_by_acquisition_group(files, train_fraction, seed, group_manifests=None):
    """Split complete groups, namespaced by manifest (or dataset when unknown).

    Explicit manifests map data-root paths to CSV paths with patient,split_group
    columns. Otherwise discover them from preprocessing provenance. If a CSV
    exists it must cover every included patient: omissions are errors, not an
    excuse to silently split a repeated acquisition. Datasets without metadata
    fall back to folder identity with an explicit warning.
    """
    if not 0 < train_fraction <= 1:
        raise ValueError("train_split must be in (0, 1]")
    explicit = {str(Path(k).resolve()): Path(v).resolve()
                for k, v in (group_manifests or {}).items()}
    roots = sorted({str(Path(x["mr"]).resolve().parent.parent) for x in files})
    unknown = set(explicit) - set(roots)
    if unknown:
        raise ValueError(f"split_group_manifests contains unused data roots: {sorted(unknown)}")
    memberships, provenance = {}, {}
    for root in roots:
        manifest = explicit[root] if root in explicit else find_group_manifest(root)
        mapping = {}
        if manifest is not None:
            with manifest.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if not {"patient", "split_group"} <= set(reader.fieldnames or []):
                    raise ValueError(f"Expected patient,split_group columns in {manifest}")
                for row in reader:
                    patient, group = row["patient"].strip(), row["split_group"].strip()
                    if not patient or not group or (patient in mapping and mapping[patient] != group):
                        raise ValueError(f"Empty or conflicting group assignment in {manifest}")
                    mapping[patient] = group
            provenance[root] = {"path": str(manifest),
                                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        else:
            warnings.warn(f"No acquisition-group manifest for {root}; using folder identity only")
            provenance[root] = {"path": None, "identity": "patient_folder"}
        memberships[root] = (str(manifest) if manifest else root, mapping, manifest is not None)
    groups = defaultdict(list)
    seen = set()
    for item in files:
        root = str(Path(item["mr"]).resolve().parent.parent)
        patient = item["id"]
        namespace, mapping, required = memberships[root]
        if (root, patient) in seen:
            raise ValueError(f"Duplicate patient folder in data_dir: {root}/{patient}")
        seen.add((root, patient))
        if required and patient not in mapping:
            raise ValueError(f"Patient {patient} missing from split-group manifest for {root}")
        group = json.dumps([namespace, mapping.get(patient, patient)])
        groups[group].append(dict(item, split_group=group))
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2 or train_fraction == 1:
        cut = len(keys)
    else:
        cumulative = 0
        choices = []
        for i, key in enumerate(keys[:-1], 1):
            cumulative += len(groups[key])
            choices.append((abs(cumulative - len(files) * train_fraction), i))
        cut = min(choices)[1]
    train = [x for k in keys[:cut] for x in groups[k]]
    validation = [x for k in keys[cut:] for x in groups[k]]
    return train, validation, {"group_manifests": provenance, "seed": seed,
                                "train_groups": cut, "validation_groups": len(keys) - cut}


def collect_generation_batch(loader, count):
    """Honor num_val_samples even when it exceeds one per-rank loader batch."""
    if count < 1:
        raise ValueError("num_val_samples must be positive")
    batches, remaining = [], count
    for batch in loader:
        take = min(remaining, batch["mr"].shape[0])
        batches.append({k: v[:take] for k, v in batch.items() if torch.is_tensor(v)})
        remaining -= take
        if remaining == 0:
            break
    if not batches:
        raise ValueError("Cannot generate samples from an empty loader")
    return {k: torch.cat([b[k] for b in batches]) for k in batches[0]}


def resolve_inference_checkpoint(checkpoint, weights="auto"):
    """Prefer saved EMA for new checkpoints; keep legacy raw files usable."""
    path = Path(checkpoint)
    if weights not in {"auto", "ema", "raw"}:
        raise ValueError(f"Unknown weights selection: {weights}")
    if path.is_dir() or path.name in {"model.safetensors", "model_ema.safetensors"}:
        directory = path if path.is_dir() else path.parent
        ema_path = directory / "model_ema.safetensors"
        if weights == "ema" or (weights == "auto" and
                                (ema_path.is_file() or path.name == "model_ema.safetensors")):
            selected, kind = ema_path, "ema"
        else:
            selected, kind = directory / "model.safetensors", "raw"
    else:
        if weights == "ema":
            raise ValueError("Explicit EMA selection requires model_ema.safetensors or its directory")
        selected, kind = path, "raw"
    if not selected.is_file():
        raise FileNotFoundError(f"Missing {kind} checkpoint: {selected}")
    return selected, kind
