"""Diagnostics for the 2.5D deformable-attention SwinFIR model.

The utilities in this module are intentionally independent of the training
entry point. This keeps notebook diagnostics deterministic and avoids starting
Accelerate, data-loader workers, or training-side logging by accident.
"""

from __future__ import annotations

import math
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

from loader_25d import MonaiNifti25DDataset, get_monai_nifti_transforms_25d
from models.deformable_attention_3d import (
    AdaLNZeroDeformableBlock3D,
    DeformableAttention3D,
)
from models.network_25d import create_swin_fir_unet_25d


RUN_MODEL_CONFIG = {
    "base_channels": 64,
    "channel_mults": (1, 2, 3, 4),
    "attention_resolutions": (3, 4),
    "num_res_blocks": 1,
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "radius_emb_dim": 256,
    "window_size": 4,
    "window_size_3d": {
        1: (2, 4, 4),
        2: (2, 4, 4),
        3: (1, 4, 4),
    },
    "num_swin_blocks": 2,
    "num_sfb_blocks": 1,
    "num_bottleneck_blocks": 1,
    "use_sfb": True,
    "dropout": 0.0,
    "drop_path_rate": 0.1,
    "gradient_checkpoint": False,
    "attention_type": "deformable",
    "deformable_kwargs": {
        "n_groups": 8,
        "kv_stride": 2,
        "offset_range_factor": 8.0,
        "use_qk_norm": True,
    },
    "use_deformable_sfb": False,
    "deformable_sfb_kwargs": {
        "kernel_size": (1, 3, 3),
        "offset_scale": 1.0,
        "use_modulation": False,
        "deformable_depth": False,
    },
    "downsample_strides": ((1, 2, 2), (2, 2, 2), (2, 2, 2)),
}


def build_model(
    checkpoint: Union[os.PathLike, str],
    device: Union[torch.device, str],
) -> Tuple[torch.nn.Module, dict]:
    """Build the exact logged architecture and strictly load the checkpoint."""
    model = create_swin_fir_unet_25d(
        # The run loads body masks, but its construction call explicitly uses
        # two image channels. Masks reach attention and loss through `mask`.
        in_channels=2,
        out_channels=1,
        num_slices=4,
        **RUN_MODEL_CONFIG,
    ).to(device)
    state = load_safetensors(str(checkpoint), device="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    model.eval()

    tensors = list(state.values())
    nonfinite = sum((~torch.isfinite(t)).sum().item() for t in tensors)
    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_tensors": len(state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "nonfinite_parameters": int(nonfinite),
        "deformable_attention_modules": sum(
            isinstance(m, DeformableAttention3D) for m in model.modules()
        ),
    }
    return model, report


def collect_data_files(
    data_dir: Union[os.PathLike, str],
    use_body_mask: bool = True,
) -> list[dict]:
    """Mirror the case-insensitive filename matching used by training."""
    data = []
    for patient_dir in sorted(os.listdir(data_dir)):
        patient_path = Path(data_dir) / patient_dir
        if not patient_path.is_dir():
            continue
        paths = {"mr": None, "ct": None, "mask": None}
        for path in patient_path.iterdir():
            name = path.name.lower()
            if "mr.nii" in name:
                paths["mr"] = str(path)
            if "ct.nii" in name:
                paths["ct"] = str(path)
            if use_body_mask and "mask" in name:
                paths["mask"] = str(path)
        if paths["mr"] and paths["ct"]:
            item = {"mr": paths["mr"], "ct": paths["ct"], "id": patient_dir}
            if use_body_mask:
                item["mask"] = paths["mask"]
            data.append(item)
    return data


def load_validation_patch(
    data_dir: Union[os.PathLike, str],
    patient_index: int = 0,
    train_split: float = 0.9,
    target_size: int = 256,
    num_slices: int = 4,
) -> Tuple[dict, dict]:
    """Load a centered four-slice patch from the held-out patient split."""
    all_files = collect_data_files(data_dir, use_body_mask=True)
    split_index = max(1, int(len(all_files) * train_split))
    validation = all_files[split_index:]
    if not validation:
        raise RuntimeError("The configured split contains no validation patients")
    item = validation[patient_index % len(validation)]
    if item.get("mask") is None:
        raise FileNotFoundError(f"No mask found for validation patient {item['id']}")

    dataset = MonaiNifti25DDataset(
        data=[item],
        transform=get_monai_nifti_transforms_25d("val", use_body_mask=True),
        target_size=target_size,
        num_slices=num_slices,
        stride=num_slices,
        use_body_mask=True,
        cache_volumes=False,
        mode="val",
    )
    if not dataset.volume_map:
        raise RuntimeError(f"No valid 2.5D patches found for patient {item['id']}")

    # The map is ordered by center slice. Its middle entry is the closest
    # available patch to the volume center for the fixed sampling stride.
    patch_index = len(dataset.volume_map) // 2
    sample = dataset[patch_index]
    metadata = {
        "patient_id": item["id"],
        "validation_patients": len(validation),
        "center_slice": dataset.volume_map[patch_index][1],
        "paths": item,
    }
    return sample, metadata


def spatial_center_crop(sample: dict, size: int) -> dict:
    """Crop each [C,D,H,W] sample tensor around the in-plane center."""
    output = {}
    for key, tensor in sample.items():
        h, w = tensor.shape[-2:]
        if size > h or size > w:
            raise ValueError(f"Requested crop {size} exceeds sample size {(h, w)}")
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        output[key] = tensor[..., y0:y0 + size, x0:x0 + size]
    return output


def to_device_batch(sample: dict, device: Union[torch.device, str]) -> dict:
    return {key: value.unsqueeze(0).to(device) for key, value in sample.items()}


def edm_scalings(
    sigma: torch.Tensor,
    sigma_data: float = 0.45,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_5d = sigma.view(-1, 1, 1, 1, 1)
    denominator = sigma_5d.square() + sigma_data ** 2
    c_skip = sigma_data ** 2 / denominator
    c_out = sigma_5d * sigma_data / denominator.sqrt()
    c_in = denominator.rsqrt()
    return c_skip, c_out, c_in


def edm_precondition(
    model: torch.nn.Module,
    noisy_ct: torch.Tensor,
    mri: torch.Tensor,
    sigma: torch.Tensor,
    mask: Optional[torch.Tensor],
    sigma_data: float = 0.45,
) -> torch.Tensor:
    c_skip, c_out, c_in = edm_scalings(sigma, sigma_data)
    c_noise = sigma.log() / 4.0
    raw = model(c_in * noisy_ct, mri, c_noise, mask=mask)
    return c_skip * noisy_ct + c_out * raw


def pfgmpp_perturb(
    clean: torch.Tensor,
    sigma: torch.Tensor,
    augmented_dim: int = 128,
    seed: int = 42,
) -> torch.Tensor:
    """Apply the same inverse-beta PFGM++ perturbation as training."""
    devices = [clean.device] if clean.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        batch = clean.shape[0]
        data_dim = clean[0].numel()
        radius_scale = sigma.double() * math.sqrt(augmented_dim)
        beta = torch.distributions.Beta(
            torch.tensor(data_dim / 2, device=clean.device, dtype=torch.float64),
            torch.tensor(augmented_dim / 2, device=clean.device, dtype=torch.float64),
        ).sample((batch,))
        inverse_beta = beta / (1 - beta + 1e-8)
        radius = radius_scale * torch.sqrt(inverse_beta + 1e-8)
        direction = torch.randn(
            batch, data_dim, device=clean.device, dtype=torch.float64
        )
        direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)
        noise = (direction * radius[:, None]).view_as(clean).float()
    return clean + noise


def masked_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> dict:
    valid = mask >= 0.5 if mask is not None else torch.ones_like(target, dtype=torch.bool)
    difference = (prediction - target)[valid]
    mse = difference.square().mean()
    data_range = 2.0
    return {
        "mse": mse.item(),
        "rmse": mse.sqrt().item(),
        "mae": difference.abs().mean().item(),
        "psnr": (20 * torch.log10(torch.tensor(data_range, device=mse.device))
                 - 10 * torch.log10(mse.clamp_min(1e-12))).item(),
    }


class AttentionCapture3D(AbstractContextManager):
    """Capture all offset fields and Q/K tensors for one selected module."""

    def __init__(self, model: torch.nn.Module, selected_name: str):
        self.model = model
        self.selected_name = selected_name
        self.offsets: Dict[str, torch.Tensor] = {}
        self.selected: dict = {}
        self.handles = []

    def __enter__(self):
        modules = dict(self.model.named_modules())
        if self.selected_name not in modules:
            available = [
                name for name, module in modules.items()
                if isinstance(module, DeformableAttention3D)
            ]
            raise KeyError(
                f"Unknown attention module {self.selected_name!r}. Available: {available}"
            )

        for name, module in modules.items():
            if isinstance(module, DeformableAttention3D):
                self.handles.append(
                    module.conv_offset.register_forward_hook(self._offset_hook(name))
                )

        selected = modules[self.selected_name]
        self.handles.append(
            selected.register_forward_pre_hook(self._input_hook, with_kwargs=True)
        )
        self.handles.append(selected.q_norm.register_forward_hook(self._q_hook))
        self.handles.append(selected.k_norm.register_forward_hook(self._k_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def _offset_hook(self, name):
        def hook(_module, _inputs, output):
            self.offsets[name] = output.detach().float().cpu()
        return hook

    def _input_hook(self, _module, args, kwargs):
        self.selected["shape"] = tuple(int(v) for v in args[1:4])
        mask = kwargs.get("spatial_mask")
        self.selected["spatial_mask"] = (
            mask.detach().float().cpu() if mask is not None else None
        )

    def _q_hook(self, _module, _inputs, output):
        self.selected["q"] = output.detach().float().cpu()

    def _k_hook(self, _module, _inputs, output):
        self.selected["k"] = output.detach().float().cpu()


def attention_module_names(model: torch.nn.Module) -> list[str]:
    return [
        name for name, module in model.named_modules()
        if isinstance(module, DeformableAttention3D)
    ]


def scaled_offsets(
    module: DeformableAttention3D,
    raw_offsets: torch.Tensor,
) -> dict:
    """Convert raw offsets to normalized-grid and K/V-voxel displacement."""
    batch_groups, _, depth, height, width = raw_offsets.shape
    batch = batch_groups // module.n_groups
    bounded = raw_offsets.tanh()
    voxel = bounded * module.offset_range_factor
    normalized = torch.empty_like(bounded)
    normalized[:, 0] = voxel[:, 0] * 2.0 / max(width - 1, 1)
    normalized[:, 1] = voxel[:, 1] * 2.0 / max(height - 1, 1)
    normalized[:, 2] = voxel[:, 2] * 2.0 / max(depth - 1, 1)
    return {
        "raw": raw_offsets.reshape(batch, module.n_groups, 3, depth, height, width),
        "voxel": voxel.reshape(batch, module.n_groups, 3, depth, height, width),
        "normalized": normalized.reshape(
            batch, module.n_groups, 3, depth, height, width
        ),
        "bounded": bounded.reshape(batch, module.n_groups, 3, depth, height, width),
        "kv_shape": (depth, height, width),
    }


def reference_grid(depth: int, height: int, width: int) -> torch.Tensor:
    z = torch.linspace(-1, 1, depth)
    y = torch.linspace(-1, 1, height)
    x = torch.linspace(-1, 1, width)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)


def deformed_grid(offset_info: dict) -> torch.Tensor:
    depth, height, width = offset_info["kv_shape"]
    offsets = offset_info["normalized"].permute(0, 1, 3, 4, 5, 2)
    return reference_grid(depth, height, width)[None, None] + offsets


def offset_statistics(
    model: torch.nn.Module,
    captured_offsets: Dict[str, torch.Tensor],
) -> list[dict]:
    modules = dict(model.named_modules())
    rows = []
    for name, raw in captured_offsets.items():
        module = modules[name]
        info = scaled_offsets(module, raw)
        voxel = info["voxel"]
        bounded = info["bounded"]
        grid = deformed_grid(info)
        magnitude = voxel.square().sum(dim=2).sqrt()
        rows.append({
            "module": name,
            "kv_shape": info["kv_shape"],
            "dx_abs_mean": voxel[:, :, 0].abs().mean().item(),
            "dy_abs_mean": voxel[:, :, 1].abs().mean().item(),
            "dz_abs_mean": voxel[:, :, 2].abs().mean().item(),
            "magnitude_mean": magnitude.mean().item(),
            "magnitude_max": magnitude.max().item(),
            "saturated_fraction": (bounded.abs() > 0.95).float().mean().item(),
            "preclamp_oob_fraction": (grid.abs() > 1).any(dim=-1).float().mean().item(),
        })
    return rows


def _off_diagonal_mean(matrix: torch.Tensor) -> float:
    if matrix.shape[0] < 2:
        return float("nan")
    keep = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    return matrix[keep].mean().item()


def group_offset_similarity(offset_info: dict) -> dict:
    """Cosine similarity of group-specific offset fields for batch item zero."""
    groups = offset_info["voxel"][0].flatten(1)
    groups = groups - groups.mean(dim=1, keepdim=True)
    groups = F.normalize(groups, dim=1, eps=1e-8)
    similarity = groups @ groups.T
    return {
        "matrix": similarity,
        "off_diagonal_mean": _off_diagonal_mean(similarity),
    }


def sampled_mask_from_capture(
    module: DeformableAttention3D,
    capture: AttentionCapture3D,
    offset_info: dict,
) -> Optional[torch.Tensor]:
    mask = capture.selected.get("spatial_mask")
    if mask is None:
        return None
    depth, height, width = capture.selected["shape"]
    mask = F.interpolate(mask, size=(depth, height, width), mode="nearest")
    batch = mask.shape[0]
    grouped = mask[:, None].expand(-1, module.n_groups, -1, -1, -1, -1)
    grouped = grouped.reshape(batch * module.n_groups, 1, depth, height, width)
    grid = deformed_grid(offset_info).clamp(-1, 1)
    grid = grid.reshape(batch * module.n_groups, *grid.shape[2:])
    sampled = F.grid_sample(
        grouped, grid, mode="nearest", padding_mode="zeros", align_corners=True
    )
    sampled = sampled.reshape(batch, module.n_groups, *offset_info["kv_shape"])
    return sampled.min(dim=1).values


def attention_statistics(
    model: torch.nn.Module,
    capture: AttentionCapture3D,
    max_queries: int = 256,
    seed: int = 42,
) -> dict:
    """Recompute sampled-query attention because SDPA does not return weights."""
    modules = dict(model.named_modules())
    module = modules[capture.selected_name]
    q = capture.selected["q"][0]
    k = capture.selected["k"][0]
    length_q = q.shape[1]
    generator = torch.Generator().manual_seed(seed)
    if length_q > max_queries:
        query_indices = torch.randperm(length_q, generator=generator)[:max_queries].sort().values
    else:
        query_indices = torch.arange(length_q)

    logits = torch.matmul(q[:, query_indices], k.transpose(-1, -2))
    logits = logits / math.sqrt(q.shape[-1])

    offset_info = scaled_offsets(module, capture.offsets[capture.selected_name])
    sampled_mask = sampled_mask_from_capture(module, capture, offset_info)
    if sampled_mask is not None:
        valid_keys = sampled_mask[0].flatten() >= 0.5
        if valid_keys.any():
            logits[..., ~valid_keys] = float("-inf")
    else:
        valid_keys = torch.ones(k.shape[1], dtype=torch.bool)

    attention = logits.softmax(dim=-1)
    entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy / math.log(max(int(valid_keys.sum()), 2))
    key_importance = attention.mean(dim=1)

    centered = key_importance - key_importance.mean(dim=1, keepdim=True)
    centered = F.normalize(centered, dim=1, eps=1e-8)
    head_similarity = centered @ centered.T
    return {
        "attention": attention,
        "query_indices": query_indices,
        "entropy": entropy,
        "entropy_mean": entropy.mean().item(),
        "entropy_min": entropy.min().item(),
        "key_importance": key_importance,
        "head_similarity": head_similarity,
        "head_similarity_off_diagonal": _off_diagonal_mean(head_similarity),
        "valid_keys": valid_keys,
        "sampled_mask": sampled_mask,
        "offset_info": offset_info,
        "query_shape": capture.selected["shape"],
        "kv_shape": offset_info["kv_shape"],
    }


@torch.no_grad()
def gate_statistics(
    model: torch.nn.Module,
    sigma_levels: Iterable[float],
    device: Union[torch.device, str],
) -> list[dict]:
    blocks = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, AdaLNZeroDeformableBlock3D)
    ]
    rows = []
    for sigma_value in sigma_levels:
        sigma = torch.tensor([sigma_value], device=device)
        radius_embedding = model.radius_emb(sigma.log() / 4.0)
        for name, block in blocks:
            modulation = block.adaLN_modulation(radius_embedding)
            chunks = modulation.chunk(6, dim=-1)
            rows.append({
                "sigma": sigma_value,
                "module": name,
                "gate_msa_abs_mean": chunks[2].abs().mean().item(),
                "gate_mlp_abs_mean": chunks[5].abs().mean().item(),
                "gate_msa_max": chunks[2].abs().max().item(),
                "gate_mlp_max": chunks[5].abs().max().item(),
            })
    return rows


@torch.no_grad()
def run_forward_diagnostics(
    model: torch.nn.Module,
    batch: dict,
    sigma_value: float = 1.0,
    seed: int = 42,
) -> dict:
    ct = batch["ct"]
    mri = batch["mr"]
    mask = batch.get("mask")
    sigma = torch.full((ct.shape[0],), sigma_value, device=ct.device)
    noisy = pfgmpp_perturb(ct, sigma, seed=seed)
    prediction = edm_precondition(model, noisy, mri, sigma, mask)
    prediction_zero_mri = edm_precondition(model, noisy, torch.zeros_like(mri), sigma, mask)
    prediction_open_mask = edm_precondition(
        model, noisy, mri, sigma, torch.ones_like(mask) if mask is not None else None
    )

    result = {
        "noisy": noisy,
        "prediction": prediction,
        "prediction_zero_mri": prediction_zero_mri,
        "prediction_open_mask": prediction_open_mask,
        "noisy_metrics": masked_metrics(noisy, ct, mask),
        "prediction_metrics": masked_metrics(prediction, ct, mask),
        "output_all_finite": bool(torch.isfinite(prediction).all()),
        "output_min": prediction.min().item(),
        "output_max": prediction.max().item(),
        "output_mean": prediction.mean().item(),
        "output_std": prediction.std().item(),
    }
    normal_scale = prediction.square().mean().sqrt().clamp_min(1e-8)
    result["zero_mri_relative_change"] = (
        (prediction - prediction_zero_mri).square().mean().sqrt() / normal_scale
    ).item()
    result["open_mask_relative_change"] = (
        (prediction - prediction_open_mask).square().mean().sqrt() / normal_scale
    ).item()
    return result


def training_scalars(event_file: Union[os.PathLike, str]) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    output = {}
    for tag in accumulator.Tags().get("scalars", []):
        values = accumulator.Scalars(tag)
        output[tag] = {
            "step": np.asarray([value.step for value in values]),
            "value": np.asarray([value.value for value in values]),
        }
    return output
