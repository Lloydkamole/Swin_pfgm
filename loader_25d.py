"""
2.5D Data Loader for PFGM-FIR.

Loads small volumetric patches (e.g., 5 consecutive slices) from 3D NIfTI volumes.
Uses fixed stride sampling without padding - some edge slices will be skipped.

Returns:
    - MRI volume: [B, 1, D, H, W] where D=num_slices (e.g., 5)
    - CT volume: [B, 1, D, H, W]
"""

import os
import torch
import random
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
)
from torch.utils.data import Dataset
import nibabel as nib


def get_monai_nifti_transforms_25d(mode="train", use_body_mask=False):
    """
    Define MONAI transforms for loading 3D NIfTI data.
    Includes data augmentation for training mode (matching loader.py).
    
    Augmentations (train only):
    - Random flips along spatial axes (H, W only — no depth flips to preserve slice ordering)
    - Random affine (small rotation + scale in H,W plane)
    - Random Gaussian noise on MRI only
    - Random contrast adjustment on MRI only

    Mask key uses allow_missing_keys=True so samples without a mask
    file simply skip mask-related transforms instead of erroring.
    """
    keys = ["mr", "ct"]
    keys_with_mask = ["mr", "ct", "mask"] if use_body_mask else keys
    resize_modes = ["bilinear", "bilinear"] + (["nearest"] if use_body_mask else [])
    
    base_transforms = [
        LoadImaged(keys=keys_with_mask, image_only=True, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys_with_mask, allow_missing_keys=True),
        Orientationd(keys=keys_with_mask, axcodes="RAS", allow_missing_keys=True),
    ]
    
    if mode == "train":
        transforms = base_transforms + [
            # Spatial flips: axis 0 = H (LR), axis 1 = W (AP) — skip axis 2 (depth/SI)
            RandFlipd(keys=keys_with_mask, prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=keys_with_mask, prob=0.5, spatial_axis=1, allow_missing_keys=True),
            # Small in-plane rotation + scale (no depth rotation)
            # RandAffined(
            #     keys=keys_with_mask,
            #     prob=0.2,
            #     rotate_range=(0.0, 0.0, np.deg2rad(5.0)),
            #     scale_range=(0.05, 0.05, 0.0),
            #     mode=resize_modes,
            #     padding_mode="border",
            #     allow_missing_keys=True,
            # ),
            # # Intensity augmentation on MRI only
            # RandGaussianNoised(keys=["mr"], prob=0.15, mean=0.0, std=0.02),
            # RandAdjustContrastd(keys=["mr"], prob=0.15, gamma=(0.9, 1.1)),
        ]
    else:
        transforms = base_transforms
    
    return Compose(transforms)


def _get_load_transforms_25d(use_body_mask=False):
    """Deterministic transforms: load + orient. Results can be cached."""
    keys_with_mask = ["mr", "ct", "mask"] if use_body_mask else ["mr", "ct"]
    return Compose([
        LoadImaged(keys=keys_with_mask, image_only=True, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys_with_mask, allow_missing_keys=True),
        Orientationd(keys=keys_with_mask, axcodes="RAS", allow_missing_keys=True),
    ])


def _get_aug_transforms_25d(mode="train", use_body_mask=False):
    """Fast PyTorch augmentations for cached 2.5D patches in [C, D, H, W]."""
    if mode != "train":
        return None

    def _augment(sample):
        keys = list(sample.keys())

        # Spatial flips only; keep depth ordering intact.
        if torch.rand(1).item() < 0.5:
            for k in keys:
                sample[k] = sample[k].flip(2)  # H
        if torch.rand(1).item() < 0.5:
            for k in keys:
                sample[k] = sample[k].flip(3)  # W

        return sample

    return _augment


class MonaiNifti25DDataset(Dataset):
    """
    A PyTorch Dataset for 2.5D volumetric slices.
    
    Samples small volumes (e.g., 5 consecutive slices) from 3D NIfTI files.
    Uses fixed stride sampling - edge slices that don't form complete volumes are skipped.
    
    Args:
        data (list of dict): List where each dict contains 'mr' and 'ct' file paths.
        transform (callable): MONAI Compose object for 3D volume transforms.
        target_size (int): Target spatial size for slices (H, W).
        num_slices (int): Number of consecutive slices to sample (depth dimension).
        stride (int): Stride between volume centers. If None, defaults to num_slices (non-overlapping).
    """
    
    def __init__(
        self,
        data,
        transform,
        target_size=256,
        num_slices=5,
        stride=None,
        use_body_mask=False,
        cache_volumes=True,
        mode="train",
    ):
        self.file_list = data
        self.transform = transform
        self.target_size = target_size
        self.num_slices = num_slices
        self.stride = stride if stride is not None else num_slices
        self.use_body_mask = use_body_mask
        self.cache_volumes = cache_volumes
        self.mode = mode
        self.volume_map = []
        self._vol_cache = {}

        if cache_volumes:
            self._load_tfm = _get_load_transforms_25d(use_body_mask)
            self._aug_tfm = _get_aug_transforms_25d(mode, use_body_mask)
        
        print(f"Pre-scanning NIfTI files for 2.5D volumes (num_slices={num_slices}, stride={self.stride})...")
        
        for i, file_dict in enumerate(self.file_list):
            try:
                # Get volume shape
                nii = nib.load(file_dict["mr"])
                total_slices = nii.shape[-1]
                
                # Calculate valid starting positions for volumes
                # We need at least num_slices consecutive slices
                if total_slices < num_slices:
                    print(f"  Warning: Patient {file_dict.get('id', i)} has only {total_slices} slices, skipping.")
                    continue
                
                # Valid center indices (ensuring we can grab num_slices//2 on each side)
                half_depth = num_slices // 2
                start_slice = half_depth
                end_slice = total_slices - half_depth
                
                # Sample with stride
                valid_centers = list(range(start_slice, end_slice, self.stride))
                
                for center_idx in valid_centers:
                    self.volume_map.append((i, center_idx))
                
                print(f"  Patient {file_dict.get('id', i)}: {total_slices} total slices → {len(valid_centers)} volumes")
                
            except Exception as e:
                print(f"Warning: Could not read metadata for {file_dict['mr']}. Skipping file. Error: {e}")
        
        print(f"Total 2.5D volumes in dataset: {len(self.volume_map)}")

        if cache_volumes:
            print(f"Pre-loading {len(self.file_list)} volumes into RAM for 2.5D patches...")
            total_bytes = 0
            for i in range(len(self.file_list)):
                self._load_volume(i)
                vol = self._vol_cache[i]
                vol_bytes = vol["mr"].nelement() * vol["mr"].element_size()
                vol_bytes += vol["ct"].nelement() * vol["ct"].element_size()
                if vol["mask"] is not None:
                    vol_bytes += vol["mask"].nelement() * vol["mask"].element_size()
                total_bytes += vol_bytes
                if (i + 1) % 10 == 0 or i == len(self.file_list) - 1:
                    print(f"  Loaded {i + 1}/{len(self.file_list)} volumes "
                          f"({total_bytes / 1e9:.1f} GB RAM)")
            print(f"2.5D volume cache ready: {len(self.file_list)} volumes, "
                  f"{total_bytes / 1e9:.2f} GB RAM")
            del self._load_tfm

    def _load_volume(self, file_idx):
        """Load one full 3D volume into RAM. Called before DataLoader workers fork."""
        if file_idx in self._vol_cache:
            return self._vol_cache[file_idx]

        file_paths = self.file_list[file_idx]
        transform_input = {"mr": file_paths["mr"], "ct": file_paths["ct"]}
        if self.use_body_mask and file_paths.get("mask") is not None:
            transform_input["mask"] = file_paths["mask"]

        volumes = self._load_tfm(transform_input)
        cached = {
            "mr": volumes["mr"].data.clone() if hasattr(volumes["mr"], "data") else volumes["mr"].clone(),
            "ct": volumes["ct"].data.clone() if hasattr(volumes["ct"], "data") else volumes["ct"].clone(),
        }
        if self.use_body_mask and "mask" in volumes:
            m = volumes["mask"]
            cached["mask"] = m.data.clone() if hasattr(m, "data") else m.clone()
        else:
            cached["mask"] = None

        self._vol_cache[file_idx] = cached
        return cached
    
    def __len__(self):
        return len(self.volume_map)
    
    def __getitem__(self, idx):
        file_idx, center_idx = self.volume_map[idx]
        file_paths = self.file_list[file_idx]
        
        try:
            if self.cache_volumes:
                return self._getitem_cached(file_idx, center_idx)

            return self._getitem_uncached(file_idx, center_idx, file_paths)
            
        except Exception as e:
            print(f"Error loading 2.5D volume {idx} (file {file_paths.get('id', 'unknown')}, center {center_idx}): {e}")
            # Return zero tensors
            output = {
                "mr": torch.zeros(1, self.num_slices, self.target_size, self.target_size, dtype=torch.float32),
                "ct": torch.zeros(1, self.num_slices, self.target_size, self.target_size, dtype=torch.float32)
            }
            if self.use_body_mask:
                output["mask"] = torch.ones(1, self.num_slices, self.target_size, self.target_size, dtype=torch.float32)
            return output

    def _getitem_cached(self, file_idx, center_idx):
        """Fast path: extract a small patch from cached volumes and augment it."""
        cached = self._vol_cache[file_idx]

        half_depth = self.num_slices // 2
        start_idx = center_idx - half_depth
        end_idx = start_idx + self.num_slices

        # Cached volumes are [C, H, W, D]; model patches are [C, D, H, W].
        mr_volume = cached["mr"][:, :, :, start_idx:end_idx].clone().permute(0, 3, 1, 2)
        ct_volume = cached["ct"][:, :, :, start_idx:end_idx].clone().permute(0, 3, 1, 2)

        mask_volume = None
        if self.use_body_mask and cached["mask"] is not None:
            mask_volume = cached["mask"][:, :, :, start_idx:end_idx].clone()
            mask_volume = (mask_volume > 0).float().permute(0, 3, 1, 2)

        if self._aug_tfm is not None:
            aug_input = {"mr": mr_volume, "ct": ct_volume}
            if mask_volume is not None:
                aug_input["mask"] = mask_volume
            aug_out = self._aug_tfm(aug_input)
            mr_volume = aug_out["mr"]
            ct_volume = aug_out["ct"]
            if mask_volume is not None and "mask" in aug_out:
                mask_volume = aug_out["mask"]

        if mr_volume.shape[2] != self.target_size or mr_volume.shape[3] != self.target_size:
            C, D, H, W = mr_volume.shape
            mr_volume = mr_volume.reshape(C * D, H, W)
            ct_volume = ct_volume.reshape(C * D, H, W)

            mr_volume = torch.nn.functional.interpolate(
                mr_volume.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            ct_volume = torch.nn.functional.interpolate(
                ct_volume.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            mr_volume = mr_volume.reshape(C, D, self.target_size, self.target_size)
            ct_volume = ct_volume.reshape(C, D, self.target_size, self.target_size)

            if mask_volume is not None:
                Cm, Dm, Hm, Wm = mask_volume.shape
                mask_volume = mask_volume.reshape(Cm * Dm, Hm, Wm)
                mask_volume = torch.nn.functional.interpolate(
                    mask_volume.unsqueeze(0),
                    size=(self.target_size, self.target_size),
                    mode='nearest',
                ).squeeze(0)
                mask_volume = mask_volume.reshape(Cm, Dm, self.target_size, self.target_size)

        mr_volume = mr_volume.clamp(-1.0, 1.0)
        ct_volume = ct_volume.clamp(-1.0, 1.0)

        output = {
            "mr": mr_volume.float(),
            "ct": ct_volume.float()
        }
        if self.use_body_mask:
            if mask_volume is not None:
                output["mask"] = mask_volume.float()
            else:
                output["mask"] = torch.ones(1, self.num_slices, self.target_size, self.target_size, dtype=torch.float32)
        return output

    def _getitem_uncached(self, file_idx, center_idx, file_paths):
        """Original path: load full volumes from disk for each sampled patch."""
        # Load 3D volumes - conditionally include mask path
        transform_input = {"mr": file_paths["mr"], "ct": file_paths["ct"]}
        if self.use_body_mask and file_paths.get("mask") is not None:
            transform_input["mask"] = file_paths["mask"]

        volumes = self.transform(transform_input)
        mr_vol = volumes["mr"]  # [C, H, W, D]
        ct_vol = volumes["ct"]  # [C, H, W, D]

        # Extract consecutive slices centered at center_idx
        half_depth = self.num_slices // 2
        start_idx = center_idx - half_depth
        end_idx = start_idx + self.num_slices

        # Extract volume slices [C, H, W, num_slices]
        mr_volume = mr_vol[:, :, :, start_idx:end_idx]
        ct_volume = ct_vol[:, :, :, start_idx:end_idx]

        # Extract mask volume if available
        mask_volume = None
        if self.use_body_mask and "mask" in volumes:
            mask_vol = volumes["mask"]  # [C, H, W, D]
            mask_volume = mask_vol[:, :, :, start_idx:end_idx]
            mask_volume = (mask_volume > 0).float()  # Binary conversion

        # Resize spatial dimensions if needed
        if mr_volume.shape[1] != self.target_size or mr_volume.shape[2] != self.target_size:
            # Reshape to [C*D, H, W] for 2D interpolation
            C, H, W, D = mr_volume.shape
            mr_volume = mr_volume.permute(0, 3, 1, 2).reshape(C*D, H, W)
            ct_volume = ct_volume.permute(0, 3, 1, 2).reshape(C*D, H, W)

            mr_volume = torch.nn.functional.interpolate(
                mr_volume.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            ct_volume = torch.nn.functional.interpolate(
                ct_volume.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # Reshape back to [C, H, W, D]
            mr_volume = mr_volume.reshape(C, D, self.target_size, self.target_size).permute(0, 2, 3, 1)
            ct_volume = ct_volume.reshape(C, D, self.target_size, self.target_size).permute(0, 2, 3, 1)

            # Resize mask with nearest interpolation to preserve binary values
            if mask_volume is not None:
                Cm, Hm, Wm, Dm = mask_volume.shape
                mask_volume = mask_volume.permute(0, 3, 1, 2).reshape(Cm*Dm, Hm, Wm)
                mask_volume = torch.nn.functional.interpolate(
                    mask_volume.unsqueeze(0),
                    size=(self.target_size, self.target_size),
                    mode='nearest',
                ).squeeze(0)
                mask_volume = mask_volume.reshape(Cm, Dm, self.target_size, self.target_size).permute(0, 2, 3, 1)

        # Rearrange to [C, D, H, W] for 3D convolutions
        mr_volume = mr_volume.permute(0, 3, 1, 2)  # [C, D, H, W]
        ct_volume = ct_volume.permute(0, 3, 1, 2)  # [C, D, H, W]

        output = {
            "mr": mr_volume.float(),
            "ct": ct_volume.float()
        }

        if self.use_body_mask:
            if mask_volume is not None:
                mask_volume = mask_volume.permute(0, 3, 1, 2)  # [C, D, H, W]
                output["mask"] = mask_volume.float()
            else:
                output["mask"] = torch.ones(1, self.num_slices, self.target_size, self.target_size, dtype=torch.float32)

        return output
