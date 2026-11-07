# training/dataset_monai_nifti.py

import torch
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
)
from torch.utils.data import Dataset
import nibabel as nib


# --- Transform Pipeline ---
def get_monai_nifti_transforms(mode="train", target_size=256, use_body_mask=False):
    """
    Define MONAI transforms for loading NIfTI data.
    Assumes data is already normalized offline to [-1, 1].

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
            RandFlipd(keys=keys_with_mask, prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=keys_with_mask, prob=0.5, spatial_axis=1, allow_missing_keys=True),
            RandAffined(
                keys=keys_with_mask,
                prob=0.2,
                rotate_range=(0.0, 0.0, np.deg2rad(5.0)),
                scale_range=(0.05, 0.05, 0.0),
                mode=resize_modes,
                padding_mode="border",
                allow_missing_keys=True,
            ),
            RandGaussianNoised(keys=["mr"], prob=0.15, mean=0.0, std=0.02),
            RandAdjustContrastd(keys=["mr"], prob=0.15, gamma=(0.9, 1.1)),
        ]
    else:
        transforms = base_transforms

    return Compose(transforms)


def _get_load_transforms(use_body_mask=False):
    """Deterministic transforms: load + orient.  Results can be cached."""
    keys_with_mask = ["mr", "ct", "mask"] if use_body_mask else ["mr", "ct"]
    return Compose([
        LoadImaged(keys=keys_with_mask, image_only=True, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys_with_mask, allow_missing_keys=True),
        Orientationd(keys=keys_with_mask, axcodes="RAS", allow_missing_keys=True),
    ])


def _get_aug_transforms(mode="train", use_body_mask=False):
    """Random augmentations applied per-sample (not cached).
    Returns None for val mode.  For train mode returns a callable that
    takes and returns a dict of [C, H, W] tensors using fast PyTorch ops
    instead of MONAI 3-D transforms."""
    if mode != "train":
        return None

    def _augment(sample):
        keys = list(sample.keys())

        # Random horizontal flip (axis 0 = H)
        if torch.rand(1).item() < 0.5:
            for k in keys:
                sample[k] = sample[k].flip(1)

        # Random vertical flip (axis 1 = W)
        if torch.rand(1).item() < 0.5:
            for k in keys:
                sample[k] = sample[k].flip(2)

        # Random affine (prob 0.2): small rotation + scale
        if torch.rand(1).item() < 0.2:
            angle = (torch.rand(1).item() - 0.5) * 2 * np.deg2rad(5.0)  # [-5°, 5°]
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.1           # [0.95, 1.05]
            cos_a, sin_a = np.cos(angle) * scale, np.sin(angle) * scale
            # 2×3 affine matrix
            theta = torch.tensor([[cos_a, -sin_a, 0.0],
                                  [sin_a,  cos_a, 0.0]], dtype=torch.float32).unsqueeze(0)
            for k in keys:
                img = sample[k].unsqueeze(0)  # [1, C, H, W]
                grid = torch.nn.functional.affine_grid(theta, img.shape, align_corners=False)
                mode = "nearest" if k == "mask" else "bilinear"
                pad = "zeros" if k == "mask" else "border"
                sample[k] = torch.nn.functional.grid_sample(
                    img, grid, mode=mode, padding_mode=pad, align_corners=False
                ).squeeze(0)

        # Gaussian noise on MR only (prob 0.15)
        if "mr" in sample and torch.rand(1).item() < 0.15:
            sample["mr"] = sample["mr"] + torch.randn_like(sample["mr"]) * 0.02

        # Random contrast on MR only (prob 0.15)
        if "mr" in sample and torch.rand(1).item() < 0.15:
            gamma = 0.9 + torch.rand(1).item() * 0.2  # [0.9, 1.1]
            # Shift to [0,1], apply gamma, shift back to [-1,1]
            mr = (sample["mr"] + 1.0) / 2.0
            mr = mr.clamp(0, 1).pow(gamma)
            sample["mr"] = mr * 2.0 - 1.0

        return sample

    return _augment


# --- PyTorch Dataset Class ---
class MonaiNiftiSliceDataset(Dataset):
    """
    A PyTorch Dataset that treats all 2D slices from a collection of 3D NIfTI
    volumes as a single large dataset. Returns slices in [-1, 1] range.

    Supports optional body mask loading for anatomically-focused training (nnsyn style).

    Volume caching: 3D volumes are loaded from disk ONCE and cached in RAM.
    Only random augmentations are applied per-sample. This avoids re-reading
    and decompressing .nii.gz files on every __getitem__ call (~5-10x speedup).
    """

    def __init__(self, data, transform, target_size=256, use_body_mask=False,
                 cache_volumes=True, mode="train"):
        """
        Args:
            data (list of dict): Each dict has 'mr', 'ct', and optionally 'mask' file paths.
            transform (callable): A MONAI Compose of transforms (used only when cache_volumes=False).
            target_size (int): Target H/W for output 2D slices.
            use_body_mask (bool): If True, load and return body mask.
            cache_volumes (bool): If True, cache loaded volumes in RAM (recommended).
            mode (str): "train" or "val" — controls which augmentations are applied.
        """
        self.file_list = data
        self.transform = transform
        self.target_size = target_size
        self.use_body_mask = use_body_mask
        self.cache_volumes = cache_volumes
        self.mode = mode
        self.slice_map = []

        # Volume cache: file_idx → {"mr": tensor, "ct": tensor, "mask": tensor|None}
        self._vol_cache = {}

        # Set up split transforms for cached mode
        if cache_volumes:
            self._load_tfm = _get_load_transforms(use_body_mask)
            self._aug_tfm = _get_aug_transforms(mode, use_body_mask)

        print("Pre-scanning NIfTI files to create slice map (using ALL slices)...")
        for i, file_dict in enumerate(self.file_list):
            try:
                nii = nib.load(file_dict["mr"])
                num_slices = nii.shape[-1]
                for s in range(num_slices):
                    self.slice_map.append((i, s))
                print(f"  Patient {file_dict.get('id', i)}: {num_slices} slices")
            except Exception as e:
                print(f"Warning: Could not read metadata for {file_dict['mr']}. Skipping. Error: {e}")

        print(f"Total slices in dataset: {len(self.slice_map)}")
        if use_body_mask:
            print("Body mask loading ENABLED - will load mask files for anatomically-focused training")

        # Eagerly load ALL volumes into RAM before DataLoader workers are forked.
        # On Linux, fork() uses copy-on-write: all worker processes share these
        # read-only pages without duplicating memory.
        if cache_volumes:
            print(f"Pre-loading {len(self.file_list)} volumes into RAM...")
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
            print(f"Volume cache ready: {len(self.file_list)} volumes, "
                  f"{total_bytes / 1e9:.2f} GB RAM")
            # Free the load transform pipeline — no longer needed
            del self._load_tfm

    def _load_volume(self, file_idx):
        """Load a 3D volume into cache. Called during __init__ only."""
        if file_idx in self._vol_cache:
            return self._vol_cache[file_idx]

        file_paths = self.file_list[file_idx]
        transform_input = {"mr": file_paths["mr"], "ct": file_paths["ct"]}
        if self.use_body_mask and file_paths.get("mask") is not None:
            transform_input["mask"] = file_paths["mask"]

        volumes = self._load_tfm(transform_input)

        # Convert to plain tensors (detach from MONAI MetaTensor to avoid
        # the slow __torch_function__ dispatch in workers)
        cached = {
            "mr": volumes["mr"].data.clone() if hasattr(volumes["mr"], 'data') else volumes["mr"].clone(),
            "ct": volumes["ct"].data.clone() if hasattr(volumes["ct"], 'data') else volumes["ct"].clone(),
        }
        if self.use_body_mask and "mask" in volumes:
            m = volumes["mask"]
            cached["mask"] = m.data.clone() if hasattr(m, 'data') else m.clone()
        else:
            cached["mask"] = None

        self._vol_cache[file_idx] = cached
        return cached

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_map[idx]
        file_paths = self.file_list[file_idx]

        try:
            if self.cache_volumes:
                return self._getitem_cached(file_idx, slice_idx)
            else:
                return self._getitem_uncached(file_idx, slice_idx, file_paths)
        except Exception as e:
            print(f"Error loading item {idx} (file {file_paths.get('id', 'unknown')}, slice {slice_idx}): {e}")
            output = {
                "mr": torch.zeros(1, self.target_size, self.target_size, dtype=torch.float32),
                "ct": torch.zeros(1, self.target_size, self.target_size, dtype=torch.float32),
            }
            if self.use_body_mask:
                output["mask"] = torch.ones(1, self.target_size, self.target_size, dtype=torch.float32)
            return output

    def _getitem_cached(self, file_idx, slice_idx):
        """Fast path: volume already in RAM, only apply augmentations to 2D slice."""
        cached = self._vol_cache[file_idx]

        # Extract 2D slices from cached 3D volumes
        mr_slice = cached["mr"][:, :, :, slice_idx].clone()  # [C, H, W]
        ct_slice = cached["ct"][:, :, :, slice_idx].clone()

        mask_slice = None
        if self.use_body_mask and cached["mask"] is not None:
            mask_slice = cached["mask"][:, :, :, slice_idx].clone()
            mask_slice = (mask_slice > 0).float()

        # Apply random augmentations on the 2D slice
        if self._aug_tfm is not None:
            aug_input = {"mr": mr_slice, "ct": ct_slice}
            if mask_slice is not None:
                aug_input["mask"] = mask_slice
            aug_out = self._aug_tfm(aug_input)
            mr_slice = aug_out["mr"]
            ct_slice = aug_out["ct"]
            if mask_slice is not None and "mask" in aug_out:
                mask_slice = aug_out["mask"]

        # Resize to target size if needed
        if mr_slice.shape[1] != self.target_size or mr_slice.shape[2] != self.target_size:
            mr_slice = torch.nn.functional.interpolate(
                mr_slice.unsqueeze(0), size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0)
            ct_slice = torch.nn.functional.interpolate(
                ct_slice.unsqueeze(0), size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0)
            if mask_slice is not None:
                mask_slice = torch.nn.functional.interpolate(
                    mask_slice.unsqueeze(0), size=(self.target_size, self.target_size),
                    mode='nearest',
                ).squeeze(0)

        mr_slice = mr_slice.clamp(-1.0, 1.0)
        ct_slice = ct_slice.clamp(-1.0, 1.0)

        output = {"mr": mr_slice.float(), "ct": ct_slice.float()}
        if self.use_body_mask:
            if mask_slice is not None:
                output["mask"] = mask_slice.float()
            else:
                output["mask"] = torch.ones(1, self.target_size, self.target_size, dtype=torch.float32)
        return output

    def _getitem_uncached(self, file_idx, slice_idx, file_paths):
        """Original path: load from disk every time (fallback)."""
        # Build transform input dict — only include mask key when a path exists
        transform_input = {"mr": file_paths["mr"], "ct": file_paths["ct"]}
        if self.use_body_mask and file_paths.get("mask") is not None:
            transform_input["mask"] = file_paths["mask"]

        # Run the full MONAI transform pipeline (load + orient + augment)
        volumes = self.transform(transform_input)
        mr_vol = volumes["mr"]   # [C, H, W, D]
        ct_vol = volumes["ct"]   # [C, H, W, D]

        # Extract 2D slice
        mr_slice = mr_vol[:, :, :, slice_idx]  # [C, H, W]
        ct_slice = ct_vol[:, :, :, slice_idx]

        # Extract mask slice (spatially aligned via joint transforms)
        mask_slice = None
        if self.use_body_mask and "mask" in volumes:
            mask_vol = volumes["mask"]
            mask_slice = mask_vol[:, :, :, slice_idx]
            mask_slice = (mask_slice > 0).float()

        # Resize to target size if needed
        if mr_slice.shape[1] != self.target_size or mr_slice.shape[2] != self.target_size:
            mr_slice = torch.nn.functional.interpolate(
                mr_slice.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            ct_slice = torch.nn.functional.interpolate(
                ct_slice.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            if mask_slice is not None:
                mask_slice = torch.nn.functional.interpolate(
                    mask_slice.unsqueeze(0),
                    size=(self.target_size, self.target_size),
                    mode='nearest',
                ).squeeze(0)

        # Clamp after interpolation + augmentation
        mr_slice = mr_slice.clamp(-1.0, 1.0)
        ct_slice = ct_slice.clamp(-1.0, 1.0)

        # Build output
        output = {"mr": mr_slice.float(), "ct": ct_slice.float()}

        if self.use_body_mask:
            if mask_slice is not None:
                output["mask"] = mask_slice.float()
            else:
                output["mask"] = torch.ones(1, self.target_size, self.target_size, dtype=torch.float32)

        return output

