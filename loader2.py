# training/dataset_monai_nifti.py

import os
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
from monai.data import CacheDataset
import nibabel as nib


# --- Transform Pipeline ---
def get_monai_nifti_transforms(mode="train", target_size=256, use_body_mask=False):
    """
    Define MONAI transforms for loading NIfTI data.
    Assumes data is already normalized offline.
    """
    keys = ["mr", "ct"]
    if use_body_mask:
        keys.append("mask")
    modes = ["bilinear", "bilinear"] + (["nearest"] if use_body_mask else [])
    
    base_transforms = [
        LoadImaged(keys=keys, image_only=True, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
    ]
    
    if mode == "train":
        # Add conservative augmentation for training.
        # Spatial transforms are applied jointly to MR/CT/(mask) to preserve alignment.
        transforms = base_transforms + [
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
            RandAffined(
                keys=keys,
                prob=0.2,
                rotate_range=(0.0, 0.0, np.deg2rad(5.0)),
                scale_range=(0.05, 0.05, 0.0),
                mode=modes,
                padding_mode="border",
                allow_missing_keys=True,
            ),
            RandGaussianNoised(keys=["mr"], prob=0.15, mean=0.0, std=0.02),
            RandAdjustContrastd(keys=["mr"], prob=0.15, gamma=(0.9, 1.1)),
        ]
    else:
        transforms = base_transforms
    
    return Compose(transforms)


# --- PyTorch Dataset Class ---
class MonaiNiftiSliceDataset(Dataset):
    """
    A PyTorch Dataset that treats all 2D slices from a collection of 3D NIfTI 
    volumes as a single large dataset. Returns slices in [-1, 1] range.
    
    Supports optional body mask loading for anatomically-focused training (nnsyn style).
    """
    def __init__(self, data, transform, target_size=256, use_body_mask=False, cache_rate=1.0):
        """
        Args:
            data (list of dict): A list where each dict contains file paths for 'mr', 'ct', 
                                 and optionally 'mask'.
            transform (callable): A MONAI Compose object of transforms to apply to the 3D volumes.
            target_size (int): Target size for 2D slices.
            use_body_mask (bool): If True, load and return body mask for anatomically-focused training.
            cache_rate (float): Fraction of samples to cache in CPU RAM via MONAI CacheDataset.
        """
        self.file_list = data
        self.transform = transform
        self.target_size = target_size
        self.use_body_mask = use_body_mask
        self.cache_rate = float(max(0.0, min(1.0, cache_rate)))
        self.slice_map = []

        print("Pre-scanning NIfTI files to create slice map (using ALL slices)...")
        for i, file_dict in enumerate(self.file_list):
            try:
                # Use nibabel to quickly get the shape without loading the whole image data
                nii = nib.load(file_dict["mr"])
                num_slices = nii.shape[-1]
                # Use ALL slices
                for s in range(num_slices):
                    self.slice_map.append((i, s))
                print(f"  Patient {file_dict.get('id', i)}: {num_slices} slices")
            except Exception as e:
                print(f"Warning: Could not read metadata for {file_dict['mr']}. Skipping file. Error: {e}")
        
        print(f"Total slices in dataset: {len(self.slice_map)}")
        if use_body_mask:
            print("Body mask loading ENABLED - will load mask files for anatomically-focused training")

        # Cache deterministic transform outputs in CPU RAM to speed up repeated volume loading.
        # CacheDataset applies random transforms after the cached deterministic prefix.
        cache_num = int(len(self.file_list) * self.cache_rate)
        self.volume_dataset = CacheDataset(
            data=self.file_list,
            transform=self.transform,
            cache_rate=self.cache_rate,
            cache_num=cache_num,
            num_workers=0,
        )
        print(f"Volume caching ENABLED: cache_rate={self.cache_rate:.2f} ({cache_num}/{len(self.file_list)} volumes)")

    def __len__(self):
        # The length is the total number of slices across all patient volumes
        return len(self.slice_map)

    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_map[idx]
        file_paths = self.file_list[file_idx]

        try:
            # Apply transforms to get 3D volume (served via CacheDataset when possible)
            volumes = self.transform[file_idx]
            mr_vol = volumes["mr"]  # [C, H, W, D]
            ct_vol = volumes["ct"]  # [C, H, W, D]
            
            # Extract 2D slice
            mr_slice = mr_vol[:, :, :, slice_idx]  # [C, H, W]
            ct_slice = ct_vol[:, :, :, slice_idx]  # [C, H, W]
            
            # Load mask if enabled
            mask_slice = None
            if self.use_body_mask and "mask" in volumes:
                mask_vol = volumes["mask"]
                mask_slice = mask_vol[:, :, :, slice_idx]
            
            # Resize to target size if needed
            if mr_slice.shape[1] != self.target_size or mr_slice.shape[2] != self.target_size:
                mr_slice = torch.nn.functional.interpolate(
                    mr_slice.unsqueeze(0), 
                    size=(self.target_size, self.target_size), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                ct_slice = torch.nn.functional.interpolate(
                    ct_slice.unsqueeze(0), 
                    size=(self.target_size, self.target_size), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                # Resize mask using nearest-neighbor to preserve binary values
                if mask_slice is not None:
                    mask_slice = torch.nn.functional.interpolate(
                        mask_slice.unsqueeze(0),
                        size=(self.target_size, self.target_size),
                        mode='nearest'
                    ).squeeze(0)

                    # Keep intensity bounds stable after MRI intensity augmentation
                    mr_slice = mr_slice.clamp(-1.0, 1.0)
                    ct_slice = ct_slice.clamp(-1.0, 1.0)
            
            # Build output dict
            output = {"mr": mr_slice.float(), "ct": ct_slice.float()}
            
            # Add mask if available, otherwise create full mask (all ones)
            if self.use_body_mask:
                if mask_slice is not None:
                    output["mask"] = (mask_slice > 0.5).float()
                else:
                    # Fallback: use all ones if mask not available
                    output["mask"] = torch.ones(1, self.target_size, self.target_size, dtype=torch.float32)
            
            return output

        except Exception as e:
            print(f"Error loading item {idx} (file {file_paths.get('id', 'unknown')}, slice {slice_idx}): {e}")
            # Return zero tensors in [-1, 1] range (actually zeros are fine)
            output = {
                "mr": torch.zeros(1, self.target_size, self.target_size, dtype=torch.float32),
                "ct": torch.zeros(1, self.target_size, self.target_size, dtype=torch.float32)
            }
            if self.use_body_mask:
                output["mask"] = torch.ones(1, self.target_size, self.target_size, dtype=torch.float32)
            return output