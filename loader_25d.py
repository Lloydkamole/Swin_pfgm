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
)
from torch.utils.data import Dataset
import nibabel as nib


def get_monai_nifti_transforms_25d(mode="train"):
    """
    Define MONAI transforms for loading 3D NIfTI data.
    No augmentation for now - can be added later with 3D augmentations.
    """
    keys = ["mr", "ct"]
    
    transforms = [
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
    ]
    
    # Note: We'll do intensity normalization per-slice in the dataset
    # to maintain compatibility with 2D preprocessing
    
    return Compose(transforms)


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
    
    def __init__(self, data, transform, target_size=256, num_slices=5, stride=None):
        self.file_list = data
        self.transform = transform
        self.target_size = target_size
        self.num_slices = num_slices
        self.stride = stride if stride is not None else num_slices
        self.volume_map = []
        
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
    
    def __len__(self):
        return len(self.volume_map)
    
    def __getitem__(self, idx):
        file_idx, center_idx = self.volume_map[idx]
        file_paths = self.file_list[file_idx]
        
        try:
            # Load 3D volumes
            volumes = self.transform(file_paths)
            mr_vol = volumes["mr"]  # [C, H, W, D]
            ct_vol = volumes["ct"]  # [C, H, W, D]
            
            # Extract consecutive slices centered at center_idx
            half_depth = self.num_slices // 2
            start_idx = center_idx - half_depth
            end_idx = start_idx + self.num_slices
            
            # Extract volume slices [C, H, W, num_slices]
            mr_volume = mr_vol[:, :, :, start_idx:end_idx]
            ct_volume = ct_vol[:, :, :, start_idx:end_idx]
            
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
            
            # Rearrange to [C, D, H, W] for 3D convolutions
            mr_volume = mr_volume.permute(0, 3, 1, 2)  # [C, D, H, W]
            ct_volume = ct_volume.permute(0, 3, 1, 2)  # [C, D, H, W]
            
            return {
                "mr": mr_volume.float(),
                "ct": ct_volume.float()
            }
            
        except Exception as e:
            print(f"Error loading 2.5D volume {idx} (file {file_paths.get('id', 'unknown')}, center {center_idx}): {e}")
            # Return zero tensors
            return {
                "mr": torch.zeros(1, self.num_slices, self.target_size, self.target_size, dtype=torch.float32),
                "ct": torch.zeros(1, self.num_slices, self.target_size, self.target_size, dtype=torch.float32)
            }
