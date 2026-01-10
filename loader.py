# training/dataset_monai_nifti.py

import os
import torch
import random
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    RandFlipd,
    RandRotate90d,
    Resized
)
from torch.utils.data import Dataset
import nibabel as nib


# --- Transform Pipeline ---
def get_monai_nifti_transforms(mode="train", target_size=256):
    """
    Define MONAI transforms for loading NIfTI data.
    Data will be normalized to [-1, 1] range for PFGM++.
    """
    keys = ["mr", "ct"]
    
    base_transforms = [
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        # Normalize to [-1, 1] range
        # ScaleIntensityRanged(
        #     keys=["mr"],
        #     a_min=0.0, a_max=1.0,  # Input range
        #     b_min=-1.0, b_max=1.0,  # Output range for PFGM++
        #     clip=True
        # ),
        # ScaleIntensityRanged(
        #     keys=["ct"],
        #     a_min=0.0, a_max=1.0,
        #     b_min=-1.0, b_max=1.0,
        #     clip=True
        # ),
    ]
    
    if mode == "train":
        # Add augmentation for training
        transforms = base_transforms + [
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),  # Flip H
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),  # Flip W
            #RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
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
    def __init__(self, data, transform, target_size=256, use_body_mask=False):
        """
        Args:
            data (list of dict): A list where each dict contains file paths for 'mr', 'ct', 
                                 and optionally 'mask'.
            transform (callable): A MONAI Compose object of transforms to apply to the 3D volumes.
            target_size (int): Target size for 2D slices.
            use_body_mask (bool): If True, load and return body mask for anatomically-focused training.
        """
        self.file_list = data
        self.transform = transform
        self.target_size = target_size
        self.use_body_mask = use_body_mask
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

    def __len__(self):
        # The length is the total number of slices across all patient volumes
        return len(self.slice_map)

    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_map[idx]
        file_paths = self.file_list[file_idx]

        try:
            # Apply transforms to get 3D volume
            volumes = self.transform(file_paths)
            mr_vol = volumes["mr"]  # [C, H, W, D]
            ct_vol = volumes["ct"]  # [C, H, W, D]
            
            # Extract 2D slice
            mr_slice = mr_vol[:, :, :, slice_idx]  # [C, H, W]
            ct_slice = ct_vol[:, :, :, slice_idx]  # [C, H, W]
            
            # Load mask if enabled
            mask_slice = None
            if self.use_body_mask and "mask" in file_paths and file_paths["mask"] is not None:
                try:
                    mask_nii = nib.load(file_paths["mask"])
                    mask_data = mask_nii.get_fdata()
                    # Handle mask shape - should match MR/CT
                    if mask_data.ndim == 3:
                        mask_2d = mask_data[:, :, slice_idx]
                    else:
                        mask_2d = mask_data
                    # Binarize mask (non-zero = body)
                    mask_2d = (mask_2d > 0).astype(np.float32)
                    mask_slice = torch.from_numpy(mask_2d).unsqueeze(0)  # [1, H, W]
                except Exception as e:
                    print(f"Warning: Could not load mask for {file_paths.get('id', 'unknown')}: {e}")
                    mask_slice = None
            
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
            
            # Build output dict
            output = {"mr": mr_slice.float(), "ct": ct_slice.float()}
            
            # Add mask if available, otherwise create full mask (all ones)
            if self.use_body_mask:
                if mask_slice is not None:
                    output["mask"] = mask_slice.float()
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

