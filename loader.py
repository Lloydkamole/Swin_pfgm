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
            data (list of dict): Each dict has 'mr', 'ct', and optionally 'mask' file paths.
            transform (callable): A MONAI Compose of transforms to apply to the 3D volumes.
            target_size (int): Target H/W for output 2D slices.
            use_body_mask (bool): If True, load and return body mask.
        """
        self.file_list = data
        self.transform = transform
        self.target_size = target_size
        self.use_body_mask = use_body_mask
        self.slice_map = []

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

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_map[idx]
        file_paths = self.file_list[file_idx]

        try:
            # Build transform input dict — only include mask key when a path exists
            # (passing None would cause LoadImaged to hang)
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

        except Exception as e:
            print(f"Error loading item {idx} (file {file_paths.get('id', 'unknown')}, slice {slice_idx}): {e}")
            output = {
                "mr": torch.zeros(1, self.target_size, self.target_size, dtype=torch.float32),
                "ct": torch.zeros(1, self.target_size, self.target_size, dtype=torch.float32),
            }
            if self.use_body_mask:
                output["mask"] = torch.ones(1, self.target_size, self.target_size, dtype=torch.float32)
            return output

