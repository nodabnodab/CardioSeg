import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    CropForegroundd,
    NormalizeIntensityd
)

def run_preprocessing_visualization():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    acdc_dir = os.path.join(base_dir, "data", "ACDC")
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    image_path = os.path.join(acdc_dir, "Images", "patient001_frame01.nii.gz")
    mask_path = os.path.join(acdc_dir, "Masks", "patient001_frame01_gt.nii.gz")
    
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print("Error: Extracted dataset not found in data/ACDC. Make sure inspect_data.py has run.")
        return

    # 1. Print Original Metadata
    img_nib = nib.load(image_path)
    print("="*60)
    print("=== ORIGINAL NIFTI METADATA ===")
    print(f"Shape: {img_nib.shape} (X, Y, Z)")
    print(f"Spacing: {img_nib.header.get_zooms()} (dx, dy, dz)")
    print("="*60)
    
    # 2. Define MONAI Preprocessing Pipeline
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Resample voxel sizes to a standardized 1.25 x 1.25 x 5.0 mm
        Spacingd(
            keys=["image", "label"], 
            pixdim=[1.25, 1.25, 5.0], 
            mode=("bilinear", "nearest")
        ),
        # Crop to the foreground bounding box to remove empty space
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Normalize voxel intensity values
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
    ])
    
    # Run the pipeline
    print("Running MONAI preprocessing pipeline...")
    data_dict = {"image": image_path, "label": mask_path}
    preprocessed = transforms(data_dict)
    
    # Convert MONAI/PyTorch tensors back to NumPy for plotting
    prep_img = preprocessed["image"].numpy()[0] # Shape: [H, W, D]
    prep_mask = preprocessed["label"].numpy()[0]
    
    # Retrieve new spacing from metadata if available, otherwise print target spacing
    try:
        new_spacing = preprocessed["image"].meta["pixdim"][1:4].tolist()
    except Exception:
        new_spacing = [1.25, 1.25, 5.0]
        
    print("\n=== PREPROCESSED METADATA (After MONAI Pipeline) ===")
    print(f"Shape: {prep_img.shape} (X, Y, Z)")
    print(f"Spacing: {new_spacing} (dx, dy, dz)")
    print(f"Intensity Range: Min={prep_img.min():.4f}, Max={prep_img.max():.4f}, Mean={prep_img.mean():.4f}")
    print("="*60)
    
    # 3. Create Before/After Visualizations
    raw_img = img_nib.get_fdata()
    raw_mask = nib.load(mask_path).get_fdata()
    
    raw_slice_idx = 5
    prep_slice_idx = prep_img.shape[2] // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Row 1: Raw Image & Raw Mask
    raw_img_slice = np.rot90(raw_img[:, :, raw_slice_idx])
    raw_mask_slice = np.rot90(raw_mask[:, :, raw_slice_idx])
    raw_img_norm = (raw_img_slice - raw_img_slice.min()) / (raw_img_slice.max() - raw_img_slice.min())
    
    axes[0, 0].imshow(raw_img_norm, cmap='gray')
    axes[0, 0].set_title(f"Original Image (Slice {raw_slice_idx})")
    axes[0, 0].axis('off')
    
    raw_rgba = np.zeros((*raw_mask_slice.shape, 4))
    raw_rgba[raw_mask_slice == 1] = [0.12, 0.47, 0.71, 0.5] # RV (Blue)
    raw_rgba[raw_mask_slice == 2] = [1.0, 0.5, 0.0, 0.5]    # MYO (Orange)
    raw_rgba[raw_mask_slice == 3] = [0.84, 0.15, 0.16, 0.5] # LV (Red)
    
    axes[0, 1].imshow(raw_img_norm, cmap='gray')
    axes[0, 1].imshow(raw_rgba)
    axes[0, 1].set_title("Original Mask Overlay")
    axes[0, 1].axis('off')
    
    # Row 2: Preprocessed Image & Preprocessed Mask
    prep_img_slice = np.rot90(prep_img[:, :, prep_slice_idx])
    prep_mask_slice = np.rot90(prep_mask[:, :, prep_slice_idx])
    
    # Clips the intensity values slightly for visualization contrast
    p_min, p_max = prep_img_slice.min(), prep_img_slice.max()
    
    axes[1, 0].imshow(prep_img_slice, cmap='gray', vmin=p_min, vmax=p_max)
    axes[1, 0].set_title(f"Preprocessed (Slice {prep_slice_idx})\n[Resampled + Cropped]")
    axes[1, 0].axis('off')
    
    prep_rgba = np.zeros((*prep_mask_slice.shape, 4))
    prep_rgba[prep_mask_slice == 1] = [0.12, 0.47, 0.71, 0.5] # RV
    prep_rgba[prep_mask_slice == 2] = [1.0, 0.5, 0.0, 0.5]    # MYO
    prep_rgba[prep_mask_slice == 3] = [0.84, 0.15, 0.16, 0.5] # LV
    
    axes[1, 1].imshow(prep_img_slice, cmap='gray', vmin=p_min, vmax=p_max)
    axes[1, 1].imshow(prep_rgba)
    axes[1, 1].set_title("Preprocessed Mask Overlay")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(assets_dir, "preprocessing_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSuccessfully saved preprocessing comparison image to:\n{comparison_path}")

if __name__ == "__main__":
    run_preprocessing_visualization()
