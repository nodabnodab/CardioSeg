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

def preprocess_and_save():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    acdc_dir = os.path.join(base_dir, "data", "ACDC")
    assets_dir = os.path.join(base_dir, "assets")
    
    image_path = os.path.join(acdc_dir, "Images", "patient001_frame01.nii.gz")
    mask_path = os.path.join(acdc_dir, "Masks", "patient001_frame01_gt.nii.gz")
    
    # 1. Define Transform Pipeline (using standard bilinear for 3D volume compatibility)
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"], 
            pixdim=[1.25, 1.25, 5.0], 
            mode=("bilinear", "nearest") 
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
    ])
    
    # Run
    data_dict = {"image": image_path, "label": mask_path}
    preprocessed = transforms(data_dict)
    
    prep_img = preprocessed["image"].numpy()[0]
    prep_mask = preprocessed["label"].numpy()[0]
    
    # Raw data for comparison
    img_nib = nib.load(image_path)
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
    raw_rgba[raw_mask_slice == 1] = [0.12, 0.47, 0.71, 0.5] # RV
    raw_rgba[raw_mask_slice == 2] = [1.0, 0.5, 0.0, 0.5]    # MYO
    raw_rgba[raw_mask_slice == 3] = [0.84, 0.15, 0.16, 0.5] # LV
    
    axes[0, 1].imshow(raw_img_norm, cmap='gray')
    axes[0, 1].imshow(raw_rgba)
    axes[0, 1].set_title("Original Mask Overlay")
    axes[0, 1].axis('off')
    
    # Row 2: Preprocessed Image & Preprocessed Mask (with Contrast Windowing)
    prep_img_slice = np.rot90(prep_img[:, :, prep_slice_idx])
    prep_mask_slice = np.rot90(prep_mask[:, :, prep_slice_idx])
    
    # Medical windowing: clip extreme values (under 2nd and over 98th percentiles)
    # This prevents extreme noise values from washing out the actual heart tissue.
    p_min, p_max = np.percentile(prep_img_slice, (2, 98))
        
    axes[1, 0].imshow(prep_img_slice, cmap='gray', vmin=p_min, vmax=p_max)
    axes[1, 0].set_title(f"Preprocessed (Slice {prep_slice_idx})\n[Bilinear + Contrast Windowing]")
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
    save_path = os.path.join(assets_dir, "preprocessing_comparison_contrast.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated: preprocessing_comparison_contrast.png")
    print(f"Window bounds (2% - 98%): Min={p_min:.4f}, Max={p_max:.4f}")

if __name__ == "__main__":
    print("="*60)
    print("RUNNING ADVANCED PREPROCESSING VISUALIZATIONS")
    print("="*60)
    preprocess_and_save()
    print("="*60)
    print("All comparison assets saved successfully in assets/")
    print("="*60)
