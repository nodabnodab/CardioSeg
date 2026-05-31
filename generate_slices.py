import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def create_and_save_slices():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    acdc_dir = os.path.join(base_dir, "data", "ACDC")
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    image_path = os.path.join(acdc_dir, "Images", "patient001_frame01.nii.gz")
    mask_path = os.path.join(acdc_dir, "Masks", "patient001_frame01_gt.nii.gz")
    
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print("Error: Extracted ACDC data files not found. Make sure inspect_data.py has run successfully.")
        return

    # Load 3D volumes
    img = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    
    # We choose slice index 5 (middle slice of the 10 slices in Z-axis)
    slice_idx = 5
    mri_slice = img[:, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]
    
    # Simple rotation to orient it upright for normal view (NIfTI coordinates are sometimes rotated)
    # Rotating 90 degrees counter-clockwise or transposing might align it.
    mri_slice = np.rot90(mri_slice)
    mask_slice = np.rot90(mask_slice)
    
    # Normalize MRI intensity for plotting
    mri_slice_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min())

    # 1. Save Raw MRI Slice
    plt.figure(figsize=(6, 6))
    plt.imshow(mri_slice_norm, cmap='gray')
    plt.title(f"Raw Cardiac MRI (Slice {slice_idx})", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "mri_raw.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Save Mask Slice
    plt.figure(figsize=(6, 6))
    # We define a custom discrete colormap for labels: 0=black, 1=cyan(RV), 2=yellow(MYO), 3=red(LV)
    from matplotlib.colors import ListedColormap
    colors = ['black', '#1f77b4', '#ff7f0e', '#d62728'] # blue, orange, red
    cmap = ListedColormap(colors)
    plt.imshow(mask_slice, cmap=cmap, vmin=0, vmax=3)
    plt.title(f"Segmentation Mask (Slice {slice_idx})", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "mask_only.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Save Overlay Slice
    plt.figure(figsize=(6, 6))
    plt.imshow(mri_slice_norm, cmap='gray')
    
    # Mask overlay: make background (label 0) transparent
    mask_rgba = np.zeros((*mask_slice.shape, 4))
    # Label 1 (RV): Blue
    mask_rgba[mask_slice == 1] = [0.12, 0.47, 0.71, 0.45] # rgba (cyan-blue, 45% alpha)
    # Label 2 (MYO): Orange (Myocardium)
    mask_rgba[mask_slice == 2] = [1.0, 0.5, 0.0, 0.45]    # rgba (orange, 45% alpha)
    # Label 3 (LV): Red (Left Ventricle)
    mask_rgba[mask_slice == 3] = [0.84, 0.15, 0.16, 0.45] # rgba (red, 45% alpha)
    
    plt.imshow(mask_rgba)
    plt.title(f"MRI & Mask Overlay (Slice {slice_idx})", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "mri_overlay.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated visual assets inside {assets_dir}:")
    print("  - mri_raw.png")
    print("  - mask_only.png")
    print("  - mri_overlay.png")

if __name__ == "__main__":
    create_and_save_slices()
