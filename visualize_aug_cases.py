import os
import matplotlib.pyplot as plt
import numpy as np
import monai
from monai.data import Dataset
from src.dataset import get_acdc_splits
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    CropForegroundd,
    ScaleIntensityRangePercentilesd,
    RandRotated,
    RandZoomd,
    RandScaleIntensityd,
    RandShiftIntensityd
)

def get_vis_aug_transforms():
    """
    Returns a pipeline that applies PREPROCESSING + AUGMENTATIONS (No Cropping)
    with 100% probability for clear visual demonstration.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=[1.25, 1.25, 5.0],
            mode=("bilinear", "nearest")
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=2.0,
            upper=98.0,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        # Force transformations (prob=1.0) to guarantee a visual difference in the demo
        RandRotated(
            keys=["image", "label"],
            range_x=0.0,
            range_y=0.0,
            range_z=0.35, # slightly larger rotation (~20 deg) for clearer visualization
            prob=1.0,
            mode=("bilinear", "nearest")
        ),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.85, # slightly wider zoom range
            max_zoom=1.15,
            prob=1.0,
            mode=("bilinear", "nearest")
        ),
        RandScaleIntensityd(keys=["image"], factors=0.15, prob=1.0),
        RandShiftIntensityd(keys=["image"], offsets=0.15, prob=1.0)
    ])

def get_vis_prep_transforms():
    """
    Returns prep-only pipeline (no distortion).
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=[1.25, 1.25, 5.0],
            mode=("bilinear", "nearest")
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=2.0,
            upper=98.0,
            b_min=0.0,
            b_max=1.0,
            clip=True
        )
    ])

def generate_multi_case_visualizations():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    print("="*60)
    print("GENERATING INTERACTIVE MULTI-CASE DATA AUGMENTATION DEMO")
    print("="*60)
    
    # 1. Get split files
    train_files, _, _ = get_acdc_splits(base_dir)
    
    # Select 3 distinct cases
    cases = [
        {"idx": 0, "name": "Case 1 (Patient 001)"},
        {"idx": 10, "name": "Case 2 (Patient 006)"},
        {"idx": 20, "name": "Case 3 (Patient 011)"}
    ]
    
    # Datasets
    prep_ds = Dataset(data=train_files, transform=get_vis_prep_transforms())
    aug_ds = Dataset(data=train_files, transform=get_vis_aug_transforms())
    
    for case_num, case in enumerate(cases):
        idx = case["idx"]
        name = case["name"]
        print(f"Generating visualizations for {name}...")
        
        # 1. Get Original Preprocessed
        orig_sample = prep_ds[idx]
        orig_img = orig_sample["image"].numpy()[0]
        orig_mask = orig_sample["label"].numpy()[0]
        orig_slice_idx = orig_img.shape[2] // 2
        
        # Slices to render
        orig_img_slice = np.rot90(orig_img[:, :, orig_slice_idx])
        orig_mask_slice = np.rot90(orig_mask[:, :, orig_slice_idx])
        
        # 2. Generate 4 different random augmentations of the SAME full-volume slice
        aug_slices = []
        for seed in range(4):
            # Seed the random generators to produce distinct, reproducible augmentations
            np.random.seed(42 + seed + idx)
            monai.utils.misc.set_determinism(seed=42 + seed + idx)
            
            aug_sample = aug_ds[idx]
            aug_img = aug_sample["image"].numpy()[0]
            aug_mask = aug_sample["label"].numpy()[0]
            
            # Use the same middle slice index
            slice_idx = aug_img.shape[2] // 2
            
            aug_slices.append({
                "img": np.rot90(aug_img[:, :, slice_idx]),
                "mask": np.rot90(aug_mask[:, :, slice_idx])
            })
            
        # Plot Layout:
        # Columns:
        # [ Col 0: Original ] [ Col 1: Aug 1 ] [ Col 2: Aug 2 ] [ Col 3: Aug 3 ] [ Col 4: Aug 4 ]
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 5, width_ratios=[1.2, 1, 1, 1, 1])
        
        # Render Original (Col 0)
        ax_orig_img = fig.add_subplot(gs[0, 0])
        ax_orig_img.imshow(orig_img_slice, cmap='gray', vmin=0.0, vmax=1.0)
        ax_orig_img.set_title(f"Original Volume\n({name}, Slice {orig_slice_idx})", fontsize=11, fontweight='bold')
        ax_orig_img.axis('off')
        
        ax_orig_overlay = fig.add_subplot(gs[1, 0])
        orig_rgba = np.zeros((*orig_mask_slice.shape, 4))
        orig_rgba[orig_mask_slice == 1] = [0.12, 0.47, 0.71, 0.6] # RV (Blue)
        orig_rgba[orig_mask_slice == 2] = [1.0, 0.5, 0.0, 0.6]    # MYO (Orange)
        orig_rgba[orig_mask_slice == 3] = [0.84, 0.15, 0.16, 0.6] # LV (Red)
        ax_orig_overlay.imshow(orig_img_slice, cmap='gray', vmin=0.0, vmax=1.0)
        ax_orig_overlay.imshow(orig_rgba)
        ax_orig_overlay.set_title("Original Mask Overlay", fontsize=11, fontweight='bold')
        ax_orig_overlay.axis('off')
        
        # Render 4 Augmented versions of the same volume/slice (Col 1 to Col 4)
        for i in range(4):
            a_slice = aug_slices[i]
            a_img_slice = a_slice["img"]
            a_mask_slice = a_slice["mask"]
            
            # Augmented Image (Row 0)
            ax_a_img = fig.add_subplot(gs[0, i + 1])
            ax_a_img.imshow(a_img_slice, cmap='gray', vmin=0.0, vmax=1.0)
            ax_a_img.set_title(f"Augmented Version {i+1}", fontsize=10, fontweight='semibold')
            ax_a_img.axis('off')
            
            # Augmented Overlay (Row 1)
            ax_a_overlay = fig.add_subplot(gs[1, i + 1])
            a_rgba = np.zeros((*a_mask_slice.shape, 4))
            a_rgba[a_mask_slice == 1] = [0.12, 0.47, 0.71, 0.6] # RV
            a_rgba[a_mask_slice == 2] = [1.0, 0.5, 0.0, 0.6]    # MYO
            a_rgba[a_mask_slice == 3] = [0.84, 0.15, 0.16, 0.6] # LV
            ax_a_overlay.imshow(a_img_slice, cmap='gray', vmin=0.0, vmax=1.0)
            ax_a_overlay.imshow(a_rgba)
            ax_a_overlay.set_title(f"Version {i+1} Overlay", fontsize=10)
            ax_a_overlay.axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(assets_dir, f"aug_case_{case_num + 1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
        
    print("="*60)
    print("All multi-case visualizations completed successfully.")
    print("="*60)

if __name__ == "__main__":
    generate_multi_case_visualizations()
