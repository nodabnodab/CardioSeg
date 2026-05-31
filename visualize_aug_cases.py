import os
import matplotlib.pyplot as plt
import numpy as np
import monai
from monai.data import Dataset
from src.dataset import get_acdc_splits, get_train_transforms, get_val_test_transforms

def generate_multi_case_visualizations():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    print("="*60)
    print("GENERATING MULTI-CASE PREPROCESSING & AUGMENTATION COMPARISONS")
    print("="*60)
    
    # 1. Get split files
    train_files, _, _ = get_acdc_splits(base_dir)
    
    # Select 3 distinct cases (e.g. index 0, index 10, index 20)
    cases = [
        {"idx": 0, "name": "Case 1 (Patient 001)"},
        {"idx": 10, "name": "Case 2 (Patient 006)"},
        {"idx": 20, "name": "Case 3 (Patient 011)"}
    ]
    
    # Get transforms
    val_transforms = get_val_test_transforms() # Preprocessing only (Full Volume)
    train_transforms = get_train_transforms()   # Preprocessing + Augmentation + Crop
    
    # Create datasets
    val_ds = Dataset(data=train_files, transform=val_transforms)
    train_ds = Dataset(data=train_files, transform=train_transforms)
    
    for case_num, case in enumerate(cases):
        idx = case["idx"]
        name = case["name"]
        print(f"Generating visualizations for {name}...")
        
        # 1. Load Original Preprocessed (Full Volume)
        orig_sample = val_ds[idx]
        orig_img = orig_sample["image"].numpy()[0]
        orig_mask = orig_sample["label"].numpy()[0]
        orig_slice_idx = orig_img.shape[2] // 2
        
        # 2. Load 4 Augmented Patches (Cropped & Distorted)
        # We temporarily set deterministic seed for augmentation of this sample to keep layout clean
        np.random.seed(42 + idx)
        monai.utils.misc.set_determinism(seed=42 + idx)
        
        patches = train_ds[idx]
        
        # Plot Layout:
        # Columns:
        # [ Col 1: Original Image & Overlay ] [ Col 2: Patch 1 ] [ Col 3: Patch 2 ] [ Col 4: Patch 3 ] [ Col 5: Patch 4 ]
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 5, width_ratios=[1.8, 1, 1, 1, 1])
        
        # Render Original (Col 0)
        ax_orig_img = fig.add_subplot(gs[0, 0])
        orig_img_slice = np.rot90(orig_img[:, :, orig_slice_idx])
        ax_orig_img.imshow(orig_img_slice, cmap='gray', vmin=0.0, vmax=1.0)
        ax_orig_img.set_title(f"Original Volume\n({name}, Slice {orig_slice_idx})", fontsize=11, fontweight='bold')
        ax_orig_img.axis('off')
        
        ax_orig_overlay = fig.add_subplot(gs[1, 0])
        orig_mask_slice = np.rot90(orig_mask[:, :, orig_slice_idx])
        orig_rgba = np.zeros((*orig_mask_slice.shape, 4))
        orig_rgba[orig_mask_slice == 1] = [0.12, 0.47, 0.71, 0.6] # RV (Blue)
        orig_rgba[orig_mask_slice == 2] = [1.0, 0.5, 0.0, 0.6]    # MYO (Orange)
        orig_rgba[orig_mask_slice == 3] = [0.84, 0.15, 0.16, 0.6] # LV (Red)
        ax_orig_overlay.imshow(orig_img_slice, cmap='gray', vmin=0.0, vmax=1.0)
        ax_orig_overlay.imshow(orig_rgba)
        ax_orig_overlay.set_title("Original Mask Overlay", fontsize=11, fontweight='bold')
        ax_orig_overlay.axis('off')
        
        # Render 4 Patches (Col 1 to Col 4)
        for p_idx in range(4):
            patch = patches[p_idx]
            p_img = patch["image"].numpy()[0]
            p_mask = patch["label"].numpy()[0]
            p_slice_idx = p_img.shape[2] // 2
            
            p_img_slice = np.rot90(p_img[:, :, p_slice_idx])
            p_mask_slice = np.rot90(p_mask[:, :, p_slice_idx])
            
            # Patch Image (Row 0)
            ax_p_img = fig.add_subplot(gs[0, p_idx + 1])
            ax_p_img.imshow(p_img_slice, cmap='gray', vmin=0.0, vmax=1.0)
            ax_p_img.set_title(f"Augmented Patch {p_idx+1}\n(Slice {p_slice_idx})", fontsize=10)
            ax_p_img.axis('off')
            
            # Patch Overlay (Row 1)
            ax_p_overlay = fig.add_subplot(gs[1, p_idx + 1])
            p_rgba = np.zeros((*p_mask_slice.shape, 4))
            p_rgba[p_mask_slice == 1] = [0.12, 0.47, 0.71, 0.6] # RV
            p_rgba[p_mask_slice == 2] = [1.0, 0.5, 0.0, 0.6]    # MYO
            p_rgba[p_mask_slice == 3] = [0.84, 0.15, 0.16, 0.6] # LV
            ax_p_overlay.imshow(p_img_slice, cmap='gray', vmin=0.0, vmax=1.0)
            ax_p_overlay.imshow(p_rgba)
            ax_p_overlay.set_title(f"Patch {p_idx+1} Overlay", fontsize=10)
            ax_p_overlay.axis('off')
            
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
