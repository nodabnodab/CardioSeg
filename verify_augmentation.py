import os
import matplotlib.pyplot as plt
import numpy as np
import monai
from monai.data import Dataset, DataLoader, list_data_collate
from src.dataset import get_acdc_splits, get_train_transforms

def verify_augmentation_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    print("="*60)
    print("RUNNING DATA AUGMENTATION & 3D PATCH CROP VERIFICATION")
    print("="*60)
    
    # 1. Get split paths
    train_files, _, _ = get_acdc_splits(base_dir)
    
    # 2. Get transforms (preprocessing + augmentations + pos/neg patch cropping)
    train_transforms = get_train_transforms()
    
    # 3. Create MONAI Dataset
    train_ds = Dataset(data=train_files, transform=train_transforms)
    
    # 4. Fetch the first patient's data
    print("Loading first patient sample and applying real-time augmentations...")
    # Because of RandCropByPosNegLabeld (num_samples=4), train_ds[0] returns a list of 4 dictionaries
    sample_patches = train_ds[0]
    
    print(f"Number of generated patches for one patient: {len(sample_patches)}")
    print("-" * 60)
    
    # Let's inspect each patch
    for idx, patch in enumerate(sample_patches):
        img_shape = patch["image"].shape
        lbl_shape = patch["label"].shape
        img_min = patch["image"].min()
        img_max = patch["image"].max()
        unique_labels = np.unique(patch["label"].numpy())
        print(f"Patch {idx + 1}:")
        print(f"  - Image Shape : {img_shape} (expected: [1, 128, 128, 8])")
        print(f"  - Label Shape : {lbl_shape}")
        print(f"  - Intensity   : Min={img_min:.4f}, Max={img_max:.4f}")
        print(f"  - Unique IDs  : {unique_labels} (0: BG, 1: RV, 2: MYO, 3: LV)")
        print("-" * 60)
        
    # 5. Create DataLoader batch test
    # We use list_data_collate which is MONAI's standard collate to handle lists of dicts
    train_loader = DataLoader(
        train_ds, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=list_data_collate
    )
    
    print("Fetching first DataLoader batch...")
    batch = next(iter(train_loader))
    # Batch size is 2, but each patient yields 4 patches, so batch size in PyTorch becomes 8!
    print(f"DataLoader Batch Image Shape: {batch['image'].shape} (expected: [8, 1, 128, 128, 8])")
    print(f"DataLoader Batch Label Shape: {batch['label'].shape}")
    print("-" * 60)
    
    # 6. Plot the 4 patches for visualization
    # We take the middle slice (slice idx 4 out of 8) in the Z direction for each patch
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        patch_img = sample_patches[i]["image"].numpy()[0] # Shape [128, 128, 8]
        patch_mask = sample_patches[i]["label"].numpy()[0]
        
        slice_idx = patch_img.shape[2] // 2
        
        img_slice = np.rot90(patch_img[:, :, slice_idx])
        mask_slice = np.rot90(patch_mask[:, :, slice_idx])
        
        # Row 1: Preprocessed & Augmented Image slice
        axes[0, i].imshow(img_slice, cmap='gray', vmin=0.0, vmax=1.0)
        axes[0, i].set_title(f"Patch {i+1} Image (Slice {slice_idx})")
        axes[0, i].axis('off')
        
        # Row 2: Mask Overlay on the same slice
        rgba = np.zeros((*mask_slice.shape, 4))
        rgba[mask_slice == 1] = [0.12, 0.47, 0.71, 0.6] # RV (Blue)
        rgba[mask_slice == 2] = [1.0, 0.5, 0.0, 0.6]    # MYO (Orange)
        rgba[mask_slice == 3] = [0.84, 0.15, 0.16, 0.6] # LV (Red)
        
        axes[1, i].imshow(img_slice, cmap='gray', vmin=0.0, vmax=1.0)
        axes[1, i].imshow(rgba)
        axes[1, i].set_title(f"Patch {i+1} Mask Overlay")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(assets_dir, "augmentation_verification.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved augmentation verification image at:\n{save_path}")
    print("="*60)

if __name__ == "__main__":
    verify_augmentation_pipeline()
