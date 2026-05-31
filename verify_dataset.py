import os
import matplotlib.pyplot as plt
import numpy as np
import monai
from monai.data import Dataset, DataLoader
from src.dataset import get_acdc_splits, get_preprocessing_transforms

def verify_dataset_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    print("="*60)
    print("RUNNING DATASET SPLIT & PREPROCESSING PIPELINE VERIFICATION")
    print("="*60)
    
    # 1. Check Split Sizes
    try:
        train_files, val_files, test_files = get_acdc_splits(base_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    print(f"Dataset split completed successfully:")
    print(f"  - Train Set      : {len(train_files)} files (Patients 1-80)")
    print(f"  - Validation Set : {len(val_files)} files (Patients 81-90)")
    print(f"  - Test Set       : {len(test_files)} files (Patients 91-100)")
    print("-" * 60)
    
    # 2. Get transforms
    preproc_transforms = get_preprocessing_transforms()
    
    # 3. Create Dataset
    train_ds = Dataset(data=train_files, transform=preproc_transforms)
    
    # Load first sample
    print("Loading and preprocessing first sample from Train Set...")
    sample = train_ds[0]
    
    # Get shapes and intensity details
    image_tensor = sample["image"]
    label_tensor = sample["label"]
    
    # Output PyTorch shapes: [Channel, Height, Width, Depth]
    print(f"Sample Image Shape : {image_tensor.shape}")
    print(f"Sample Label Shape : {label_tensor.shape}")
    print(f"Image Intensity    : Min={image_tensor.min():.4f}, Max={image_tensor.max():.4f}, Mean={image_tensor.mean():.4f}")
    print(f"Unique Label IDs   : {np.unique(label_tensor.numpy())}")
    print("-" * 60)
    
    # 4. Generate visual check image
    # Select middle slice in the Z direction
    prep_img = image_tensor.numpy()[0] # Shape [H, W, D]
    prep_mask = label_tensor.numpy()[0]
    
    slice_idx = prep_img.shape[2] // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot image
    axes[0].imshow(np.rot90(prep_img[:, :, slice_idx]), cmap='gray')
    axes[0].set_title(f"Preprocessed Image (Slice {slice_idx})")
    axes[0].axis('off')
    
    # Plot mask overlay
    mask_slice = np.rot90(prep_mask[:, :, slice_idx])
    rgba = np.zeros((*mask_slice.shape, 4))
    rgba[mask_slice == 1] = [0.12, 0.47, 0.71, 0.6] # RV (Blue)
    rgba[mask_slice == 2] = [1.0, 0.5, 0.0, 0.6]    # MYO (Orange)
    rgba[mask_slice == 3] = [0.84, 0.15, 0.16, 0.6] # LV (Red)
    
    axes[1].imshow(np.rot90(prep_img[:, :, slice_idx]), cmap='gray')
    axes[1].imshow(rgba)
    axes[1].set_title("Preprocessed Overlay")
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(assets_dir, "dataset_verification.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated verification image at:\n{save_path}")
    print("="*60)

if __name__ == "__main__":
    verify_dataset_pipeline()
