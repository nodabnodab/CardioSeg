import os
import zipfile
import nibabel as nib
import SimpleITK as sitk
import numpy as np

def extract_zip(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

def inspect_nifti(image_path, mask_path):
    print("\n" + "="*50)
    print("=== INSPECTING NIFTI METADATA ===")
    print("="*50)
    
    # 1. Inspect Image using Nibabel
    img_nib = nib.load(image_path)
    img_data = img_nib.get_fdata()
    img_header = img_nib.header
    
    print(f"[Nibabel] Image File: {os.path.basename(image_path)}")
    print(f"  - Shape: {img_nib.shape}")
    print(f"  - Data Type: {img_data.dtype}")
    print(f"  - Voxel Spacing: {img_header.get_zooms()}")
    print(f"  - Intensity Range: Min={img_data.min():.2f}, Max={img_data.max():.2f}, Mean={img_data.mean():.2f}")
    
    # 2. Inspect Mask using SimpleITK
    mask_sitk = sitk.ReadImage(mask_path)
    mask_data = sitk.GetArrayFromImage(mask_sitk)
    
    print(f"\n[SimpleITK] Mask File: {os.path.basename(mask_path)}")
    print(f"  - Shape: {mask_sitk.GetSize()} (X, Y, Z)")
    print(f"  - Voxel Spacing: {mask_sitk.GetSpacing()} (dx, dy, dz)")
    print(f"  - Origin: {mask_sitk.GetOrigin()}")
    print(f"  - Direction: {mask_sitk.GetDirection()}")
    print(f"  - Unique Label Values: {np.unique(mask_data)}")
    print("="*50)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    acdc_dir = os.path.join(base_dir, "data", "ACDC")
    
    images_zip = os.path.join(acdc_dir, "Images.zip")
    masks_zip = os.path.join(acdc_dir, "Masks.zip")
    
    # Extraction paths
    extracted_images_dir = os.path.join(acdc_dir)
    extracted_masks_dir = os.path.join(acdc_dir)
    
    sample_image = os.path.join(acdc_dir, "Images", "patient001_frame01.nii.gz")
    sample_mask = os.path.join(acdc_dir, "Masks", "patient001_frame01_gt.nii.gz")
    
    # Extract only if not already extracted
    if not os.path.exists(sample_image):
        extract_zip(images_zip, extracted_images_dir)
    else:
        print("Images already extracted.")
        
    if not os.path.exists(sample_mask):
        extract_zip(masks_zip, extracted_masks_dir)
    else:
        print("Masks already extracted.")
        
    # Inspect the sample files
    if os.path.exists(sample_image) and os.path.exists(sample_mask):
        inspect_nifti(sample_image, sample_mask)
    else:
        print("Error: Sample files not found after extraction.")
