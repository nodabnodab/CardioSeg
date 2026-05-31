import os
import re
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    CropForegroundd,
    ScaleIntensityRangePercentilesd
)

def get_acdc_splits(base_dir):
    """
    Scans data/ACDC/Images and data/ACDC/Masks, pairs them up,
    and splits them at the patient level into Train (80), Val (10), and Test (10).
    """
    acdc_dir = os.path.join(base_dir, "data", "ACDC")
    images_dir = os.path.join(acdc_dir, "Images")
    masks_dir = os.path.join(acdc_dir, "Masks")
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        raise FileNotFoundError(f"ACDC dataset directories not found in {acdc_dir}")
        
    # List and pair files
    image_files = os.listdir(images_dir)
    pairs = []
    
    for img_name in image_files:
        if not img_name.endswith(".nii.gz"):
            continue
        # Extract patient ID (e.g. 'patient001' from 'patient001_frame01.nii.gz')
        match = re.match(r"(patient\d+)_frame\d+", img_name)
        if not match:
            continue
        patient_id = match.group(1)
        patient_num = int(patient_id.replace("patient", ""))
        
        # Expected mask name (e.g. 'patient001_frame01_gt.nii.gz')
        mask_name = img_name.replace(".nii.gz", "_gt.nii.gz")
        mask_path = os.path.join(masks_dir, mask_name)
        
        if os.path.exists(mask_path):
            pairs.append({
                "patient_num": patient_num,
                "image": os.path.join(images_dir, img_name),
                "label": mask_path
            })
            
    # Sort pairs by patient number and frame to keep it deterministic
    pairs.sort(key=lambda x: (x["patient_num"], x["image"]))
    
    # Split by patient number:
    # Train: patients 1 to 80
    # Val: patients 81 to 90
    # Test: patients 91 to 100
    train_files = []
    val_files = []
    test_files = []
    
    for p in pairs:
        file_dict = {"image": p["image"], "label": p["label"]}
        p_num = p["patient_num"]
        
        if 1 <= p_num <= 80:
            train_files.append(file_dict)
        elif 81 <= p_num <= 90:
            val_files.append(file_dict)
        elif 91 <= p_num <= 100:
            test_files.append(file_dict)
            
    return train_files, val_files, test_files

def get_preprocessing_transforms():
    """
    Returns the standard preprocessing Compose pipeline with Spacing, Crop, and Contrast Windowing.
    Note: Data augmentation (random rotations/crops) will be added in the next step.
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
        # Medical Contrast Windowing: clips extreme 2% and 98% intensities, maps to [0.0, 1.0]
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=2.0,
            upper=98.0,
            b_min=0.0,
            b_max=1.0,
            clip=True
        )
    ])
