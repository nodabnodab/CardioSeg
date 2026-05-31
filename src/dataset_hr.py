import os
import re
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
    RandShiftIntensityd,
    RandCropByPosNegLabeld
)

def get_acdc_splits_hr(base_dir):
    """
    Same split configuration as standard dataset.py to maintain 
    consistent Train (80) / Val (10) / Test (10) split.
    """
    acdc_dir = os.path.join(base_dir, "data", "ACDC")
    images_dir = os.path.join(acdc_dir, "Images")
    masks_dir = os.path.join(acdc_dir, "Masks")
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        raise FileNotFoundError(f"ACDC dataset directories not found in {acdc_dir}")
        
    image_files = os.listdir(images_dir)
    pairs = []
    
    for img_name in image_files:
        if not img_name.endswith(".nii.gz"):
            continue
        match = re.match(r"(patient\d+)_frame\d+", img_name)
        if not match:
            continue
        patient_id = match.group(1)
        patient_num = int(patient_id.replace("patient", ""))
        
        mask_name = img_name.replace(".nii.gz", "_gt.nii.gz")
        mask_path = os.path.join(masks_dir, mask_name)
        
        if os.path.exists(mask_path):
            pairs.append({
                "patient_num": patient_num,
                "image": os.path.join(images_dir, img_name),
                "label": mask_path
            })
            
    pairs.sort(key=lambda x: (x["patient_num"], x["image"]))
    
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

def get_train_transforms_hr():
    """
    High-Resolution (HR) training pipeline:
    - Upscales physical voxel spacing to [1.0, 1.0, 2.5] mm (Z-axis resolution doubled).
    - Image uses bilinear (trilinear) interpolation, Label uses nearest neighbor.
    - Crops 3D patches of size [128, 128, 16] (Z-depth doubled to match high-resolution spacing).
    """
    return Compose([
        # 1. High-Resolution Upscaling Resampling
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=[1.0, 1.0, 2.5],             # High-Res Target (Original Z ~ 5.0 -> 2.5 mm upscaled)
            mode=("bilinear", "nearest")       # Bilinear (Trilinear) for MRI, Nearest for Integer Labels
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
        
        # 2. Data Augmentations (Only applied during training)
        RandRotated(
            keys=["image", "label"],
            range_x=0.0,
            range_y=0.0,
            range_z=0.26,                      # Z-axis rotation up to ~15 degrees
            prob=0.5,
            mode=("bilinear", "nearest")
        ),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.5,
            mode=("bilinear", "nearest")
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        
        # 3. High-Resolution Target-focused 3D Patch Cropping
        # Crops 4 patches of size 128x128x16 (doubled Z-depth).
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[128, 128, 16],        # Expanded Z-axis patch depth
            pos=1.0,
            neg=1.0,
            num_samples=4,
            image_key="image",
            image_threshold=0.0
        )
    ])

def get_val_test_transforms_hr():
    """
    High-Resolution (HR) Validation & Testing pipeline:
    - Only applies high-res resampling, contrast normalization, and foreground crop.
    - No random distortions or patch cropping.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=[1.0, 1.0, 2.5],             # High-Res Target
            mode=("bilinear", "nearest")       # Bilinear for MRI, Nearest for Integer Labels
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
