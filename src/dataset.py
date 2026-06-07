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

def get_train_transforms():
    """
    Training pipeline: Standard Preprocessing + Data Augmentations + Patch Cropping.
    Generates 4 random 128x128x8 patches centered near the heart targets.
    """
    return Compose([
        # 1. Core Preprocessing
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
        
        # 2. Data Augmentations (Only applied during training)
        # Random Rotation around Z-axis (up to ~15 degrees or 0.26 rad)
        RandRotated(
            keys=["image", "label"],
            range_x=0.0,
            range_y=0.0,
            range_z=0.26,
            prob=0.5,
            mode=("bilinear", "nearest")
        ),
        # Random Zoom (scale patient size slightly)
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.5,
            mode=("bilinear", "nearest")
        ),
        # Random Intensity changes (simulate scanner calibration variations)
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        
        # 3. Target-focused 3D Patch Cropping
        # Crops 4 patches of size 128x128x8.
        # Weighs cropping centers: 1:1 ratio between heart classes (pos) and background (neg).
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[128, 128, 8],
            pos=1.0,
            neg=1.0,
            num_samples=4,
            image_key="image",
            image_threshold=0.0
        )
    ])

def get_val_test_transforms():
    """
    Validation & Testing pipeline: Core Preprocessing ONLY.
    No rot/zoom distortions, no patch cropping. Loads full volume.
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
