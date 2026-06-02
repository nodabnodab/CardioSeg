import os
import re
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Spacing

from src.dataset import get_acdc_splits, get_val_test_transforms
from src.model import get_3d_unet
from src.model_hr import get_3d_unet_hr

def classify_lvef(ef):
    if ef >= 50.0:
        return "Normal"
    elif 40.0 <= ef < 50.0:
        return "Borderline"
    else:
        return "Heart Failure"

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load splits
    train_files, val_files, test_files = get_acdc_splits(base_dir)
    eval_files = val_files + test_files # 20 patients, 40 volumes total
    
    # Group by patient
    patients = {}
    for f in eval_files:
        img_path = f["image"]
        filename = os.path.basename(img_path)
        pid = re.match(r"(patient\d+)", filename).group(1)
        if pid not in patients:
            patients[pid] = []
        patients[pid].append(f)
        
    # Sort frames for each patient
    for pid in patients:
        patients[pid].sort(key=lambda x: x["image"])
        
    # 2. Load Models
    print("Loading models...")
    model_lr = get_3d_unet(1, 4).to(device)
    model_lr.load_state_dict(torch.load("best_metric_model.pth", map_location=device))
    model_lr.eval()
    
    model_hr = get_3d_unet_hr(1, 4).to(device)
    model_hr.load_state_dict(torch.load("best_metric_model_hr.pth", map_location=device))
    model_hr.eval()
    
    # Validation transforms
    val_transforms = get_val_test_transforms()
    
    # Spacing resamplers
    resampler_to_hr = Spacing(pixdim=[1.0, 1.0, 2.5], mode="bilinear")
    resampler_to_lr = Spacing(pixdim=[1.25, 1.25, 5.0], mode="nearest")
    
    # CCA post-processing
    from monai.transforms import KeepLargestConnectedComponent, Compose
    cca_post_lr = Compose([
        AsDiscrete(to_onehot=4),
        KeepLargestConnectedComponent(applied_labels=[1, 2, 3], is_onehot=True)
    ])
    cca_post_std = Compose([
        AsDiscrete(argmax=True, to_onehot=4),
        KeepLargestConnectedComponent(applied_labels=[1, 2, 3], is_onehot=True)
    ])
    
    # Result logging
    gt_efs = []
    lr_efs = []
    hr_efs = []
    hr_cca_efs = []
    
    gt_diagnoses = []
    lr_diagnoses = []
    hr_diagnoses = []
    hr_cca_diagnoses = []
    
    print("\nStarting Clinical Metric Evaluation...")
    with torch.no_grad():
        for pid in tqdm(sorted(patients.keys()), desc="Processing Patients"):
            files = patients[pid]
            patient_results = {
                "gt_vols": [],
                "lr_vols": [],
                "hr_vols": [],
                "hr_cca_vols": []
            }
            
            for f in files:
                img_path = f["image"]
                lbl_path = f["label"]
                
                # Load metadata
                nib_img = nib.load(img_path)
                spacing = nib_img.header.get_zooms()
                voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
                
                # Preprocess
                data = val_transforms({"image": img_path, "label": lbl_path})
                inputs = data["image"].unsqueeze(0).to(device)
                labels = data["label"].unsqueeze(0).to(device)
                
                # A. GT Volume
                gt_mask = data["label"][0].numpy()
                gt_vol = np.sum(gt_mask == 3) * voxel_vol_ml
                patient_results["gt_vols"].append(gt_vol)
                
                # B. Baseline Prediction
                outputs_lr = sliding_window_inference(inputs, (128, 128, 8), 4, model_lr, overlap=0.5)
                pred_lr_argmax = torch.argmax(outputs_lr[0], dim=0).cpu().numpy()
                lr_vol = np.sum(pred_lr_argmax == 3) * voxel_vol_ml
                patient_results["lr_vols"].append(lr_vol)
                
                # C. HR Prediction
                inputs_hr = resampler_to_hr(data["image"]).unsqueeze(0).to(device)
                outputs_hr = sliding_window_inference(inputs_hr, (128, 128, 16), 4, model_hr, overlap=0.5)
                outputs_argmax_cpu = torch.argmax(outputs_hr, dim=1, keepdim=True).cpu()
                outputs_lr_argmax = resampler_to_lr(outputs_argmax_cpu[0], output_spatial_shape=labels.shape[2:]).unsqueeze(0).to(device)
                
                # HR Standard
                pred_hr_argmax = outputs_lr_argmax[0, 0].cpu().numpy()
                hr_vol = np.sum(pred_hr_argmax == 3) * voxel_vol_ml
                patient_results["hr_vols"].append(hr_vol)
                
                # HR CCA
                outputs_processed = cca_post_lr(outputs_lr_argmax[0])
                pred_cca_argmax = torch.argmax(outputs_processed, dim=0).cpu().numpy()
                hr_cca_vol = np.sum(pred_cca_argmax == 3) * voxel_vol_ml
                patient_results["hr_cca_vols"].append(hr_cca_vol)
                
            # Compute EFs
            def get_ef(vols):
                edv = max(vols[0], vols[1])
                esv = min(vols[0], vols[1])
                return ((edv - esv) / edv) * 100 if edv > 0 else 0
                
            gt_ef = get_ef(patient_results["gt_vols"])
            lr_ef = get_ef(patient_results["lr_vols"])
            hr_ef = get_ef(patient_results["hr_vols"])
            hr_cca_ef = get_ef(patient_results["hr_cca_vols"])
            
            gt_efs.append(gt_ef)
            lr_efs.append(lr_ef)
            hr_efs.append(hr_ef)
            hr_cca_efs.append(hr_cca_ef)
            
            gt_diag = classify_lvef(gt_ef)
            lr_diag = classify_lvef(lr_ef)
            hr_diag = classify_lvef(hr_ef)
            hr_cca_diag = classify_lvef(hr_cca_ef)
            
            gt_diagnoses.append(gt_diag)
            lr_diagnoses.append(lr_diag)
            hr_diagnoses.append(hr_diag)
            hr_cca_diagnoses.append(hr_cca_diag)
            
    # Calculate performance metrics
    gt_efs = np.array(gt_efs)
    lr_efs = np.array(lr_efs)
    hr_efs = np.array(hr_efs)
    hr_cca_efs = np.array(hr_cca_efs)
    
    lr_mae = np.mean(np.abs(gt_efs - lr_efs))
    hr_mae = np.mean(np.abs(gt_efs - hr_efs))
    hr_cca_mae = np.mean(np.abs(gt_efs - hr_cca_efs))
    
    def get_accuracy(diag_list):
        matches = [1 if d == gt_diagnoses[idx] else 0 for idx, d in enumerate(diag_list)]
        return (sum(matches) / len(matches)) * 100
        
    lr_acc = get_accuracy(lr_diagnoses)
    hr_acc = get_accuracy(hr_diagnoses)
    hr_cca_acc = get_accuracy(hr_cca_diagnoses)
    
    print("\n" + "="*50)
    print("CLINICAL IMPACT VALIDATION REPORT")
    print("="*50)
    print(f"Total Patients evaluated: {len(gt_efs)}")
    print(f"Ground Truth Diagnoses distribution: Normal={gt_diagnoses.count('Normal')}, Borderline={gt_diagnoses.count('Borderline')}, Heart Failure={gt_diagnoses.count('Heart Failure')}")
    print("-"*50)
    print(f"{'Model Configuration':<30} | {'LVEF MAE (오차)':<15} | {'Diagnostic Acc (진단 일치율)':<25}")
    print("-"*75)
    print(f"{'Baseline Model (Low-Res)':<30} | {lr_mae:<13.2f}% | {lr_acc:<23.1f}%")
    print(f"{'HR Model (Standard)':<30} | {hr_mae:<13.2f}% | {hr_acc:<23.1f}%")
    print(f"{'HR Model + CCA Filter':<30} | {hr_cca_mae:<13.2f}% | {hr_cca_acc:<23.1f}%")
    print("="*75)
    
    # Detailed patient table
    print("\n[Detailed Patient Diagnosis Log]")
    print(f"{'Patient':<12} | {'GT EF (Diag)':<22} | {'LR EF (Diag)':<22} | {'HR+CCA EF (Diag)':<22}")
    print("-"*85)
    for idx, pid in enumerate(sorted(patients.keys())):
        gt_str = f"{gt_efs[idx]:.1f}% ({gt_diagnoses[idx][0]})"
        lr_str = f"{lr_efs[idx]:.1f}% ({lr_diagnoses[idx][0]})"
        hr_cca_str = f"{hr_cca_efs[idx]:.1f}% ({hr_cca_diagnoses[idx][0]})"
        print(f"{pid:<12} | {gt_str:<22} | {lr_str:<22} | {hr_cca_str:<22}")

if __name__ == "__main__":
    main()
