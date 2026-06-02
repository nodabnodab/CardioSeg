import os
import torch
import numpy as np
from tqdm import tqdm
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent, Compose, Spacing

from src.dataset import get_acdc_splits, get_val_test_transforms
from src.model import get_3d_unet
from src.model_hr import get_3d_unet_hr

def evaluate_model(model, loader, device, roi_size, is_hr=False, use_cca=False):
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    
    post_label = AsDiscrete(to_onehot=4)
    
    if use_cca:
        post_pred_lr = Compose([
            AsDiscrete(to_onehot=4),
            KeepLargestConnectedComponent(applied_labels=[1, 2, 3], is_onehot=True)
        ])
        post_pred_std = Compose([
            AsDiscrete(argmax=True, to_onehot=4),
            KeepLargestConnectedComponent(applied_labels=[1, 2, 3], is_onehot=True)
        ])
    else:
        post_pred_lr = AsDiscrete(to_onehot=4)
        post_pred_std = AsDiscrete(argmax=True, to_onehot=4)
        
    model.eval()
    
    # Setup spacing resamplers for HR model if needed
    if is_hr:
        resampler_to_hr = Spacing(pixdim=[1.0, 1.0, 2.5], mode="bilinear")
        resampler_to_lr = Spacing(pixdim=[1.25, 1.25, 5.0], mode="nearest")
        
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = data["image"].to(device), data["label"].to(device)
            
            if is_hr:
                # Resample image to HR on CPU, then send to device
                inputs_hr = resampler_to_hr(data["image"][0]).unsqueeze(0).to(device)
                
                outputs_hr = sliding_window_inference(
                    inputs_hr, 
                    roi_size, 
                    sw_batch_size=4, 
                    predictor=model,
                    overlap=0.5
                )
                
                outputs_argmax_cpu = torch.argmax(outputs_hr, dim=1, keepdim=True).cpu()
                outputs_lr_argmax = resampler_to_lr(outputs_argmax_cpu[0], output_spatial_shape=labels.shape[2:]).unsqueeze(0).to(device)
                
                outputs_processed = [post_pred_lr(x) for x in decollate_batch(outputs_lr_argmax)]
            else:
                outputs = sliding_window_inference(
                    inputs, 
                    roi_size, 
                    sw_batch_size=4, 
                    predictor=model,
                    overlap=0.5
                )
                outputs_processed = [post_pred_std(x) for x in decollate_batch(outputs)]
                
            labels_processed = [post_label(x) for x in decollate_batch(labels)]
            dice_metric(y_pred=outputs_processed, y=labels_processed)
            
    metric_batch = dice_metric.aggregate()
    dice_metric.reset()
    
    return {
        "mean": metric_batch.mean().item(),
        "rv": metric_batch[0].item(),
        "myo": metric_batch[1].item(),
        "lv": metric_batch[2].item()
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get datasets
    train_files, val_files, test_files = get_acdc_splits(base_dir)
    val_transforms = get_val_test_transforms()
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    test_ds = Dataset(data=test_files, transform=val_transforms)
    
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)
    
    # 1. Load Baseline Model
    print("Loading Baseline model...")
    model_lr = get_3d_unet(1, 4).to(device)
    lr_weights = os.path.join(base_dir, "best_metric_model.pth")
    if os.path.exists(lr_weights):
        model_lr.load_state_dict(torch.load(lr_weights, map_location=device))
        print("Baseline weights loaded successfully.")
    else:
        print("Warning: best_metric_model.pth not found!")
        return
        
    # 2. Load HR Model
    print("Loading High-Resolution model...")
    model_hr = get_3d_unet_hr(1, 4).to(device)
    hr_weights = os.path.join(base_dir, "best_metric_model_hr.pth")
    if os.path.exists(hr_weights):
        model_hr.load_state_dict(torch.load(hr_weights, map_location=device))
        print("HR weights loaded successfully.")
    else:
        print("Warning: best_metric_model_hr.pth not found!")
        return
        
    # Evaluate Baseline
    print("Evaluating Baseline standard...")
    base_val_std = evaluate_model(model_lr, val_loader, device, (128, 128, 8), is_hr=False, use_cca=False)
    base_test_std = evaluate_model(model_lr, test_loader, device, (128, 128, 8), is_hr=False, use_cca=False)
    
    print("Evaluating Baseline with CCA...")
    base_val_cca = evaluate_model(model_lr, val_loader, device, (128, 128, 8), is_hr=False, use_cca=True)
    base_test_cca = evaluate_model(model_lr, test_loader, device, (128, 128, 8), is_hr=False, use_cca=True)
    
    # Evaluate HR
    print("Evaluating HR standard...")
    hr_val_std = evaluate_model(model_hr, val_loader, device, (128, 128, 16), is_hr=True, use_cca=False)
    hr_test_std = evaluate_model(model_hr, test_loader, device, (128, 128, 16), is_hr=True, use_cca=False)
    
    print("Evaluating HR with CCA...")
    hr_val_cca = evaluate_model(model_hr, val_loader, device, (128, 128, 16), is_hr=True, use_cca=True)
    hr_test_cca = evaluate_model(model_hr, test_loader, device, (128, 128, 16), is_hr=True, use_cca=True)
    
    print("\n" + "="*50)
    print("RESULTS COMPARISON TABLE")
    print("="*50)
    
    def print_table(title, std_metrics, cca_metrics):
        print(f"\n[{title}]")
        print(f"{'Class':<10} | {'Standard':<10} | {'With CCA':<10} | {'Delta':<10}")
        print("-"*45)
        for name in ["mean", "rv", "myo", "lv"]:
            diff = cca_metrics[name] - std_metrics[name]
            print(f"{name.upper():<10} | {std_metrics[name]:.4f}     | {cca_metrics[name]:.4f}     | {diff:+.4f}")
            
    print_table("Baseline Model (Low-Res) - Validation Set", base_val_std, base_val_cca)
    print_table("Baseline Model (Low-Res) - Test Set", base_test_std, base_test_cca)
    print_table("High-Resolution Model - Validation Set", hr_val_std, hr_val_cca)
    print_table("High-Resolution Model - Test Set", hr_test_std, hr_test_cca)

if __name__ == "__main__":
    main()
