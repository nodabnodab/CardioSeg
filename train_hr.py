import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import monai
from monai.data import Dataset, DataLoader, list_data_collate, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Spacing, KeepLargestConnectedComponent, Compose
from src.dataset import get_val_test_transforms
from src.dataset_hr import get_acdc_splits_hr, get_train_transforms_hr
from src.model_hr import get_3d_unet_hr

def save_training_plot_hr(history, save_path):
    """
    Generates and saves the High-Resolution training progress plot.
    """
    epochs = history["epoch"]
    if not epochs:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Plot Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#ff4b4b", linewidth=2.5)
    ax1.set_title("HR Training Loss Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss (Dice + CrossEntropy)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)
    
    # 2. Plot Dice Scores
    val_epochs = [e for i, e in enumerate(epochs) if i < len(history["val_mean_dice"]) and history["val_mean_dice"][i] > 0]
    val_indices = [i for i, e in enumerate(epochs) if i < len(history["val_mean_dice"]) and history["val_mean_dice"][i] > 0]
    
    if val_epochs:
        val_mean = [history["val_mean_dice"][i] for i in val_indices]
        test_mean = [history["test_mean_dice"][i] for i in val_indices] if "test_mean_dice" in history else []
        
        # Plot Means (Thick solid lines)
        ax2.plot(val_epochs, val_mean, label="Val Mean Dice", color="#2b6cb0", linewidth=3.0)
        if test_mean:
            ax2.plot(val_epochs, test_mean, label="Test Mean Dice", color="#2f855a", linewidth=3.0) # Green for test
            
        # Plot Per-Class Val Dices (Thin dashed lines)
        rv_dice = [history["val_rv_dice"][i] for i in val_indices]
        myo_dice = [history["val_myo_dice"][i] for i in val_indices]
        lv_dice = [history["val_lv_dice"][i] for i in val_indices]
        ax2.plot(val_epochs, rv_dice, label="Val RV Dice (우심실)", color="#3182ce", linestyle="--", alpha=0.5)
        ax2.plot(val_epochs, myo_dice, label="Val MYO Dice (심근)", color="#dd6b20", linestyle="--", alpha=0.5)
        ax2.plot(val_epochs, lv_dice, label="Val LV Dice (좌심실)", color="#e53e3e", linestyle="--", alpha=0.5)
        ax2.legend(fontsize=10, loc="lower right")
        
    ax2.set_title("HR Validation & Test Accuracy (Dice Score)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Dice Score (0.0 ~ 1.0)", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
 
def train_pipeline_hr():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Backup previous run's files if they exist to prevent overwrite (Windows safe rename)
    best_weight_path = os.path.join(base_dir, "best_metric_model_hr.pth")
    latest_checkpoint_path = os.path.join(base_dir, "latest_checkpoint_hr.pth")
    history_file_init = os.path.join(assets_dir, "training_history_hr.json")
    plot_file_init = os.path.join(assets_dir, "training_curves_hr.png")
    
    if os.path.exists(best_weight_path):
        backup_path = os.path.join(base_dir, "best_metric_model_hr_backup.pth")
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(best_weight_path, backup_path)
        print("[BACKUP] Existing best HR model weights backed up to best_metric_model_hr_backup.pth")
        
    if os.path.exists(latest_checkpoint_path):
        backup_path = os.path.join(base_dir, "latest_checkpoint_hr_backup.pth")
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(latest_checkpoint_path, backup_path)
        print("[BACKUP] Existing latest HR checkpoint backed up to latest_checkpoint_hr_backup.pth")
        
    if os.path.exists(history_file_init):
        backup_path = os.path.join(assets_dir, "training_history_hr_backup.json")
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(history_file_init, backup_path)
        print("[BACKUP] Existing HR history JSON backed up to training_history_hr_backup.json")
        
    if os.path.exists(plot_file_init):
        backup_path = os.path.join(assets_dir, "training_curves_hr_backup.png")
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(plot_file_init, backup_path)
        print("[BACKUP] Existing HR training curve plot backed up to training_curves_hr_backup.png")

    # 1. Hyperparameters
    max_epochs = 150
    val_interval = 2
    batch_size = 2        # B=2 loading 4 crops of size 128x128x16 = 8 patches per step
    learning_rate = 3e-4
    weight_decay = 1e-5
    roi_size = (128, 128, 16) # HR patch size
    early_stopping_patience = 12
    
    # 2. Get splits & datasets
    print("Loading HR datasets and setting up loaders...")
    train_files, val_files, test_files = get_acdc_splits_hr(base_dir)
    
    train_transforms = get_train_transforms_hr()
    val_transforms = get_val_test_transforms()
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    test_ds = Dataset(data=test_files, transform=val_transforms)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        collate_fn=list_data_collate,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        num_workers=0, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=1, 
        num_workers=0, 
        pin_memory=True
    )
    
    # 3. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_3d_unet_hr(in_channels=1, out_channels=4).to(device)
    print(f"Using device: {device}")
    
    # 4. Optimizer & Loss Setup
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    # AMP GradScaler for FP16 training
    scaler = torch.cuda.amp.GradScaler()
    
    # Dice metrics setup
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    
    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)
    
    # Tracking metrics
    history = {
        "epoch": [],
        "train_loss": [],
        "val_mean_dice": [],
        "val_rv_dice": [],
        "val_myo_dice": [],
        "val_lv_dice": [],
        "test_mean_dice": [],
        "test_rv_dice": [],
        "test_myo_dice": [],
        "test_lv_dice": []
    }
    history_file = os.path.join(assets_dir, "training_history_hr.json")
    plot_file = os.path.join(assets_dir, "training_curves_hr.png")
    
    best_metric = -1
    best_metric_epoch = -1
    epochs_no_improve = 0  # Patience counter for early stopping
    
    print("="*60)
    print("STARTING CardioSeg3D HIGH-RESOLUTION TRAINING RUN")
    print(f"Train Cases: {len(train_files)} | Val Cases: {len(val_files)}")
    print(f"Max Epochs: {max_epochs} | Val Interval: {val_interval}")
    print(f"Early Stopping Patience: {early_stopping_patience} evaluations ({early_stopping_patience * val_interval} epochs)")
    print("="*60)
    
    try:
        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_loss = 0
            step = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} (HR)")
            for batch_data in progress_bar:
                step += 1
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass with AMP autocast
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    
                # Backward pass with AMP scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            lr_scheduler.step()
            epoch_loss /= step
            
            history["epoch"].append(epoch)
            history["train_loss"].append(epoch_loss)
            print(f"Epoch {epoch} finished - Average Loss: {epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Evaluation Phase
            if epoch % val_interval == 0:
                model.eval()
                
                # Setup resamplers for original space evaluation
                resampler_to_hr = Spacing(pixdim=[1.0, 1.0, 2.5], mode="bilinear")
                resampler_to_lr = Spacing(pixdim=[1.25, 1.25, 5.0], mode="nearest")
                post_pred_lr = Compose([
                    AsDiscrete(to_onehot=4),
                    KeepLargestConnectedComponent(applied_labels=[1, 2, 3], is_onehot=True)
                ])
                
                with torch.no_grad():
                    # 1. Validation set evaluation
                    for val_data in val_loader:
                        val_labels = val_data["label"].to(device)
                        
                        # Resample input image to HR [1.0, 1.0, 2.5] for model inference (on CPU, then move to GPU)
                        val_inputs_hr = resampler_to_hr(val_data["image"][0]).unsqueeze(0).to(device)
                        
                        # Sliding window evaluation (Full 3D Volume)
                        val_outputs_hr = sliding_window_inference(
                            val_inputs_hr, 
                            roi_size, 
                            sw_batch_size=4, 
                            predictor=model,
                            overlap=0.5
                        )
                        
                        # Post-process HR outputs to argmax [1, 1, H_hr, W_hr, D_hr] and resample to CPU
                        val_outputs_argmax_cpu = torch.argmax(val_outputs_hr, dim=1, keepdim=True).cpu()
                        
                        # Resample predicted argmax back to baseline LR [1.25, 1.25, 5.0]
                        val_outputs_lr_argmax = resampler_to_lr(val_outputs_argmax_cpu[0], output_spatial_shape=val_labels.shape[2:]).unsqueeze(0).to(device)
                        
                        # Decollate batch and apply one-hot post-processing
                        val_outputs_lr_onehot = [post_pred_lr(x) for x in decollate_batch(val_outputs_lr_argmax)]
                        val_labels_onehot = [post_label(x) for x in decollate_batch(val_labels)]
                        
                        dice_metric(y_pred=val_outputs_lr_onehot, y=val_labels_onehot)
                    
                    # Aggregate metrics
                    metric_batch = dice_metric.aggregate()
                    dice_metric.reset()
                    
                    # Extract per-class dice scores
                    rv_dice = metric_batch[0].item()
                    myo_dice = metric_batch[1].item()
                    lv_dice = metric_batch[2].item()
                    mean_dice = metric_batch.mean().item()
                    
                    history["val_mean_dice"].append(mean_dice)
                    history["val_rv_dice"].append(rv_dice)
                    history["val_myo_dice"].append(myo_dice)
                    history["val_lv_dice"].append(lv_dice)
                    
                    print(f"--- [Validation Epoch {epoch} (HR)] ---")
                    print(f"  RV Dice (우심실):  {rv_dice:.4f}")
                    print(f"  MYO Dice (심근):   {myo_dice:.4f}")
                    print(f"  LV Dice (좌심실):  {lv_dice:.4f}")
                    print(f"  Mean Dice (평균):  {mean_dice:.4f}")
                    print("-" * 32)
                    
                    # 2. Test set evaluation (Purely for tracking and visualization)
                    for test_data in test_loader:
                        test_labels = test_data["label"].to(device)
                        
                        # Resample test image to HR for model inference
                        test_inputs_hr = resampler_to_hr(test_data["image"][0]).unsqueeze(0).to(device)
                        
                        test_outputs_hr = sliding_window_inference(
                            test_inputs_hr, 
                            roi_size, 
                            sw_batch_size=4, 
                            predictor=model,
                            overlap=0.5
                        )
                        
                        test_outputs_argmax_cpu = torch.argmax(test_outputs_hr, dim=1, keepdim=True).cpu()
                        test_outputs_lr_argmax = resampler_to_lr(test_outputs_argmax_cpu[0], output_spatial_shape=test_labels.shape[2:]).unsqueeze(0).to(device)
                        
                        test_outputs_lr_onehot = [post_pred_lr(x) for x in decollate_batch(test_outputs_lr_argmax)]
                        test_labels_onehot = [post_label(x) for x in decollate_batch(test_labels)]
                        
                        dice_metric(y_pred=test_outputs_lr_onehot, y=test_labels_onehot)
                        
                    metric_batch_test = dice_metric.aggregate()
                    dice_metric.reset()
                    
                    test_rv_dice = metric_batch_test[0].item()
                    test_myo_dice = metric_batch_test[1].item()
                    test_lv_dice = metric_batch_test[2].item()
                    test_mean_dice = metric_batch_test.mean().item()
                    
                    history["test_mean_dice"].append(test_mean_dice)
                    history["test_rv_dice"].append(test_rv_dice)
                    history["test_myo_dice"].append(test_myo_dice)
                    history["test_lv_dice"].append(test_lv_dice)
                    
                    print(f"--- [Testing Epoch {epoch} (HR)] ---")
                    print(f"  RV Dice (우심실):  {test_rv_dice:.4f}")
                    print(f"  MYO Dice (심근):   {test_myo_dice:.4f}")
                    print(f"  LV Dice (좌심실):  {test_lv_dice:.4f}")
                    print(f"  Mean Dice (평균):  {test_mean_dice:.4f}")
                    print("-" * 32)
                    
                    # Early stopping and best model saving logic (Validation-driven)
                    if mean_dice > best_metric:
                        best_metric = mean_dice
                        best_metric_epoch = epoch
                        epochs_no_improve = 0  # Reset patience
                        
                        torch.save(model.state_dict(), os.path.join(base_dir, "best_metric_model_hr.pth"))
                        print(f"[BEST] New best HR validation metric: {best_metric:.4f}! Saved model weight.")
                    else:
                        epochs_no_improve += 1
                        print(f"[INFO] No improvement. Patience count: {epochs_no_improve}/{early_stopping_patience}")
                
                # Check for Early Stopping
                if epochs_no_improve >= early_stopping_patience:
                    print("="*60)
                    print(f"[STOP] EARLY STOPPING TRIGGERED AT EPOCH {epoch} (HR)!")
                    print(f"No improvement for {early_stopping_patience} consecutive validations.")
                    print(f"Loading best weights from epoch {best_metric_epoch} (Mean Dice: {best_metric:.4f}).")
                    print("="*60)
                    break
                
            else:
                # Carry over last known validation scores
                last_mean = history["val_mean_dice"][-1] if history["val_mean_dice"] else 0.0
                last_rv = history["val_rv_dice"][-1] if history["val_rv_dice"] else 0.0
                last_myo = history["val_myo_dice"][-1] if history["val_myo_dice"] else 0.0
                last_lv = history["val_lv_dice"][-1] if history["val_lv_dice"] else 0.0
                
                history["val_mean_dice"].append(last_mean)
                history["val_rv_dice"].append(last_rv)
                history["val_myo_dice"].append(last_myo)
                history["val_lv_dice"].append(last_lv)
                
                # Carry over last known test scores for alignment
                last_test_mean = history["test_mean_dice"][-1] if history["test_mean_dice"] else 0.0
                last_test_rv = history["test_rv_dice"][-1] if history["test_rv_dice"] else 0.0
                last_test_myo = history["test_myo_dice"][-1] if history["test_myo_dice"] else 0.0
                last_test_lv = history["test_lv_dice"][-1] if history["test_lv_dice"] else 0.0
                
                history["test_mean_dice"].append(last_test_mean)
                history["test_rv_dice"].append(last_test_rv)
                history["test_myo_dice"].append(last_test_myo)
                history["test_lv_dice"].append(last_test_lv)
            
            # Periodic checkpoint backup
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_metric": best_metric,
                "best_metric_epoch": best_metric_epoch
            }
            torch.save(checkpoint, os.path.join(base_dir, "latest_checkpoint_hr.pth"))
            
            # Save history JSON and update curves plot
            with open(history_file, "w") as f:
                json.dump(history, f, indent=4)
            save_training_plot_hr(history, plot_file)
            
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("[WARN] TRAINING INTERRUPTED BY USER (KeyboardInterrupt)")
        print("Saving current history and state...")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print(f"[ERROR] TRAINING CRASHED WITH ERROR: {e}")
        print("="*60)
        raise e
    finally:
        if history["epoch"]:
            with open(history_file, "w") as f:
                json.dump(history, f, indent=4)
            save_training_plot_hr(history, plot_file)
            print("Successfully saved HR training history JSON and generated training curves chart.")
            
if __name__ == "__main__":
    train_pipeline_hr()
