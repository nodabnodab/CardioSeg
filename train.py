import os
import json
import torch
import numpy as np
from tqdm import tqdm

import monai
from monai.data import Dataset, DataLoader, list_data_collate, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

from src.dataset import get_acdc_splits, get_train_transforms, get_val_test_transforms
from src.model import get_3d_unet

def train_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Hyperparameters
    max_epochs = 150
    val_interval = 2
    batch_size = 2        # B=2 loading 4 crops per sample = 8 patches per step
    learning_rate = 1e-3
    weight_decay = 1e-5
    roi_size = (128, 128, 8) # sliding window evaluation size
    
    # 2. Get splits & datasets
    print("Loading datasets and setting up loaders...")
    train_files, val_files, _ = get_acdc_splits(base_dir)
    
    train_transforms = get_train_transforms()
    val_transforms = get_val_test_transforms()
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    # We use num_workers=0 on Windows to avoid multiprocessing issues
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
    
    # 3. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_3d_unet(in_channels=1, out_channels=4).to(device)
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
        "val_lv_dice": []
    }
    history_file = os.path.join(assets_dir, "training_history.json")
    
    best_metric = -1
    best_metric_epoch = -1
    
    print("="*60)
    print("STARTING CardioSeg3D TRAINING RUN")
    print(f"Train Cases: {len(train_files)} | Val Cases: {len(val_files)}")
    print(f"Max Epochs: {max_epochs} | Val Interval: {val_interval}")
    print("="*60)
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0
        step = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}")
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
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    
                    # Sliding window evaluation (Full 3D Volume)
                    val_outputs = sliding_window_inference(
                        val_inputs, 
                        roi_size, 
                        sw_batch_size=4, 
                        predictor=model,
                        overlap=0.5
                    )
                    
                    # Decollate batch and apply post-processing
                    val_outputs = [post_pred(x) for x in decollate_batch(val_outputs)]
                    val_labels = [post_label(x) for x in decollate_batch(val_labels)]
                    
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                # Aggregate metrics for the whole batch
                metric_batch = dice_metric.aggregate()
                dice_metric.reset()
                
                # Extract per-class dice scores (RV, MYO, LV)
                rv_dice = metric_batch[0].item()
                myo_dice = metric_batch[1].item()
                lv_dice = metric_batch[2].item()
                mean_dice = metric_batch.mean().item()
                
                history["val_mean_dice"].append(mean_dice)
                history["val_rv_dice"].append(rv_dice)
                history["val_myo_dice"].append(myo_dice)
                history["val_lv_dice"].append(lv_dice)
                
                print(f"--- [Validation Epoch {epoch}] ---")
                print(f"  RV Dice (우심실):  {rv_dice:.4f}")
                print(f"  MYO Dice (심근):   {myo_dice:.4f}")
                print(f"  LV Dice (좌심실):  {lv_dice:.4f}")
                print(f"  Mean Dice (평균):  {mean_dice:.4f}")
                print("-" * 32)
                
                # Save best model
                if mean_dice > best_metric:
                    best_metric = mean_dice
                    best_metric_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(base_dir, "best_metric_model.pth"))
                    print(f"🏆 New best validation metric: {best_metric:.4f} at epoch {best_metric_epoch}! Saved model.")
            
        else:
            # For epochs without validation, carry over last known val scores
            last_mean = history["val_mean_dice"][-1] if history["val_mean_dice"] else 0.0
            last_rv = history["val_rv_dice"][-1] if history["val_rv_dice"] else 0.0
            last_myo = history["val_myo_dice"][-1] if history["val_myo_dice"] else 0.0
            last_lv = history["val_lv_dice"][-1] if history["val_lv_dice"] else 0.0
            
            history["val_mean_dice"].append(last_mean)
            history["val_rv_dice"].append(last_rv)
            history["val_myo_dice"].append(last_myo)
            history["val_lv_dice"].append(last_lv)
            
        # Write history log after each epoch (enables real-time updating in web browser)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)
            
    print("="*60)
    print("TRAINING PROCESS COMPLETED")
    print(f"Best Validation Mean Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
    print("="*60)

if __name__ == "__main__":
    train_pipeline()
