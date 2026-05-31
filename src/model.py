import torch
from monai.networks.nets import UNet

def get_3d_unet(in_channels=1, out_channels=4):
    """
    Creates a standard MONAI 3D U-Net model.
    
    Args:
        in_channels (int): Number of input channels (1 for grayscale cardiac MRI).
        out_channels (int): Number of output classes (4 for background, RV, MYO, LV).
        
    Returns:
        torch.nn.Module: Configured 3D U-Net model.
    """
    model = UNet(
        spatial_dims=3,          # 3D Medical volume
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),  # Encoder/Decoder channels
        strides=(2, 2, 2, 2),             # Downsampling factors
        num_res_units=2,                  # Number of residual units in each block
        norm="instance",                  # Instance Normalization (essential for small batch sizes like 2)
        dropout=0.1                       # Dropout for regularization
    )
    return model

if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_3d_unet().to(device)
    print("="*60)
    print("3D U-NET MODEL INITIALIZED SUCCESSFULLY")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} weights")
    print("="*60)
