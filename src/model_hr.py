import torch
from monai.networks.nets import UNet

def get_3d_unet_hr(in_channels=1, out_channels=4):
    """
    Creates a standard MONAI 3D U-Net configured for High-Resolution (HR) inputs.
    Uses isotropic (2, 2, 2) strides since input patch depth is 16.
    
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
        strides=(
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2)
        ),             # Isotropic downsampling strides (Z-depth 16 -> 8 -> 4 -> 2 -> 1)
        num_res_units=2,                  # Number of residual units in each block
        norm="instance",                  # Instance Normalization for small batch sizes
        dropout=0.1                       # Dropout for regularization
    )
    return model

if __name__ == "__main__":
    # Sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_3d_unet_hr().to(device)
    print("="*60)
    print("HIGH-RESOLUTION 3D U-NET MODEL INITIALIZED SUCCESSFULLY")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} weights")
    print("="*60)
