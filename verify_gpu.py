import torch
import monai
import sys

def verify_system():
    print("="*60)
    print("=== SYSTEM & ENVIRONMENT VERIFICATION ===")
    print("="*60)
    
    # 1. Python & Libraries Version
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MONAI Version: {monai.__version__}")
    
    # 2. CUDA (GPU) Check
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"GPU Device Count: {device_count}")
        for i in range(device_count):
            print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
            
        # 3. Simple Tensor Operation on GPU (Verification)
        print("\nTesting simple tensor operation on GPU...")
        try:
            x = torch.rand(3, 3).cuda()
            y = torch.rand(3, 3).cuda()
            z = torch.matmul(x, y)
            print("  - Tensor calculation on GPU: SUCCESS!")
            print(f"  - Calculated matrix shape: {z.shape}")
        except Exception as e:
            print(f"  - Tensor calculation on GPU: FAILED! Error: {str(e)}")
    else:
        print("\nWARNING: CUDA is NOT available. PyTorch will run on CPU only.")
        print("Please check your NVIDIA drivers and CUDA toolkit installation.")
        
    print("="*60)

if __name__ == "__main__":
    verify_system()
