import os
import torch
import numpy as np
from src.model_hr import get_3d_unet_hr

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_weight_path = os.path.join(base_dir, "best_metric_model_hr.pth")
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)
    input_shape = (1, 1, 128, 128, 16)
    
    # Generate a realistic test input
    input_np = np.random.randn(*input_shape).astype(np.float32)
    
    # Load PyTorch model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading PyTorch model on {device}...")
    model = get_3d_unet_hr(in_channels=1, out_channels=4).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    
    # Run PyTorch inference
    input_torch = torch.tensor(input_np).to(device)
    with torch.no_grad():
        output_torch = model(input_torch)
        # Apply Softmax to compare probabilities
        output_prob = torch.softmax(output_torch, dim=1)
        output_np = output_prob.cpu().numpy()
        
    # Save input and output
    np.save(os.path.join(assets_dir, "test_input.npy"), input_np)
    np.save(os.path.join(assets_dir, "pytorch_output.npy"), output_np)
    print("PyTorch input and output successfully saved to assets/ directory.")

if __name__ == "__main__":
    main()
