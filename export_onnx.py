import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import onnx

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime is not installed. We will install it in the next step if needed.")

from src.model_hr import get_3d_unet_hr

def export_and_validate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_weight_path = os.path.join(base_dir, "best_metric_model_hr.pth")
    onnx_output_path = os.path.join(base_dir, "best_metric_model_hr.onnx")
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    print("="*60)
    print("STEP 2: EXPORTING PYTORCH MODEL TO ONNX")
    print("="*60)

    # 1. Initialize and Load PyTorch Model
    device = torch.device("cpu") # Exporting on CPU is safer and more portable
    print(f"Loading PyTorch weights from: {model_weight_path}")
    model = get_3d_unet_hr(in_channels=1, out_channels=4)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 2. Define Dummy Input for 3D Shape (Batch, Channel, Depth, Height, Width)
    # Patch size used in training: 128x128x16
    dummy_input = torch.randn(1, 1, 128, 128, 16, dtype=torch.float32)
    print(f"Dummy Input shape defined: {dummy_input.shape}")

    # 3. Export to ONNX
    print(f"Exporting model to ONNX: {onnx_output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,        # Store the trained parameter weights inside the model file
        opset_version=15,          # Modern opset version compatible with TensorRT 8.x/10.x
        do_constant_folding=True,  # Optimize constants for inference speed
        input_names=["input"],     # Define input node name
        output_names=["output"],   # Define output node name
        dynamic_axes={             # Enable dynamic batch size and spatial dimensions
            "input": {0: "batch_size", 2: "depth", 3: "height", 4: "width"},
            "output": {0: "batch_size", 2: "depth", 3: "height", 4: "width"}
        }
    )
    print("ONNX Export completed successfully!")

    # 4. Verify ONNX Model Integrity
    print("\nVerifying ONNX model structure...")
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX checker: Model structure is valid! [SUCCESS]")

    # 5. Run Validation with ONNX Runtime
    print("\nInitializing ONNX Runtime session for validation...")
    try:
        # We specify CPU provider first for local verification
        ort_sess = ort.InferenceSession(onnx_output_path, providers=["CPUExecutionProvider"])
    except NameError:
        print("ONNX Runtime is not available. Please install 'onnxruntime' to run validation.")
        return

    # Create a test input tensor (using a random but fixed array)
    np.random.seed(42)
    test_input_np = np.random.randn(1, 1, 128, 128, 16).astype(np.float32)
    test_input_torch = torch.tensor(test_input_np)

    # PyTorch Inference
    with torch.no_grad():
        pytorch_output = model(test_input_torch).numpy()

    # ONNX Runtime Inference
    ort_inputs = {"input": test_input_np}
    ort_output = ort_sess.run(["output"], ort_inputs)[0]

    # Calculate differences
    abs_diff = np.abs(pytorch_output - ort_output)
    mean_abs_error = np.mean(abs_diff)
    max_abs_error = np.max(abs_diff)

    print("\n" + "="*50)
    print("PYTORCH VS ONNX NUMERICAL ERROR REPORT")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {mean_abs_error:.2e}")
    print(f"Maximum Absolute Error   : {max_abs_error:.2e}")
    
    threshold = 1e-4
    if max_abs_error < threshold:
        print(f"SUCCESS: Numerical errors are well below the safety threshold ({threshold:.2e})! [OK]")
    else:
        print(f"WARNING: Numerical errors exceed the threshold ({threshold:.2e})! Please inspect.")
    print("="*50)

    # 6. Visualizing the Error Re-validation
    print("\nGenerating visualization of PyTorch vs ONNX output maps...")
    
    # We choose the middle slice along the Z-axis (depth = 8)
    z_slice = 8
    
    # Probabilities can be obtained by applying softmax to the raw logits
    def softmax(x, axis=1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    py_probs = softmax(pytorch_output, axis=1)[0]  # Shape: (4, 128, 128, 16)
    ort_probs = softmax(ort_output, axis=1)[0]

    classes = ["Background", "Right Ventricle (RV)", "Myocardium (MYO)", "Left Ventricle (LV)"]
    
    # We will plot for class 1 (RV), 2 (MYO), and 3 (LV)
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    
    for idx, class_idx in enumerate([1, 2, 3]):
        # PyTorch Probability Slice
        im1 = axes[0, idx].imshow(py_probs[class_idx, :, :, z_slice], cmap="inferno", vmin=0, vmax=1)
        axes[0, idx].set_title(f"PyTorch Prob: {classes[class_idx]}", fontsize=11, fontweight="bold")
        axes[0, idx].axis("off")
        fig.colorbar(im1, ax=axes[0, idx], fraction=0.046, pad=0.04)

        # ONNX Runtime Probability Slice
        im2 = axes[1, idx].imshow(ort_probs[class_idx, :, :, z_slice], cmap="inferno", vmin=0, vmax=1)
        axes[1, idx].set_title(f"ONNX Runtime Prob: {classes[class_idx]}", fontsize=11, fontweight="bold")
        axes[1, idx].axis("off")
        fig.colorbar(im2, ax=axes[1, idx], fraction=0.046, pad=0.04)

        # Absolute Difference Map
        diff_map = np.abs(py_probs[class_idx, :, :, z_slice] - ort_probs[class_idx, :, :, z_slice])
        im3 = axes[2, idx].imshow(diff_map, cmap="viridis")
        axes[2, idx].set_title(f"Abs Diff (Max: {diff_map.max():.2e})", fontsize=11, fontweight="bold")
        axes[2, idx].axis("off")
        fig.colorbar(im3, ax=axes[2, idx], fraction=0.046, pad=0.04)

    plt.suptitle("PyTorch vs ONNX Runtime: Output Probability Map & Error Verification Slices", 
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    plot_save_path = os.path.join(assets_dir, "onnx_validation_comparison.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Visualization saved to: {plot_save_path} [SUCCESS]")
    print("Step 2 completed successfully!")

if __name__ == "__main__":
    export_and_validate()
