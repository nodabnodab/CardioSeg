import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    engine_path = os.path.join(base_dir, "best_metric_model_hr.engine")
    input_path = os.path.join(base_dir, "assets", "test_input.npy")
    pytorch_out_path = os.path.join(base_dir, "assets", "pytorch_output.npy")
    
    if not os.path.exists(engine_path):
        print(f"Error: Engine file not found at {engine_path}")
        return
        
    # Load input data
    input_data = np.load(input_path)
    
    # Load TensorRT engine
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    with open(engine_path, "rb") as f:
        engine_bytes = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    
    # Set active input shape
    context.set_input_shape("input", input_data.shape)
    
    # Allocate memory
    h_input = np.ascontiguousarray(input_data)
    h_output = np.empty((1, 4, 128, 128, 16), dtype=np.float32)
    
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    # Set addresses
    context.set_tensor_address("input", int(d_input))
    context.set_tensor_address("output", int(d_output))
    
    # Copy host to device
    cuda.memcpy_htod(d_input, h_input)
    
    # Run inference
    context.execute_async_v3(stream_handle=0)
    
    # Copy device to host
    cuda.memcpy_dtoh(h_output, d_output)
    
    # Apply softmax
    trt_prob = softmax(h_output, axis=1)
    
    # Load PyTorch output
    pytorch_prob = np.load(pytorch_out_path)
    
    # Calculate difference
    mae = np.mean(np.abs(trt_prob - pytorch_prob))
    max_error = np.max(np.abs(trt_prob - pytorch_prob))
    
    # Calculate structural agreement (percentage of pixels with identical max argmax)
    trt_pred = np.argmax(trt_prob, axis=1)
    pytorch_pred = np.argmax(pytorch_prob, axis=1)
    agreement = np.mean(trt_pred == pytorch_pred) * 100.0
    
    print("="*60)
    print("TensorRT FP16 vs PyTorch FP32 Accuracy Verification")
    print("="*60)
    print(f"Mean Absolute Error (MAE)  : {mae:.2e}")
    print(f"Maximum Absolute Error      : {max_error:.2e}")
    print(f"Pixel Argmax Agreement Rate : {agreement:.4f}%")
    print("="*60)
    
    # Save the metrics to a text file for easily loading on host
    with open(os.path.join(base_dir, "assets", "accuracy_metrics.txt"), "w") as f:
        f.write(f"{mae:.2e},{max_error:.2e},{agreement:.4f}")
    print("Metrics saved to assets/accuracy_metrics.txt [SUCCESS]")

if __name__ == "__main__":
    main()
