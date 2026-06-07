@echo off
echo ============================================================
echo STEP 3: TensorRT Compilation via NVIDIA Docker Container
echo ============================================================

set BASE_DIR=%~dp0
set ONNX_FILE=best_metric_model_hr.onnx
set ENGINE_FILE=best_metric_model_hr.engine

if not exist "%BASE_DIR%%ONNX_FILE%" (
    echo [ERROR] %ONNX_FILE% not found! Please run export_onnx.py first.
    exit /b 1
)

echo [INFO] Pulling official NVIDIA TensorRT Docker image (nvcr.io/nvidia/tensorrt:23.08-py3)...
echo [INFO] This image is large (~10GB) and may take a few minutes to download on first run.
docker pull nvcr.io/nvidia/tensorrt:23.08-py3
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to pull NVIDIA TensorRT docker image. Please check internet connection or docker daemon.
    exit /b %ERRORLEVEL%
)

echo [INFO] Running TensorRT compilation in container with GPU acceleration...
echo [INFO] Source ONNX: %ONNX_FILE%
echo [INFO] Target Engine: %ENGINE_FILE%

:: Run trtexec inside the container, saving output log to assets/tensorrt_compile_log.txt
docker run --gpus all --rm -v "%BASE_DIR%:/workspace" -w /workspace nvcr.io/nvidia/tensorrt:23.08-py3 trtexec --onnx=/workspace/%ONNX_FILE% --saveEngine=/workspace/%ENGINE_FILE% --fp16 --minShapes=input:1x1x128x128x8 --optShapes=input:1x1x128x128x16 --maxShapes=input:1x1x128x128x32 > "%BASE_DIR%assets\tensorrt_compile_log.txt" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] TensorRT compilation failed!
    echo Please check assets\tensorrt_compile_log.txt for details.
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] TensorRT compilation completed!
echo [SUCCESS] Engine saved to: %ENGINE_FILE%
echo [SUCCESS] Compile logs saved to: assets\tensorrt_compile_log.txt
