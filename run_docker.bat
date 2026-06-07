@echo off
echo ============================================================
echo CardioSeg3D: Docker Deployment Automation Script
echo ============================================================
echo.

:: 1. Stop and remove existing container if it exists
echo [1/3] Checking for existing containers...
docker ps -a --filter "name=cardioseg3d_app" | findstr "cardioseg3d_app" > nul
if %errorlevel% equ 0 (
    echo [INFO] Stopping and removing existing 'cardioseg3d_app' container...
    docker stop cardioseg3d_app > nul 2>&1
    docker rm cardioseg3d_app > nul 2>&1
) else (
    echo [INFO] No existing container found. Proceeding.
)

:: 2. Build the Docker image
echo.
echo [2/3] Building CardioSeg3D deployment image...
docker build -t cardioseg3d:latest .
if %errorlevel% neq 0 (
    echo [ERROR] Docker build failed. Please check the build logs above.
    pause
    exit /b %errorlevel%
)

:: 3. Run the Docker container with GPU support
echo.
echo [3/3] Starting CardioSeg3D container with GPU pass-through...
docker run --gpus all -d -p 8501:8501 --name cardioseg3d_app cardioseg3d:latest
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start Docker container with GPU support.
    echo [INFO] Attempting fallback run without GPU (CPU-only)...
    docker run -d -p 8501:8501 --name cardioseg3d_app cardioseg3d:latest
    if %errorlevel% neq 0 (
        echo [ERROR] CPU fallback run also failed.
        pause
        exit /b %errorlevel%
    )
    echo [WARNING] Container started in CPU-only mode (No GPU detected/configured).
) else (
    echo [SUCCESS] Container started successfully with GPU (GeForce RTX) support!
)

echo.
echo ============================================================
echo SUCCESS: CardioSeg3D App is running!
echo URL: http://localhost:8501
echo ============================================================
echo.
pause
