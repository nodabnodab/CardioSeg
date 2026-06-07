#include "inference_engine.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

int main(int argc, char* argv[]) {
    std::cout << "============================================================" << std::endl;
    std::cout << "CardioSeg3D C++ Inference Engine Standalone Demo" << std::endl;
    std::cout << "============================================================" << std::endl;

    // 1. Resolve model path from command argument or fallback to default
    std::string modelPath = "../../best_metric_model_hr.onnx";
    if (argc > 1) {
        modelPath = argv[1];
    }
    std::cout << "[INFO] Target Model File: " << modelPath << std::endl;

    // 2. Instantiate and Initialize the C++ Inference Engine
    CardioSeg::InferenceEngine engine;
    
    // Attempt GPU (CUDA) initialization first, fallback to CPU
    std::cout << "[INFO] Initializing Session..." << std::endl;
    bool success = engine.Initialize(modelPath, true);
    if (!success) {
        std::cerr << "[FATAL] Failed to initialize the C++ Inference Engine session." << std::endl;
        return -1;
    }

    // 3. Generate mock 3D MRI voxel intensity inputs
    std::cout << "[INFO] Generating simulated 3D voxel input buffer (" 
              << CardioSeg::InferenceEngine::INPUT_SIZE << " floats)..." << std::endl;
              
    std::vector<float> mockInput(CardioSeg::InferenceEngine::INPUT_SIZE);
    
    // Fill with simulated normalized MR intensity values [0.0, 1.0]
    std::mt19937 rng(42); // Seeded random generator
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < mockInput.size(); ++i) {
        mockInput[i] = dist(rng);
    }

    // 4. Run GPU-accelerated C++ network inference
    std::vector<uint8_t> outputMask;
    std::cout << "[INFO] Running inference on model..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    bool runSuccess = engine.RunInference(mockInput, outputMask);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;

    if (!runSuccess) {
        std::cerr << "[FATAL] Inference run returned failure." << std::endl;
        return -1;
    }

    // 5. Compute voxel segmentation class statistics
    size_t bgCount = 0;
    size_t rvCount = 0;
    size_t myoCount = 0;
    size_t lvCount = 0;
    size_t errorCount = 0;

    for (size_t val : outputMask) {
        if (val == 0) bgCount++;
        else if (val == 1) rvCount++;
        else if (val == 2) myoCount++;
        else if (val == 3) lvCount++;
        else errorCount++;
    }

    std::cout << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "🎉 INFERENCE COMPLETED SUCCESSFULLY!" << std::endl;
    std::cout << "⏱️  Inference Time: " << duration.count() << " ms" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "[STATS] Segmented Voxels Output Summary:" << std::endl;
    std::cout << "  - Class 0 (Background):  " << bgCount << " voxels" << std::endl;
    std::cout << "  - Class 1 (Right Ventricle): " << rvCount << " voxels" << std::endl;
    std::cout << "  - Class 2 (Myocardium):      " << myoCount << " voxels" << std::endl;
    std::cout << "  - Class 3 (Left Ventricle):  " << lvCount << " voxels" << std::endl;
    if (errorCount > 0) {
        std::cout << "  - [ALERT] Invalid Label Class: " << errorCount << " voxels" << std::endl;
    }
    std::cout << "============================================================" << std::endl;

    return 0;
}
