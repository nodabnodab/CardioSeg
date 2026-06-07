#ifndef CARDIOSEG_INFERENCE_ENGINE_H
#define CARDIOSEG_INFERENCE_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace CardioSeg {

/**
 * @brief High-performance C++ Inference Engine for CardioSeg3D.
 * Wraps the ONNX Runtime C++ API to run real-time 3D Unet segmentation
 * with optional CUDA GPU acceleration.
 */
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    /**
     * @brief Initializes the ONNX Runtime session and loads the model.
     * @param modelPath Path to the .onnx model file.
     * @param useGPU Whether to enable NVIDIA CUDA GPU execution provider.
     * @return True if initialized successfully, false otherwise.
     */
    bool Initialize(const std::string& modelPath, bool useGPU = true);

    /**
     * @brief Performs 3D Cardiac MRI segmentation.
     * @param inputVoxelData Flat vector containing raw intensity values (size: 1 * 1 * 128 * 128 * 16 = 262,144 floats).
     * @param outputMask Flat vector to store the predicted label class for each voxel (same size, values 0-3).
     * @return True if inference completed successfully, false otherwise.
     */
    bool RunInference(const std::vector<float>& inputVoxelData, std::vector<uint8_t>& outputMask);

    // Model configuration constants
    static constexpr int64_t BATCH_SIZE = 1;
    static constexpr int64_t CHANNELS = 1;
    static constexpr int64_t HEIGHT = 128;
    static constexpr int64_t WIDTH = 128;
    static constexpr int64_t DEPTH = 16;
    static constexpr int64_t NUM_CLASSES = 4;
    static constexpr int64_t INPUT_SIZE = BATCH_SIZE * CHANNELS * HEIGHT * WIDTH * DEPTH;

private:
    // ONNX Runtime Environment and Session handles
    Ort::Env m_env;
    std::unique_ptr<Ort::Session> m_session;
    
    // Model Node Names
    std::string m_inputName;
    std::string m_outputName;
    std::vector<const char*> m_inputNodeNames;
    std::vector<const char*> m_outputNodeNames;

    // Memory helper for tensor binding
    Ort::MemoryInfo m_memoryInfo;

    bool m_initialized;
};

} // namespace CardioSeg

#endif // CARDIOSEG_INFERENCE_ENGINE_H
