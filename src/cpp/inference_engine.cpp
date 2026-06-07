#include "inference_engine.h"
#include <iostream>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace CardioSeg {

InferenceEngine::InferenceEngine()
    : m_env(ORT_LOGGING_LEVEL_WARNING, "CardioSegInferenceEngine"),
      m_memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      m_initialized(false) {}

InferenceEngine::~InferenceEngine() {}

bool InferenceEngine::Initialize(const std::string& modelPath, bool useGPU) {
    try {
        Ort::SessionOptions sessionOptions;
        
        // Optimize configuration for inference execution
        sessionOptions.SetIntraOpNumThreads(4);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Configure CUDA Execution Provider if requested
        if (useGPU) {
            try {
                // Check ONNX Runtime's available execution providers
                std::vector<std::string> providers = Ort::GetAvailableProviders();
                auto it = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider");
                if (it != providers.end()) {
                    OrtCUDAProviderOptions cudaOptions;
                    cudaOptions.device_id = 0;
                    cudaOptions.gpu_mem_limit = SIZE_MAX;
                    cudaOptions.arena_extend_strategy = 0;
                    cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                    cudaOptions.do_copy_in_default_stream = 1;
                    
                    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
                    std::cout << "[INFO] Successfully appended CUDAExecutionProvider to SessionOptions." << std::endl;
                } else {
                    std::cout << "[WARNING] CUDAExecutionProvider is not available in this build. Falling back to CPU." << std::endl;
                }
            } catch (const std::exception& ex) {
                std::cerr << "[WARNING] Failed to initialize CUDA execution provider: " << ex.what() 
                          << ". Falling back to CPU." << std::endl;
            }
        }

        // Initialize session (Windows requires wstring; Unix requires string)
#ifdef _WIN32
        std::wstring wModelPath(modelPath.begin(), modelPath.end());
        m_session = std::make_unique<Ort::Session>(m_env, wModelPath.c_str(), sessionOptions);
#else
        m_session = std::make_unique<Ort::Session>(m_env, modelPath.c_str(), sessionOptions);
#endif

        // Dynamic naming acquisition supporting backward/forward compatibility
        Ort::AllocatorWithDefaultOptions allocator;

        // Get Input Node Details
        size_t numInputNodes = m_session->GetInputCount();
        if (numInputNodes > 0) {
#if ORT_API_VERSION >= 12
            Ort::AllocatedStringPtr inputName = m_session->GetInputNameAllocated(0, allocator);
            m_inputName = inputName.get();
#else
            char* inputName = m_session->GetInputName(0, allocator);
            m_inputName = inputName;
            allocator.Free(inputName);
#endif
        } else {
            throw std::runtime_error("Model has no input nodes.");
        }

        // Get Output Node Details
        size_t numOutputNodes = m_session->GetOutputCount();
        if (numOutputNodes > 0) {
#if ORT_API_VERSION >= 12
            Ort::AllocatedStringPtr outputName = m_session->GetOutputNameAllocated(0, allocator);
            m_outputName = outputName.get();
#else
            char* outputName = m_session->GetOutputName(0, allocator);
            m_outputName = outputName;
            allocator.Free(outputName);
#endif
        } else {
            throw std::runtime_error("Model has no output nodes.");
        }

        m_initialized = true;
        std::cout << "[SUCCESS] InferenceEngine loaded model: " << modelPath << std::endl;
        std::cout << "[INFO] Input Node Name: " << m_inputName << " | Output Node Name: " << m_outputName << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] InferenceEngine Initialization failed: " << e.what() << std::endl;
        m_initialized = false;
        return false;
    }
}

bool InferenceEngine::RunInference(const std::vector<float>& inputVoxelData, std::vector<uint8_t>& outputMask) {
    if (!m_initialized) {
        std::cerr << "[ERROR] InferenceEngine is not initialized." << std::endl;
        return false;
    }

    if (inputVoxelData.size() != INPUT_SIZE) {
        std::cerr << "[ERROR] Input buffer size mismatch. Expected: " << INPUT_SIZE 
                  << ", Got: " << inputVoxelData.size() << std::endl;
        return false;
    }

    try {
        // Define input and output shape metrics
        std::vector<int64_t> inputShape = { BATCH_SIZE, CHANNELS, HEIGHT, WIDTH, DEPTH };
        std::vector<int64_t> outputShape = { BATCH_SIZE, NUM_CLASSES, HEIGHT, WIDTH, DEPTH };
        
        size_t outputRawSize = BATCH_SIZE * NUM_CLASSES * HEIGHT * WIDTH * DEPTH;
        std::vector<float> outputTensorValues(outputRawSize, 0.0f);

        // Bind raw data buffers directly into Ort Tensors (Zero-Copy wrapper)
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            m_memoryInfo,
            const_cast<float*>(inputVoxelData.data()),
            inputVoxelData.size(),
            inputShape.data(),
            inputShape.size()
        );

        Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
            m_memoryInfo,
            outputTensorValues.data(),
            outputTensorValues.size(),
            outputShape.data(),
            outputShape.size()
        );

        // Run network inference on GPU/CPU
        const char* inputNodeNames[] = { m_inputName.c_str() };
        const char* outputNodeNames[] = { m_outputName.c_str() };

        m_session->Run(
            Ort::RunOptions{nullptr},
            inputNodeNames,
            &inputTensor,
            1,
            outputNodeNames,
            &outputTensor,
            1
        );

        // Post-Processing: Parallelized ArgMax over channels using OpenMP
        int64_t numVoxels = HEIGHT * WIDTH * DEPTH;
        outputMask.resize(numVoxels);

#pragma omp parallel for
        for (int64_t i = 0; i < numVoxels; ++i) {
            float maxLogit = -std::numeric_limits<float>::infinity();
            uint8_t predictedClass = 0;

            for (int64_t c = 0; c < NUM_CLASSES; ++c) {
                // Buffer layout is contiguous [B, C, H, W, D], where B=1.
                // Voxel intensity for channel c is at: c * (H*W*D) + voxel_index
                int64_t idx = c * numVoxels + i;
                float logitVal = outputTensorValues[idx];
                if (logitVal > maxLogit) {
                    maxLogit = logitVal;
                    predictedClass = static_cast<uint8_t>(c);
                }
            }
            outputMask[i] = predictedClass;
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Inference run failed: " << e.what() << std::endl;
        return false;
    }
}

} // namespace CardioSeg
