cmake_minimum_required(VERSION 3.12)
project(CardioSegInferenceEngine LANGUAGES CXX)

# Require C++17 for ONNX Runtime compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 1. Locate OpenMP for parallelized ArgMax post-processing acceleration
find_package(OpenMP REQUIRED)

# 2. Configurable path for ONNX Runtime C++ SDK
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "Path to ONNX Runtime SDK root directory")

# Declare Shared Library Target
add_library(cardioseg_engine SHARED
    inference_engine.h
    inference_engine.cpp
)

# Set compile definitions and links
target_link_libraries(cardioseg_engine PRIVATE OpenMP::OpenMP_CXX)

# Configure header search paths and binary links
if(EXISTS "${ONNXRUNTIME_ROOT_DIR}")
    message(STATUS "ONNX Runtime root configured at: ${ONNXRUNTIME_ROOT_DIR}")
    target_include_directories(cardioseg_engine PUBLIC "${ONNXRUNTIME_ROOT_DIR}/include")
    target_link_directories(cardioseg_engine PUBLIC "${ONNXRUNTIME_ROOT_DIR}/lib")
    
    if(WIN32)
        target_link_libraries(cardioseg_engine PRIVATE onnxruntime.lib)
    else()
        target_link_libraries(cardioseg_engine PRIVATE onnxruntime)
    endif()
else()
    message(STATUS "ONNXRUNTIME_ROOT_DIR not set. Searching system default paths...")
    # Attempt fallback search for onnxruntime headers & library
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
    
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        message(STATUS "Found ONNX Runtime Include: ${ONNXRUNTIME_INCLUDE_DIR}")
        message(STATUS "Found ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")
        target_include_directories(cardioseg_engine PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(cardioseg_engine PRIVATE ${ONNXRUNTIME_LIBRARY})
    else()
        message(WARNING "ONNX Runtime SDK could not be located on standard system paths.")
        message(WARNING "Please reconfigure CMake and set -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime")
    endif()
endif()

# 3. Demo Executable for end-to-end C++ compilation and run check
add_executable(cardioseg_demo
    demo.cpp
)

target_link_libraries(cardioseg_demo PRIVATE cardioseg_engine)
