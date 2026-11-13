cmake_minimum_required(VERSION 3.20)
project(CUDA_Ultrasound LANGUAGES CXX CUDA)

# C++ and CUDA Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target NVIDIA GPU Architecture: Ampere sm_86 (RTX 3050 Laptop / Compute Capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Output Directory for Binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Compilation Flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas=-v")

# Include Directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# -----------------------------------------------------------------------------
# Diagnostic & Verification Targets
# -----------------------------------------------------------------------------
add_executable(cuda_check src/kernels/cuda_check.cu)
add_executable(cuda_load_test src/kernels/cuda_load_test.cu)

# -----------------------------------------------------------------------------
# CUDA DAS Beamforming Implementations (V1–V7)
# -----------------------------------------------------------------------------
add_executable(cuda_das_v1 src/kernels/cuda_das_v1.cu)
add_executable(cuda_das_v2 src/kernels/cuda_das_v2.cu)
add_executable(cuda_das_v3 src/kernels/cuda_das_v3.cu)
add_executable(cuda_das_v4 src/kernels/cuda_das_v4.cu)
add_executable(cuda_das_v5 src/kernels/cuda_das_v5.cu)
add_executable(cuda_das_v6 src/kernels/cuda_das_v6.cu)
add_executable(cuda_das_v7 src/kernels/cuda_das_v7.cu)

