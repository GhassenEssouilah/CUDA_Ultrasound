#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0)
    {
        std::cerr << "CUDA Error: "
                  << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "========== CUDA DEVICE CHECK ==========\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute capability: "
              << prop.major << "." << prop.minor << "\n";
    std::cout << "GPU memory: "
              << prop.totalGlobalMem / (1024 * 1024)
              << " MB\n";

    std::cout << "\nCUDA runtime initialized successfully!\n";

    return 0;
}