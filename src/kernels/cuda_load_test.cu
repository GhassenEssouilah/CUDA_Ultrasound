
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t error = call;                             \
        if (error != cudaSuccess) {                            \
            std::cerr << "CUDA error: "                        \
                      << cudaGetErrorString(error)             \
                      << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                           \
        }                                                     \
    } while (0)

// Dataset dimensions
constexpr size_t RF_COUNT = 75ULL * 1527 * 128;
constexpr size_t GRID_COUNT = 327680ULL * 3;
constexpr size_t IMAGE_COUNT = 1024ULL * 320;

// Load a binary file and copy it to GPU memory.
template <typename T>
std::vector<T> load_and_upload(
    const std::string& filename,
    T*& device_pointer,
    size_t expected_count)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Cannot open: " + filename);
    }

    std::streamsize file_size = file.tellg();

    if (file_size != static_cast<std::streamsize>(
            expected_count * sizeof(T))) {
        throw std::runtime_error("Unexpected file size: " + filename);
    }

    std::vector<T> host_data(expected_count);

    file.seekg(0, std::ios::beg);

    if (!file.read(
            reinterpret_cast<char*>(host_data.data()),
            file_size)) {
        throw std::runtime_error("Cannot read: " + filename);
    }

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&device_pointer),
        file_size));

    CUDA_CHECK(cudaMemcpy(
        device_pointer,
        host_data.data(),
        file_size,
        cudaMemcpyHostToDevice));

    std::cout << std::left << std::setw(24)
              << filename
              << std::right << std::setw(10)
              << std::fixed << std::setprecision(2)
              << file_size / (1024.0 * 1024.0)
              << " MB  [GPU OK]\n";

    return host_data;
}

// A tiny kernel to verify that the GPU can read uploaded data.
__global__ void verify_data_kernel(
    const float* rf,
    const float* grid,
    float* output)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = rf[0];
        output[1] = rf[RF_COUNT - 1];

        output[2] = grid[0];
        output[3] = grid[GRID_COUNT - 1];
    }
}

int main()
{
    std::cout << "========== CUDA DATA LOADING TEST ==========\n\n";

    float* d_rf = nullptr;
    float* d_geometry = nullptr;
    float* d_t0 = nullptr;
    float* d_apodization = nullptr;
    float* d_initial_times = nullptr;
    float* d_t_peak = nullptr;
    float* d_grid = nullptr;
    float* d_cpu_image = nullptr;
    float* d_fs = nullptr;
    float* d_sound_speed = nullptr;
    int* d_grid_shape = nullptr;
    float* d_verification = nullptr;

    try {
        // Display GPU information.
        int device_id = 0;
        CUDA_CHECK(cudaSetDevice(device_id));

        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));

        std::cout << "GPU: " << properties.name << "\n";
        std::cout << "Compute capability: "
                  << properties.major << "."
                  << properties.minor << "\n\n";

        std::cout << "========== UPLOADING DATA ==========\n";

        const std::string folder = "data/binary/";

        auto rf = load_and_upload(
            folder + "rf_data.bin", d_rf, RF_COUNT);

        auto geometry = load_and_upload(
            folder + "probe_geometry.bin", d_geometry, 128 * 3);

        auto t0 = load_and_upload(
            folder + "t0_delays.bin", d_t0, 75 * 128);

        auto apodization = load_and_upload(
            folder + "tx_apodizations.bin", d_apodization, 75 * 128);

        auto initial_times = load_and_upload(
            folder + "initial_times.bin", d_initial_times, 75);

        auto t_peak = load_and_upload(
            folder + "t_peak.bin", d_t_peak, 75);

        auto grid = load_and_upload(
            folder + "flatgrid.bin", d_grid, GRID_COUNT);

        auto cpu_image = load_and_upload(
            folder + "cpu_das_image.bin", d_cpu_image, IMAGE_COUNT);

        auto fs = load_and_upload(
            folder + "sampling_frequency.bin", d_fs, 1);

        auto sound_speed = load_and_upload(
            folder + "sound_speed.bin", d_sound_speed, 1);

        auto grid_shape = load_and_upload(
            folder + "grid_shape.bin", d_grid_shape, 2);

        std::cout << "\n========== DATASET PARAMETERS ==========\n";
        std::cout << "RF dimensions: 75 x 1527 x 128\n";
        std::cout << "Image dimensions: "
                  << grid_shape[0] << " x "
                  << grid_shape[1] << "\n";

        std::cout << "Sampling frequency: "
                  << fs[0] / 1e6 << " MHz\n";

        std::cout << "Sound speed: "
                  << sound_speed[0] << " m/s\n";

        // Allocate a small output buffer for GPU verification.
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_verification),
            4 * sizeof(float)));

        // Execute the verification kernel.
        verify_data_kernel<<<1, 1>>>(
            d_rf, d_grid, d_verification);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float gpu_values[4];

        CUDA_CHECK(cudaMemcpy(
            gpu_values,
            d_verification,
            sizeof(gpu_values),
            cudaMemcpyDeviceToHost));

        std::cout << "\n========== GPU READ VERIFICATION ==========\n";

        std::cout << "RF first value: "
                  << gpu_values[0]
                  << " | CPU: " << rf[0] << "\n";

        std::cout << "RF last value:  "
                  << gpu_values[1]
                  << " | CPU: " << rf.back() << "\n";

        std::cout << "Grid first value: "
                  << gpu_values[2]
                  << " | CPU: " << grid[0] << "\n";

        std::cout << "Grid last value:  "
                  << gpu_values[3]
                  << " | CPU: " << grid.back() << "\n";

        bool valid =
            gpu_values[0] == rf[0] &&
            gpu_values[1] == rf.back() &&
            gpu_values[2] == grid[0] &&
            gpu_values[3] == grid.back();

        if (!valid) {
            throw std::runtime_error(
                "GPU data verification failed!");
        }

        std::cout << "\nSUCCESS: GPU data verification passed!\n";

        // Release GPU memory.
        CUDA_CHECK(cudaFree(d_rf));
        CUDA_CHECK(cudaFree(d_geometry));
        CUDA_CHECK(cudaFree(d_t0));
        CUDA_CHECK(cudaFree(d_apodization));
        CUDA_CHECK(cudaFree(d_initial_times));
        CUDA_CHECK(cudaFree(d_t_peak));
        CUDA_CHECK(cudaFree(d_grid));
        CUDA_CHECK(cudaFree(d_cpu_image));
        CUDA_CHECK(cudaFree(d_fs));
        CUDA_CHECK(cudaFree(d_sound_speed));
        CUDA_CHECK(cudaFree(d_grid_shape));
        CUDA_CHECK(cudaFree(d_verification));

        std::cout << "GPU memory released successfully.\n";
    }
    catch (const std::exception& error) {
        std::cerr << "\nERROR: " << error.what() << "\n";

        cudaFree(d_rf);
        cudaFree(d_geometry);
        cudaFree(d_t0);
        cudaFree(d_apodization);
        cudaFree(d_initial_times);
        cudaFree(d_t_peak);
        cudaFree(d_grid);
        cudaFree(d_cpu_image);
        cudaFree(d_fs);
        cudaFree(d_sound_speed);
        cudaFree(d_grid_shape);
        cudaFree(d_verification);

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
