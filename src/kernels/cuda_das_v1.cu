
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <cstdint>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t error = call;                             \
        if (error != cudaSuccess) {                            \
            std::cerr << "CUDA ERROR: "                        \
                      << cudaGetErrorString(error)             \
                      << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                           \
        }                                                     \
    } while (0)

// Dataset dimensions
constexpr int N_TX = 75;
constexpr int N_SAMPLES = 1527;
constexpr int N_ELEMENTS = 128;

constexpr int IMAGE_WIDTH = 320;
constexpr int IMAGE_HEIGHT = 1024;
constexpr int TOTAL_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;

// ---------------------------------------------------------
// Read binary file
// ---------------------------------------------------------

template <typename T>
std::vector<T> load_binary(
    const std::string& filename,
    size_t count)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Cannot open " + filename);
    }

    size_t expected_size = count * sizeof(T);
    size_t actual_size = static_cast<size_t>(file.tellg());

    if (actual_size != expected_size) {
        throw std::runtime_error("Incorrect file size: " + filename);
    }

    std::vector<T> data(count);

    file.seekg(0);

    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(expected_size)
    );

    if (!file) {
        throw std::runtime_error("Cannot read " + filename);
    }

    return data;
}

// ---------------------------------------------------------
// Upload data to GPU
// ---------------------------------------------------------

template <typename T>
T* upload_to_gpu(const std::vector<T>& host_data)
{
    T* device_data = nullptr;

    size_t bytes = host_data.size() * sizeof(T);

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&device_data),
        bytes
    ));

    CUDA_CHECK(cudaMemcpy(
        device_data,
        host_data.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    return device_data;
}

// ---------------------------------------------------------
// CUDA DAS KERNEL
//
// One CUDA thread reconstructs one image pixel.
// ---------------------------------------------------------

__global__ void das_kernel(
    const float* rf_data,
    const float* probe_geometry,
    const float* t0_delays,
    const float* tx_apodizations,
    const float* initial_times,
    const float* t_peak,
    const float* flatgrid,
    float* output,
    float sampling_frequency,
    float sound_speed,
    int output_count,
    int total_pixels)
{
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;

    if (pixel >= output_count)
        return;

    // For validation, distribute sampled pixels across
    // the complete image.
    int source_pixel;

    if (output_count == total_pixels) {
        source_pixel = pixel;
    }
    else {
        source_pixel = static_cast<int>(
            (static_cast<long long>(pixel) *
             (total_pixels - 1)) /
            (output_count - 1)
        );
    }

    // Image pixel coordinates
    float px = flatgrid[source_pixel * 3 + 0];
    float py = flatgrid[source_pixel * 3 + 1];
    float pz = flatgrid[source_pixel * 3 + 2];

    float pixel_sum = 0.0f;

    // Process all 75 transmission events
    for (int tx = 0; tx < N_TX; tx++) {

        float minimum_tx_time = 1.0e30f;

        // -------------------------------------------------
        // 1. Calculate transmit delay
        // -------------------------------------------------

        for (int element = 0;
             element < N_ELEMENTS;
             element++)
        {
            int apod_index = tx * N_ELEMENTS + element;

            float apodization =
                tx_apodizations[apod_index];

            // Ignore inactive transmit elements
            if (apodization <= 0.0f)
                continue;

            int geometry_index = element * 3;

            float ex = probe_geometry[geometry_index + 0];
            float ey = probe_geometry[geometry_index + 1];
            float ez = probe_geometry[geometry_index + 2];

            float dx = px - ex;
            float dy = py - ey;
            float dz = pz - ez;

            float distance = sqrtf(
                dx * dx + dy * dy + dz * dz
            );

            float receive_time = distance / sound_speed;

            float candidate =
                receive_time +
                t0_delays[apod_index];

            if (candidate < minimum_tx_time) {
                minimum_tx_time = candidate;
            }
        }

        if (minimum_tx_time == 1.0e30f)
            continue;

        float transmit_delay =
            minimum_tx_time - initial_times[tx];

        // -------------------------------------------------
        // 2. Receive delays and RF interpolation
        // -------------------------------------------------

        float event_sum = 0.0f;

        for (int element = 0;
             element < N_ELEMENTS;
             element++)
        {
            int apod_index = tx * N_ELEMENTS + element;

            float apodization =
                tx_apodizations[apod_index];

            if (apodization <= 0.0f)
                continue;

            int geometry_index = element * 3;

            float ex = probe_geometry[geometry_index + 0];
            float ey = probe_geometry[geometry_index + 1];
            float ez = probe_geometry[geometry_index + 2];

            float dx = px - ex;
            float dy = py - ey;
            float dz = pz - ez;

            float distance = sqrtf(
                dx * dx + dy * dy + dz * dz
            );

            float receive_time = distance / sound_speed;

            // Total travel time, including RF pulse reference
            float total_time =
                transmit_delay +
                receive_time +
                t_peak[tx];

            // Convert time into RF sample position
            float sample_position =
                total_time * sampling_frequency;

            int i0 = static_cast<int>(floorf(sample_position));
            int i1 = i0 + 1;

            // Both interpolation samples must be valid
            if (i0 < 0 || i1 >= N_SAMPLES)
                continue;

            float alpha = sample_position - i0;

            // RF memory layout:
            // [transmission][sample][receive element]

            int index0 =
                (tx * N_SAMPLES + i0) * N_ELEMENTS + element;

            int index1 =
                (tx * N_SAMPLES + i1) * N_ELEMENTS + element;

            float rf0 = rf_data[index0];
            float rf1 = rf_data[index1];

            // Linear interpolation
            float interpolated =
                rf0 + alpha * (rf1 - rf0);

            event_sum += interpolated * apodization;
        }

        // Coherent summation across transmissions
        pixel_sum += event_sum;
    }

    output[pixel] = pixel_sum;
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------

int main(int argc, char** argv)
{
    try {

        bool full_mode =
            (argc > 1 && std::string(argv[1]) == "--full");

        const int output_count =
            full_mode ? TOTAL_PIXELS : 4096;

        std::cout << "\n========== CUDA DAS RECONSTRUCTION ==========\n";

        std::cout << "Mode: "
                  << (full_mode ? "FULL IMAGE" : "VALIDATION TEST")
                  << "\n";

        std::cout << "Pixels: " << output_count << "\n";

        // -------------------------------------------------
        // Load dataset
        // -------------------------------------------------

        const std::string folder = "data/binary/";

        std::cout << "\nLoading dataset...\n";

        auto rf = load_binary<float>(
            folder + "rf_data.bin",
            N_TX * N_SAMPLES * N_ELEMENTS
        );

        auto geometry = load_binary<float>(
            folder + "probe_geometry.bin",
            N_ELEMENTS * 3
        );

        auto t0 = load_binary<float>(
            folder + "t0_delays.bin",
            N_TX * N_ELEMENTS
        );

        auto apodization = load_binary<float>(
            folder + "tx_apodizations.bin",
            N_TX * N_ELEMENTS
        );

        auto initial_times = load_binary<float>(
            folder + "initial_times.bin", N_TX
        );

        auto peak = load_binary<float>(
            folder + "t_peak.bin", N_TX
        );

        auto grid = load_binary<float>(
            folder + "flatgrid.bin",
            TOTAL_PIXELS * 3
        );

        auto cpu_image = load_binary<float>(
            folder + "cpu_das_image.bin",
            TOTAL_PIXELS
        );

        auto fs = load_binary<float>(
            folder + "sampling_frequency.bin", 1
        );

        auto sound_speed = load_binary<float>(
            folder + "sound_speed.bin", 1
        );

        // -------------------------------------------------
        // Transfer arrays to GPU
        // -------------------------------------------------

        std::cout << "Uploading arrays to GPU...\n";

        float* d_rf = upload_to_gpu(rf);
        float* d_geometry = upload_to_gpu(geometry);
        float* d_t0 = upload_to_gpu(t0);
        float* d_apod = upload_to_gpu(apodization);
        float* d_initial = upload_to_gpu(initial_times);
        float* d_peak = upload_to_gpu(peak);
        float* d_grid = upload_to_gpu(grid);

        float* d_output = nullptr;

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_output),
            output_count * sizeof(float)
        ));

        // -------------------------------------------------
        // Launch CUDA kernel
        // -------------------------------------------------

        constexpr int THREADS = 256;

        int blocks =
            (output_count + THREADS - 1) / THREADS;

        std::cout << "\nLaunching DAS kernel...\n";
        std::cout << "Blocks: " << blocks << "\n";
        std::cout << "Threads per block: " << THREADS << "\n";

        cudaEvent_t start, stop;

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        das_kernel<<<blocks, THREADS>>>(
            d_rf,
            d_geometry,
            d_t0,
            d_apod,
            d_initial,
            d_peak,
            d_grid,
            d_output,
            fs[0],
            sound_speed[0],
            output_count,
            TOTAL_PIXELS
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float kernel_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &kernel_ms, start, stop
        ));

        // -------------------------------------------------
        // Copy reconstructed image back to CPU
        // -------------------------------------------------

        std::vector<float> gpu_output(output_count);

        CUDA_CHECK(cudaMemcpy(
            gpu_output.data(),
            d_output,
            output_count * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        // -------------------------------------------------
        // CPU vs GPU validation
        // -------------------------------------------------

        double squared_error_sum = 0.0;
        double cpu_squared_sum = 0.0;
        double maximum_error = 0.0;

        for (int i = 0; i < output_count; i++) {

            int source_pixel;

            if (full_mode) {
                source_pixel = i;
            }
            else {
                source_pixel = static_cast<int>(
                    (static_cast<long long>(i) *
                     (TOTAL_PIXELS - 1)) /
                    (output_count - 1)
                );
            }

            double cpu_value = cpu_image[source_pixel];
            double gpu_value = gpu_output[i];

            double error = std::abs(
                gpu_value - cpu_value
            );

            squared_error_sum += error * error;
            cpu_squared_sum += cpu_value * cpu_value;

            if (error > maximum_error)
                maximum_error = error;
        }

        double rmse = std::sqrt(
            squared_error_sum / output_count
        );

        double cpu_rms = std::sqrt(
            cpu_squared_sum / output_count
        );

        double relative_rmse =
            cpu_rms > 0.0
            ? 100.0 * rmse / cpu_rms
            : 0.0;

        // -------------------------------------------------
        // Save output
        // -------------------------------------------------

        std::string output_filename =
            full_mode
            ? "results/cuda_das_image.bin"
            : "results/cuda_das_test.bin";

        std::ofstream output_file(
            output_filename,
            std::ios::binary
        );

        output_file.write(
            reinterpret_cast<const char*>(gpu_output.data()),
            output_count * sizeof(float)
        );

        if (!output_file)
            throw std::runtime_error("Cannot save GPU output");

        // -------------------------------------------------
        // Results
        // -------------------------------------------------

        std::cout << "\n========== DAS RESULTS ==========\n";

        std::cout << "Pixels reconstructed: "
                  << output_count << "\n";

        std::cout << std::fixed << std::setprecision(6);

        std::cout << "Kernel execution time: "
                  << kernel_ms << " ms\n";

        std::cout << "RMSE: "
                  << rmse << "\n";

        std::cout << "Maximum absolute error: "
                  << maximum_error << "\n";

        std::cout << "Relative RMSE: "
                  << relative_rmse << " %\n";

        std::cout << "Output saved: "
                  << output_filename << "\n";

        // -------------------------------------------------
        // Cleanup
        // -------------------------------------------------

        CUDA_CHECK(cudaFree(d_rf));
        CUDA_CHECK(cudaFree(d_geometry));
        CUDA_CHECK(cudaFree(d_t0));
        CUDA_CHECK(cudaFree(d_apod));
        CUDA_CHECK(cudaFree(d_initial));
        CUDA_CHECK(cudaFree(d_peak));
        CUDA_CHECK(cudaFree(d_grid));
        CUDA_CHECK(cudaFree(d_output));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        std::cout << "\nCUDA DAS completed successfully!\n";

    }
    catch (const std::exception& error) {

        std::cerr << "\nERROR: "
                  << error.what() << "\n";

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
