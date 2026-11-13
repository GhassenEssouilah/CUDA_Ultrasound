
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <algorithm>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t error = call;                                      \
    if (error != cudaSuccess) {                                    \
        std::cerr << "CUDA ERROR: "                                \
                  << cudaGetErrorString(error)                    \
                  << " at line " << __LINE__ << std::endl;         \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)

// ---------------------------------------------------------
// DATASET DIMENSIONS
// ---------------------------------------------------------

constexpr int N_TX = 75;
constexpr int N_SAMPLES = 1527;
constexpr int N_ELEMENTS = 128;

constexpr int IMAGE_WIDTH = 320;
constexpr int IMAGE_HEIGHT = 1024;
constexpr int TOTAL_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;

// 32 threads cooperate on one pixel.
constexpr int WARP_SIZE = 32;
constexpr int THREADS = 256;
constexpr int WARPS_PER_BLOCK = THREADS / WARP_SIZE;

constexpr int WARMUP_RUNS = 2;
constexpr int BENCHMARK_RUNS = 20;

// ---------------------------------------------------------
// BINARY FILE LOADER
// ---------------------------------------------------------

template <typename T>
std::vector<T> load_binary(
    const std::string& filename,
    size_t count)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file)
        throw std::runtime_error("Cannot open " + filename);

    size_t expected_size = count * sizeof(T);
    size_t actual_size = static_cast<size_t>(file.tellg());

    if (actual_size != expected_size)
        throw std::runtime_error("Incorrect file size: " + filename);

    std::vector<T> data(count);

    file.seekg(0);

    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(expected_size)
    );

    if (!file)
        throw std::runtime_error("Cannot read " + filename);

    return data;
}

// ---------------------------------------------------------
// WARP REDUCTIONS
// All 32 lanes must participate.
// ---------------------------------------------------------

__device__ __forceinline__
float warp_reduce_min(float value)
{
    constexpr unsigned int MASK = 0xffffffffu;

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(MASK, value, offset);
        value = fminf(value, other);
    }

    return __shfl_sync(MASK, value, 0);
}

__device__ __forceinline__
float warp_reduce_sum(float value)
{
    constexpr unsigned int MASK = 0xffffffffu;

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(MASK, value, offset);
        value += other;
    }

    return value;
}

// ---------------------------------------------------------
// RECEIVE TIME
// One calculation per pixel-element pair.
// ---------------------------------------------------------

__device__ __forceinline__
float calculate_receive_time(
    float px,
    float py,
    float pz,
    int element,
    const float* probe_geometry,
    float sound_speed)
{
    int index = element * 3;

    float dx = px - probe_geometry[index + 0];
    float dy = py - probe_geometry[index + 1];
    float dz = pz - probe_geometry[index + 2];

    float distance = sqrtf(
        dx * dx + dy * dy + dz * dz
    );

    return distance / sound_speed;
}

// ---------------------------------------------------------
// TRANSMIT ARRIVAL CANDIDATE
// V6: apodization is always 1.0 for this dataset.
// ---------------------------------------------------------

__device__ __forceinline__
float transmit_candidate(
    int tx,
    int element,
    float receive_time,
    const float* t0_delays)
{
    int index = tx * N_ELEMENTS + element;

    return receive_time + t0_delays[index];
}

// ---------------------------------------------------------
// RF INTERPOLATION
// V6: no apodization read or multiplication.
// ---------------------------------------------------------

__device__ __forceinline__
float receive_contribution(
    int tx,
    int element,
    float receive_time,
    float transmit_delay,
    const float* rf_data,
    const float* t_peak,
    float sampling_frequency)
{
    float total_time =
        transmit_delay +
        receive_time +
        t_peak[tx];

    float sample_position = total_time * sampling_frequency;

    int i0 = static_cast<int>(floorf(sample_position));
    int i1 = i0 + 1;

    if (i0 < 0 || i1 >= N_SAMPLES)
        return 0.0f;

    float alpha = sample_position - static_cast<float>(i0);

    int index0 =
        (tx * N_SAMPLES + i0) * N_ELEMENTS + element;

    int index1 =
        (tx * N_SAMPLES + i1) * N_ELEMENTS + element;

    float rf0 = rf_data[index0];
    float rf1 = rf_data[index1];

    float interpolated = rf0 + alpha * (rf1 - rf0);

    return interpolated;
}

// ---------------------------------------------------------
// CUDA DAS V6: SPECIALIZED WARP-COOPERATIVE KERNEL
//
// One warp reconstructs one pixel.
// Each thread handles four receive elements.
// 32 threads x 4 elements = 128 elements.
//
// Specialization:
// All tx_apodizations are exactly 1.0 in this dataset.
// ---------------------------------------------------------

__global__ void das_kernel_v6(
    const float* rf_data,
    const float* probe_geometry,
    const float* t0_delays,
    const float* initial_times,
    const float* t_peak,
    const float* flatgrid,
    float* output,
    float sampling_frequency,
    float sound_speed,
    int output_count,
    int total_pixels)
{
    int lane = threadIdx.x % WARP_SIZE;
    int warp_in_block = threadIdx.x / WARP_SIZE;

    int pixel =
        blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    // Every lane in a warp has the same pixel.
    if (pixel >= output_count)
        return;

    // -----------------------------------------------------
    // 1. MAP OUTPUT PIXEL TO SOURCE PIXEL
    // -----------------------------------------------------

    int source_pixel;

    if (output_count == total_pixels)
    {
        source_pixel = pixel;
    }
    else
    {
        source_pixel = static_cast<int>(
            (static_cast<long long>(pixel) *
             (total_pixels - 1)) /
            (output_count - 1)
        );
    }

    float px = flatgrid[source_pixel * 3 + 0];
    float py = flatgrid[source_pixel * 3 + 1];
    float pz = flatgrid[source_pixel * 3 + 2];

    // -----------------------------------------------------
    // 2. EACH THREAD CALCULATES FOUR RECEIVE TIMES
    // -----------------------------------------------------

    int e0 = lane;
    int e1 = lane + 32;
    int e2 = lane + 64;
    int e3 = lane + 96;

    float rt0 = calculate_receive_time(
        px, py, pz, e0, probe_geometry, sound_speed
    );

    float rt1 = calculate_receive_time(
        px, py, pz, e1, probe_geometry, sound_speed
    );

    float rt2 = calculate_receive_time(
        px, py, pz, e2, probe_geometry, sound_speed
    );

    float rt3 = calculate_receive_time(
        px, py, pz, e3, probe_geometry, sound_speed
    );

    // -----------------------------------------------------
    // 3. DAS RECONSTRUCTION
    // -----------------------------------------------------

    float pixel_sum = 0.0f;

    for (int tx = 0; tx < N_TX; tx++)
    {
        // -------------------------------------------------
        // 3A. EACH THREAD FINDS ITS LOCAL MINIMUM
        // -------------------------------------------------

        float local_min = 1.0e30f;

        local_min = fminf(
            local_min,
            transmit_candidate(tx, e0, rt0, t0_delays)
        );

        local_min = fminf(
            local_min,
            transmit_candidate(tx, e1, rt1, t0_delays)
        );

        local_min = fminf(
            local_min,
            transmit_candidate(tx, e2, rt2, t0_delays)
        );

        local_min = fminf(
            local_min,
            transmit_candidate(tx, e3, rt3, t0_delays)
        );

        // -------------------------------------------------
        // 3B. WARP FINDS THE GLOBAL MINIMUM
        // -------------------------------------------------

        float minimum_tx_time = warp_reduce_min(local_min);

        if (minimum_tx_time >= 1.0e30f)
            continue;

        float transmit_delay =
            minimum_tx_time - initial_times[tx];

        // -------------------------------------------------
        // 3C. EACH THREAD CALCULATES FOUR RF CONTRIBUTIONS
        // -------------------------------------------------

        float local_sum = 0.0f;

        local_sum += receive_contribution(
            tx, e0, rt0, transmit_delay,
            rf_data, t_peak, sampling_frequency
        );

        local_sum += receive_contribution(
            tx, e1, rt1, transmit_delay,
            rf_data, t_peak, sampling_frequency
        );

        local_sum += receive_contribution(
            tx, e2, rt2, transmit_delay,
            rf_data, t_peak, sampling_frequency
        );

        local_sum += receive_contribution(
            tx, e3, rt3, transmit_delay,
            rf_data, t_peak, sampling_frequency
        );

        // -------------------------------------------------
        // 3D. WARP COMBINES ALL 128 CONTRIBUTIONS
        // -------------------------------------------------

        float event_sum = warp_reduce_sum(local_sum);

        // Lane zero owns the complete event sum.
        if (lane == 0)
            pixel_sum += event_sum;
    }

    // -----------------------------------------------------
    // 4. ONE THREAD WRITES THE PIXEL
    // -----------------------------------------------------

    if (lane == 0)
        output[pixel] = pixel_sum;
}

// ---------------------------------------------------------
// STATISTICS
// ---------------------------------------------------------

double calculate_median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());

    size_t n = values.size();

    if (n % 2 == 0)
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;

    return values[n / 2];
}

double calculate_p95(std::vector<double> values)
{
    std::sort(values.begin(), values.end());

    size_t index = static_cast<size_t>(
        std::ceil(0.95 * values.size())
    ) - 1;

    return values[index];
}

void print_statistics(
    const std::string& name,
    const std::vector<double>& values)
{
    std::cout << "\n" << name << ":\n";

    std::cout << "Median: "
              << calculate_median(values)
              << " ms\n";

    std::cout << "P95: "
              << calculate_p95(values)
              << " ms\n";
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        bool full_mode =
            (argc > 1 && std::string(argv[1]) == "--full");

        const int output_count =
            full_mode ? TOTAL_PIXELS : 4096;

        const std::string folder = "data/binary/";

        std::cout << "\n========== CUDA DAS V6 ==========\n";

        std::cout << "Mode: "
                  << (full_mode ? "FULL IMAGE" : "VALIDATION TEST")
                  << "\n";

        std::cout << "Pixels: " << output_count << "\n";
        std::cout << "Threads per block: " << THREADS << "\n";
        std::cout << "Warps per block: " << WARPS_PER_BLOCK << "\n";
        std::cout << "Threads per pixel: " << WARP_SIZE << "\n";
        std::cout << "Apodization: specialized (all values = 1.0)\n";

        // -------------------------------------------------
        // 1. LOAD DATASET
        // -------------------------------------------------

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
        // 2. ALLOCATE GPU MEMORY
        // -------------------------------------------------

        std::cout << "\nAllocating GPU buffers...\n";

        float* d_rf = nullptr;
        float* d_geometry = nullptr;
        float* d_t0 = nullptr;
        float* d_initial = nullptr;
        float* d_peak = nullptr;
        float* d_grid = nullptr;
        float* d_output = nullptr;

        CUDA_CHECK(cudaMalloc(
            (void**)&d_rf, rf.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_geometry, geometry.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_t0, t0.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_initial, initial_times.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_peak, peak.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_grid, grid.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_output, output_count * sizeof(float)
        ));

        std::cout << "GPU buffers allocated successfully.\n";

        // -------------------------------------------------
        // 3. BENCHMARK CONFIGURATION
        // -------------------------------------------------

        int blocks =
            (output_count + WARPS_PER_BLOCK - 1) /
            WARPS_PER_BLOCK;

        std::cout << "\n========== GPU PIPELINE BENCHMARK ==========\n";

        std::cout << "Blocks: " << blocks << "\n";
        std::cout << "Threads per block: " << THREADS << "\n";
        std::cout << "Warm-up runs: " << WARMUP_RUNS << "\n";
        std::cout << "Measured runs: " << BENCHMARK_RUNS << "\n";

        cudaEvent_t start, stop;

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        std::vector<float> gpu_output(output_count);

        std::vector<double> h2d_times;
        std::vector<double> kernel_times;
        std::vector<double> d2h_times;
        std::vector<double> pipeline_times;

        // -------------------------------------------------
        // 4. PIPELINE EXECUTION
        // -------------------------------------------------

        auto run_pipeline = [&](bool measure)
        {
            using Clock = std::chrono::steady_clock;

            auto pipeline_start = Clock::now();

            // H2D: Host to Device
            auto h2d_start = Clock::now();

            CUDA_CHECK(cudaMemcpy(
                d_rf, rf.data(),
                rf.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            CUDA_CHECK(cudaMemcpy(
                d_geometry, geometry.data(),
                geometry.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            CUDA_CHECK(cudaMemcpy(
                d_t0, t0.data(),
                t0.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            CUDA_CHECK(cudaMemcpy(
                d_initial, initial_times.data(),
                initial_times.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            CUDA_CHECK(cudaMemcpy(
                d_peak, peak.data(),
                peak.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            CUDA_CHECK(cudaMemcpy(
                d_grid, grid.data(),
                grid.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            auto h2d_end = Clock::now();

            double h2d_ms =
                std::chrono::duration<double, std::milli>(
                    h2d_end - h2d_start
                ).count();

            // CUDA kernel
            CUDA_CHECK(cudaEventRecord(start));

            das_kernel_v6<<<blocks, THREADS>>>(
                d_rf,
                d_geometry,
                d_t0,
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

            // D2H: Device to Host
            auto d2h_start = Clock::now();

            CUDA_CHECK(cudaMemcpy(
                gpu_output.data(),
                d_output,
                output_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ));

            auto d2h_end = Clock::now();

            double d2h_ms =
                std::chrono::duration<double, std::milli>(
                    d2h_end - d2h_start
                ).count();

            auto pipeline_end = Clock::now();

            double pipeline_ms =
                std::chrono::duration<double, std::milli>(
                    pipeline_end - pipeline_start
                ).count();

            if (measure)
            {
                h2d_times.push_back(h2d_ms);
                kernel_times.push_back(kernel_ms);
                d2h_times.push_back(d2h_ms);
                pipeline_times.push_back(pipeline_ms);
            }
        };

        // -------------------------------------------------
        // 5. WARM-UP
        // -------------------------------------------------

        std::cout << "\nRunning warm-up...\n";

        for (int i = 0; i < WARMUP_RUNS; i++)
            run_pipeline(false);

        // -------------------------------------------------
        // 6. BENCHMARK
        // -------------------------------------------------

        std::cout << "Running benchmark...\n";

        for (int i = 0; i < BENCHMARK_RUNS; i++)
        {
            run_pipeline(true);

            std::cout << "Run "
                      << (i + 1) << "/"
                      << BENCHMARK_RUNS
                      << " completed.\n";
        }

        // -------------------------------------------------
        // 7. LATENCY STATISTICS
        // -------------------------------------------------

        std::cout << std::fixed << std::setprecision(4);

        std::cout << "\n========== PIPELINE LATENCY RESULTS ==========\n";

        print_statistics("H2D transfer", h2d_times);
        print_statistics("CUDA DAS V6 kernel", kernel_times);
        print_statistics("D2H transfer", d2h_times);
        print_statistics("Complete GPU pipeline", pipeline_times);

        // -------------------------------------------------
        // 8. CPU VS GPU VALIDATION
        // -------------------------------------------------

        double squared_error_sum = 0.0;
        double cpu_squared_sum = 0.0;
        double maximum_error = 0.0;

        for (int i = 0; i < output_count; i++)
        {
            int source_pixel;

            if (full_mode)
            {
                source_pixel = i;
            }
            else
            {
                source_pixel = static_cast<int>(
                    (static_cast<long long>(i) *
                     (TOTAL_PIXELS - 1)) /
                    (output_count - 1)
                );
            }

            double cpu_value = cpu_image[source_pixel];
            double gpu_value = gpu_output[i];

            double error = std::abs(gpu_value - cpu_value);

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
        // 9. SAVE OUTPUT
        // -------------------------------------------------

        std::string output_filename =
            full_mode
            ? "results/cuda_das_v6_image.bin"
            : "results/cuda_das_v6_test.bin";

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
        // 10. FINAL RESULTS
        // -------------------------------------------------

        std::cout << "\n========== DAS V6 VALIDATION RESULTS ==========\n";

        std::cout << "Pixels reconstructed: "
                  << output_count << "\n";

        std::cout << std::fixed << std::setprecision(6);

        std::cout << "RMSE: " << rmse << "\n";

        std::cout << "Maximum absolute error: "
                  << maximum_error << "\n";

        std::cout << "Relative RMSE: "
                  << relative_rmse << " %\n";

        std::cout << "Output saved: "
                  << output_filename << "\n";

        // -------------------------------------------------
        // 11. CLEANUP
        // -------------------------------------------------

        CUDA_CHECK(cudaFree(d_rf));
        CUDA_CHECK(cudaFree(d_geometry));
        CUDA_CHECK(cudaFree(d_t0));
        CUDA_CHECK(cudaFree(d_initial));
        CUDA_CHECK(cudaFree(d_peak));
        CUDA_CHECK(cudaFree(d_grid));
        CUDA_CHECK(cudaFree(d_output));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        std::cout << "\nCUDA DAS V6 benchmark completed successfully!\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "\nERROR: "
                  << error.what() << "\n";

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
