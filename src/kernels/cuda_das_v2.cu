
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

// ---------------------------------------------------------
// CUDA ERROR CHECK
// ---------------------------------------------------------

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

// ---------------------------------------------------------
// DATASET DIMENSIONS
// ---------------------------------------------------------

constexpr int N_TX = 75;
constexpr int N_SAMPLES = 1527;
constexpr int N_ELEMENTS = 128;

constexpr int IMAGE_WIDTH = 320;
constexpr int IMAGE_HEIGHT = 1024;
constexpr int TOTAL_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;

// ---------------------------------------------------------
// READ BINARY FILE
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
// KERNEL 1: PRECOMPUTE RECEIVE TIMES
//
// One CUDA thread calculates one pixel-element pair.
//
// Storage layout:
// receive_times[element * total_pixels + pixel]
//
// This calculation is independent of the transmission.
// Therefore, it is performed only once.
// ---------------------------------------------------------

__global__ void precompute_receive_times_kernel(
    const float* probe_geometry,
    const float* flatgrid,
    float* receive_times,
    float sound_speed,
    int total_pixels,
    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_pairs = total_pixels * num_elements;

    if (idx >= total_pairs)
        return;

    // Identify pixel and receiving element
    int element = idx / total_pixels;
    int pixel = idx % total_pixels;

    // Pixel coordinates
    float px = flatgrid[pixel * 3 + 0];
    float py = flatgrid[pixel * 3 + 1];
    float pz = flatgrid[pixel * 3 + 2];

    // Receiving element coordinates
    int geometry_index = element * 3;

    float ex = probe_geometry[geometry_index + 0];
    float ey = probe_geometry[geometry_index + 1];
    float ez = probe_geometry[geometry_index + 2];

    // Euclidean distance
    float dx = px - ex;
    float dy = py - ey;
    float dz = pz - ez;

    float distance = sqrtf(
        dx * dx + dy * dy + dz * dz
    );

    // Receive propagation time
    float receive_time = distance / sound_speed;

    // Store result
    receive_times[
        element * total_pixels + pixel
    ] = receive_time;
}

// ---------------------------------------------------------
// KERNEL 2: OPTIMIZED CUDA DAS
//
// One CUDA thread reconstructs one image pixel.
//
// Receive times are read from precomputed GPU memory.
// ---------------------------------------------------------

__global__ void das_kernel(
    const float* rf_data,
    const float* t0_delays,
    const float* tx_apodizations,
    const float* initial_times,
    const float* t_peak,
    const float* receive_times,
    float* output,
    float sampling_frequency,
    int output_count,
    int total_pixels)
{
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;

    if (pixel >= output_count)
        return;

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

    float pixel_sum = 0.0f;

    // Process all 75 transmissions
    for (int tx = 0; tx < N_TX; tx++) {

        float minimum_tx_time = 1.0e30f;

        // ---------------------------------------------
        // 1. TRANSMIT DELAY
        // ---------------------------------------------

        for (int element = 0;
             element < N_ELEMENTS;
             element++)
        {
            int apod_index = tx * N_ELEMENTS + element;

            float apodization =
                tx_apodizations[apod_index];

            if (apodization <= 0.0f)
                continue;

            // Reuse precomputed receive time
            float receive_time =
                receive_times[
                    element * total_pixels + source_pixel
                ];

            float candidate =
                receive_time + t0_delays[apod_index];

            if (candidate < minimum_tx_time) {
                minimum_tx_time = candidate;
            }
        }

        if (minimum_tx_time == 1.0e30f)
            continue;

        float transmit_delay =
            minimum_tx_time - initial_times[tx];

        // ---------------------------------------------
        // 2. RECEIVE DELAYS AND RF INTERPOLATION
        // ---------------------------------------------

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

            // Same precomputed receive time
            float receive_time =
                receive_times[
                    element * total_pixels + source_pixel
                ];

            float total_time =
                transmit_delay +
                receive_time +
                t_peak[tx];

            float sample_position =
                total_time * sampling_frequency;

            int i0 = static_cast<int>(
                floorf(sample_position)
            );

            int i1 = i0 + 1;

            if (i0 < 0 || i1 >= N_SAMPLES)
                continue;

            // Linear interpolation
            float alpha = sample_position - i0;

            int index0 =
                (tx * N_SAMPLES + i0) * N_ELEMENTS + element;

            int index1 =
                (tx * N_SAMPLES + i1) * N_ELEMENTS + element;

            float rf0 = rf_data[index0];
            float rf1 = rf_data[index1];

            float interpolated =
                rf0 + alpha * (rf1 - rf0);

            event_sum += interpolated * apodization;
        }

        pixel_sum += event_sum;
    }

    output[pixel] = pixel_sum;
}

// ---------------------------------------------------------
// STATISTICS
// ---------------------------------------------------------

double calculate_median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());

    size_t n = values.size();

    if (n % 2 == 0) {
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    }

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

// ---------------------------------------------------------
// PRINT STATISTICS
// ---------------------------------------------------------

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
    try {

        bool full_mode =
            (argc > 1 && std::string(argv[1]) == "--full");

        const int output_count =
            full_mode ? TOTAL_PIXELS : 4096;

        constexpr int THREADS = 256;
        constexpr int WARMUP_RUNS = 2;
        constexpr int BENCHMARK_RUNS = 20;

        std::cout << "\n========== OPTIMIZED CUDA DAS ==========\n";

        std::cout << "Mode: "
                  << (full_mode ? "FULL IMAGE" : "VALIDATION TEST")
                  << "\n";

        std::cout << "Pixels: " << output_count << "\n";

        // -------------------------------------------------
        // 1. LOAD DATASET
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
            folder + "initial_times.bin",
            N_TX
        );

        auto peak = load_binary<float>(
            folder + "t_peak.bin",
            N_TX
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
            folder + "sampling_frequency.bin",
            1
        );

        auto sound_speed = load_binary<float>(
            folder + "sound_speed.bin",
            1
        );

        // -------------------------------------------------
        // 2. ALLOCATE GPU MEMORY
        // -------------------------------------------------

        std::cout << "\nAllocating GPU buffers...\n";

        float* d_rf = nullptr;
        float* d_geometry = nullptr;
        float* d_t0 = nullptr;
        float* d_apod = nullptr;
        float* d_initial = nullptr;
        float* d_peak = nullptr;
        float* d_grid = nullptr;
        float* d_receive_times = nullptr;
        float* d_output = nullptr;

        size_t receive_count =
            static_cast<size_t>(TOTAL_PIXELS) * N_ELEMENTS;

        size_t receive_bytes =
            receive_count * sizeof(float);

        CUDA_CHECK(cudaMalloc(
            (void**)&d_rf,
            rf.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_geometry,
            geometry.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_t0,
            t0.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_apod,
            apodization.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_initial,
            initial_times.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_peak,
            peak.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_grid,
            grid.size() * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_receive_times,
            receive_bytes
        ));

        CUDA_CHECK(cudaMalloc(
            (void**)&d_output,
            output_count * sizeof(float)
        ));

        std::cout << "GPU buffers allocated successfully.\n";

        std::cout << "Receive-time buffer: "
                  << receive_bytes / (1024.0 * 1024.0)
                  << " MiB\n";

        // -------------------------------------------------
        // 3. INITIALIZATION: UPLOAD STATIC GEOMETRY
        // -------------------------------------------------

        std::cout << "\nUploading probe geometry and image grid...\n";

        CUDA_CHECK(cudaMemcpy(
            d_geometry,
            geometry.data(),
            geometry.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_grid,
            grid.data(),
            grid.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        // -------------------------------------------------
        // 4. PRECOMPUTE RECEIVE TIMES ONCE
        // -------------------------------------------------

        std::cout << "\n========== RECEIVE-TIME PRECOMPUTATION ==========\n";

        constexpr int PRECOMPUTE_THREADS = 256;

        int total_pairs =
            TOTAL_PIXELS * N_ELEMENTS;

        int precompute_blocks =
            (total_pairs + PRECOMPUTE_THREADS - 1)
            / PRECOMPUTE_THREADS;

        std::cout << "Pixel-element pairs: "
                  << total_pairs << "\n";

        std::cout << "Precomputation blocks: "
                  << precompute_blocks << "\n";

        cudaEvent_t precompute_start;
        cudaEvent_t precompute_stop;

        CUDA_CHECK(cudaEventCreate(&precompute_start));
        CUDA_CHECK(cudaEventCreate(&precompute_stop));

        CUDA_CHECK(cudaEventRecord(precompute_start));

        precompute_receive_times_kernel
            <<<precompute_blocks, PRECOMPUTE_THREADS>>>(
                d_geometry,
                d_grid,
                d_receive_times,
                sound_speed[0],
                TOTAL_PIXELS,
                N_ELEMENTS
            );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(precompute_stop));
        CUDA_CHECK(cudaEventSynchronize(precompute_stop));

        float precompute_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &precompute_ms,
            precompute_start,
            precompute_stop
        ));

        std::cout << "Precomputation time: "
                  << precompute_ms << " ms\n";

        std::cout << "Receive times ready for reuse.\n";

        // -------------------------------------------------
        // 5. BENCHMARK CONFIGURATION
        // -------------------------------------------------

        int blocks =
            (output_count + THREADS - 1) / THREADS;

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
        // 6. PIPELINE EXECUTION
        //
        // Static geometry/grid and receive times are
        // initialized once, outside the per-frame pipeline.
        //
        // The RF data and transmission parameters are
        // transferred for each reconstruction.
        // -------------------------------------------------

        auto run_pipeline = [&](bool measure)
        {
            using Clock = std::chrono::steady_clock;

            auto pipeline_start = Clock::now();

            // H2D: Host to Device
            auto h2d_start = Clock::now();

            CUDA_CHECK(cudaMemcpy(
                d_rf,
                rf.data(),
                rf.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            CUDA_CHECK(cudaMemcpy(
                d_t0,
                t0.data(),
                t0.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            CUDA_CHECK(cudaMemcpy(
                d_apod,
                apodization.data(),
                apodization.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            CUDA_CHECK(cudaMemcpy(
                d_initial,
                initial_times.data(),
                initial_times.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            CUDA_CHECK(cudaMemcpy(
                d_peak,
                peak.data(),
                peak.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            auto h2d_end = Clock::now();

            double h2d_ms =
                std::chrono::duration<double, std::milli>(
                    h2d_end - h2d_start
                ).count();

            // DAS kernel execution
            CUDA_CHECK(cudaEventRecord(start));

            das_kernel<<<blocks, THREADS>>>(
                d_rf,
                d_t0,
                d_apod,
                d_initial,
                d_peak,
                d_receive_times,
                d_output,
                fs[0],
                output_count,
                TOTAL_PIXELS
            );

            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float kernel_ms = 0.0f;

            CUDA_CHECK(cudaEventElapsedTime(
                &kernel_ms,
                start,
                stop
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

            if (measure) {
                h2d_times.push_back(h2d_ms);
                kernel_times.push_back(kernel_ms);
                d2h_times.push_back(d2h_ms);
                pipeline_times.push_back(pipeline_ms);
            }
        };

        // -------------------------------------------------
        // 7. WARM-UP
        // -------------------------------------------------

        std::cout << "\nRunning warm-up...\n";

        for (int i = 0; i < WARMUP_RUNS; i++) {
            run_pipeline(false);
        }

        // -------------------------------------------------
        // 8. MEASURED RUNS
        // -------------------------------------------------

        std::cout << "Running benchmark...\n";

        for (int i = 0; i < BENCHMARK_RUNS; i++) {

            run_pipeline(true);

            std::cout << "Run "
                      << (i + 1) << "/"
                      << BENCHMARK_RUNS
                      << " completed.\n";
        }

        // -------------------------------------------------
        // 9. STATISTICS
        // -------------------------------------------------

        std::cout << std::fixed << std::setprecision(4);

        std::cout << "\n========== PIPELINE LATENCY RESULTS ==========\n";

        std::cout << "One-time precomputation: "
                  << precompute_ms << " ms\n";

        print_statistics("H2D transfer", h2d_times);
        print_statistics("Optimized CUDA DAS kernel", kernel_times);
        print_statistics("D2H transfer", d2h_times);
        print_statistics("Per-frame GPU pipeline", pipeline_times);

        // -------------------------------------------------
        // 10. CPU VS GPU VALIDATION
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
        // 11. SAVE OUTPUT
        // -------------------------------------------------

        std::string output_filename =
            full_mode
            ? "results/cuda_das_optimized_image.bin"
            : "results/cuda_das_optimized_test.bin";

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
        // 12. FINAL RESULTS
        // -------------------------------------------------

        std::cout << "\n========== DAS VALIDATION RESULTS ==========\n";

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
        // 13. CLEANUP
        // -------------------------------------------------

        CUDA_CHECK(cudaFree(d_rf));
        CUDA_CHECK(cudaFree(d_geometry));
        CUDA_CHECK(cudaFree(d_t0));
        CUDA_CHECK(cudaFree(d_apod));
        CUDA_CHECK(cudaFree(d_initial));
        CUDA_CHECK(cudaFree(d_peak));
        CUDA_CHECK(cudaFree(d_grid));
        CUDA_CHECK(cudaFree(d_receive_times));
        CUDA_CHECK(cudaFree(d_output));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        CUDA_CHECK(cudaEventDestroy(precompute_start));
        CUDA_CHECK(cudaEventDestroy(precompute_stop));

        std::cout << "\nOptimized CUDA DAS completed successfully!\n";
    }
    catch (const std::exception& error) {

        std::cerr << "\nERROR: "
                  << error.what() << "\n";

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
