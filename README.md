# High-Performance CUDA GPU Acceleration for Ultrasound Delay-and-Sum (DAS) Beamforming

[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20%2B-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![Target Architecture](https://img.shields.io/badge/Architecture-Ampere%20sm__86-blue)](https://developer.nvidia.com/cuda-gpus)
[![License](https://img.shields.io/badge/License-Academic%20%2F%20Research-green.svg)](#license)

High-throughput, power-efficient CUDA acceleration of coherent plane-wave Delay-and-Sum (DAS) ultrasound beamforming for edge and embedded computing platforms. Developed as part of an embedded GPU ultrasound research project for the **Commissariat à l'énergie atomique et aux énergies alternatives (CEA)**.

---

## 1. Executive Summary

Ultrasound Delay-and-Sum (DAS) beamforming across wide synthetic aperture datasets is computationally demanding, requiring the reconstruction of hundreds of thousands of spatial grid points from millions of temporal RF echo samples across multiple steered transmission angles. 

On a standard multi-core CPU, full-field coherent compounding of 75 steered plane-wave transmissions across 128 transducer channels for a $1024 \times 320$ grid takes **~135 seconds** in Python/NumPy ($135,730.02\text{ ms}$). This project achieves real-time embedded reconstruction on an NVIDIA Ampere architecture:
- **Controlled Mobile Power Profile (RTX 3050 Laptop):** **254.90 ms** kernel median latency (**532.5× speedup** over CPU reference; **275.31 ms** full pipeline latency, **493.0× speedup**).
- **Historical Specification Reference (Status: NOT VERIFIED):** ~89.6 ms reference latency from initial task documentation; no raw benchmark log or execution trace exists in this repository.
- **Strict Numerical Parity:** V4, V5, V6, and V7 achieve **bit-identical** mathematical parity ($\Delta_{\text{max}} = 0.0$, $\text{RMSE} = 0.0$), with a relative RMSE of $4.41 \times 10^{-4}$ against the floating-point CPU reference image.

---

## 2. Mathematical Foundation

### Coherent Plane-Wave Delay-and-Sum (DAS) Beamforming

For a 2D Cartesian reconstruction grid point $\vec{p} = (x, z)$ on the imaging plane ($y = 0$), the coherently compounded beamformed signal $S(x, z)$ across $N_{\text{tx}}$ steered plane wave transmissions and $N_{\text{elements}}$ receive elements is defined as:

$$S(x, z) = \sum_{tx=1}^{N_{\text{tx}}} A_{tx} \sum_{e=1}^{N_{\text{elements}}} a_{rx}(x, z, e) \cdot \tilde{s}_{\text{rf}}\left(tx, u(x, z, tx, e), e\right)$$

Where:
- $u(x, z, tx, e) = \tau(x, z, tx, e) \cdot f_s \in \mathbb{R}$ is the continuous temporal sample coordinate corresponding to acoustic time-of-flight $\tau$.
- $\tilde{s}_{\text{rf}}(tx, u, e)$ is the sub-sample linearly interpolated RF echo amplitude evaluated at coordinate $u$.
- In memory, the discrete sampled tensor $s_{\text{rf}}$ has shape $(75, 1527, 128)$ corresponding to $(N_{\text{tx}}, N_{\text{samples}}, N_{\text{elements}})$, with linear C-order index $(tx \times N_{\text{samples}} + sample) \times N_{\text{elements}} + element$. The receive element dimension is contiguous with unit stride.
- $A_{tx}$ is the transmission apodization weight.
- $a_{rx}(x, z, e)$ is dynamic receive apodization (evaluated at $f\text{-number} = 0$ for full-aperture compounding; all 128 receive elements participate).
- $\tau(x, z, tx, e)$ is the total two-way acoustic time-of-flight.

### Time-of-Flight Calculation

$$\tau(x, z, tx, e) = \tau_{\text{tx}}(x, z, tx) + \tau_{\text{rx}}(x, z, e) + t_0(tx, e) + t_{\text{peak}} - t_{\text{init}}(tx)$$

1. **Forward Transmit Delay (Plane Wave at Steering Angle $\alpha_{\text{steer}}$):**
   $$\tau_{\text{tx}}(x, z, tx) = \frac{x \sin \alpha_{\text{steer}} + z \cos \alpha_{\text{steer}}}{c}$$
2. **Backward Receive Delay (Spherical Echo to Element $(x_e, z_e)$):**
   $$\tau_{\text{rx}}(x, z, e) = \frac{\sqrt{(x - x_e)^2 + (z - z_e)^2}}{c}$$
3. **Temporal Offsets:**
   - $t_0(tx, e)$: Transmit element delay offset.
   - $t_{\text{peak}} = \frac{1}{f_c}$: Transmit pulse envelope peak offset ($f_c = 5.208\text{ MHz} \implies t_{\text{peak}} \approx 1.92 \times 10^{-7}\text{ s}$). Assumed analytical convention generated via the ZEA beamforming library default ($1/f_c$); not an empirical pulse-peak parameter from PICMUS HDF5 metadata.
   - $t_{\text{init}}(tx)$: Receiver acquisition start time per transmission.
   - $c = 1540.0\text{ m/s}$: Acoustic propagation velocity.

### Sub-Sample Linear Interpolation

Because arrival time $\tau$ corresponds to a continuous (fractional) temporal sample coordinate $u = \tau \cdot f_s \in \mathbb{R}$, the value $\tilde{s}_{\text{rf}}(tx, u, e)$ is reconstructed via linear interpolation between adjacent integer sample indices $i$ and $i+1$:

$$i = \lfloor u \rfloor, \qquad \alpha = u - i \quad (\alpha \in [0, 1))$$

$$\tilde{s}_{\text{rf}}(tx, u, e) = (1 - \alpha) \cdot s_{\text{rf}}[tx, i, e] + \alpha \cdot s_{\text{rf}}[tx, i+1, e]$$

Boundary checks (`i >= 0 && i < N_SAMPLES - 1`) guarantee that out-of-range temporal samples contribute zero to the accumulated sum.

---

## 3. Repository Architecture

```
CUDA_Ultrasound/
├── CMakeLists.txt                 # Multi-target build configuration (sm_86, C++17, CUDA 17)
├── requirements.txt               # Verified Python dependencies (NumPy, SciPy, Matplotlib, zea)
├── REFERENCE_AND_VALIDATION.md    # Formal CEA internship scientific & benchmark documentation
├── .gitignore                     # Repository hygiene (excludes binaries, large datasets, caches)
│
├── src/
│   ├── kernels/                   # Progressive CUDA implementations (V1 through V7)
│   │   ├── cuda_das_v1.cu         # V1: Naive one-thread-per-pixel baseline
│   │   ├── cuda_das_v2.cu         # V2: Precomputed receive-times global-memory buffer experiment (160 MB)
│   │   ├── cuda_das_v3.cu         # V3: Per-thread 128-float local array on stack frame (512 B local memory)
│   │   ├── cuda_das_v4.cu         # V4: Warp-cooperative DAS (4 elements/lane, shuffle-down reduction)
│   │   ├── cuda_das_v5.cu         # V5: Warp XOR butterfly all-reduce reduction network (__shfl_xor_sync)
│   │   ├── cuda_das_v6.cu         # V6: Transmit-apodization specialization (removes apodization branch/loads)
│   │   ├── cuda_das_v7.cu         # V7: V6 with three explicit __syncwarp(0xffffffffu) barrier points
│   │   ├── cuda_check.cu          # CUDA runtime and Ampere SM capability inspector
│   │   └── cuda_load_test.cu      # Standalone global memory streaming benchmark
│   │
│   └── scripts/                   # Verification, profiling, and comparison tools
│       ├── export_data.py         # Extracts picmus_cuda_input.npz into flat binary buffers
│       ├── inspect_data.py        # Inspects binary dimensions, types, and byte offsets
│       ├── compare_all_versions.py# Cross-version validation vs CPU reference image
│       ├── compare_v4_v7.py       # Numerical bit-parity test between V4 and V7
│       ├── compare_ptx.py         # PTX instruction parser and register/memory comparison
│       ├── compare_sass.py        # SASS assembly instruction analysis (LDG.E, BSYNC)
│       ├── visualize_results.py   # B-mode reconstruction visualization for V4
│       └── visualize_v7.py        # B-mode reconstruction visualization for V7
│
├── notebooks/                     # Exploratory physics & reference CPU implementations
│   ├── 01_cpu_das_reference_pipeline.ipynb        # Cleaned CPU reference pipeline & PICMUS fetch
│   └── 02_ultrasound_das_physics_prototyping.ipynb# Physics prototyping and geometry exploration
│
├── data/
│   ├── README.md                  # Comprehensive dataset specification, schema, and reproduction guide
│   └── binary/                    # Flat raw input buffers (geometry, delays, reference image)
│
└── results/
    ├── figures/                   # Rendered B-mode comparisons and error maps
    ├── profiling/                 # SASS, PTX, resource usage logs, and Nsight profiles
    └── logs/                      # Raw benchmark logs and timing measurements
```

---

## 4. CUDA Optimization Progression (V1 – V7)

Every implementation preserves identical physical mathematics and grid dimensions ($1024 \times 320 = 327,680$ pixels, 75 transmits, 128 receive elements):

| Version | Key Architectural Concept | Registers / Thread | Local Memory / Spills | Controlled Mobile Kernel Median | Key Performance Insight |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **V1** | Naive one-thread-per-pixel baseline | 33 | 0 bytes | *Not in controlled comparison* | Monolithic prototype; compute-bound by $19,200\text{ sqrtf}$ evaluations/pixel. |
| **V2** | Precomputed receive-times global memory buffer | K1: 16<br>K2: 40 | 0 bytes | *Not in controlled comparison* | Precomputes $327,680 \times 128$ receive times in 160 MB global memory; high DRAM footprint. |
| **V3** | Per-thread local array (stack frame) | 31 | 512 B stack (0 B spill) | *Not in controlled comparison* | 128-float dynamically indexed array placed on thread stack frame in local memory; high latency. |
| **V4** | Warp-cooperative DAS (4 elements/lane) | 32 | 0 bytes | **289.7211 ms** | **Warp-Cooperative Baseline Reference.** Tree reduction (`__shfl_down_sync`); evaluates transmit apodization branch. |
| **V5** | Warp XOR butterfly all-reduce | 34 | 0 bytes | *Not in controlled comparison* | Tests `__shfl_xor_sync` all-reduce network; yields parity with V4 tree reduction. |
| **V6** | Transmit apodization specialization | 37 | 0 bytes | *Not in controlled comparison* | **Severe Regression:** Removed apodization check/loads; triggered unphased load burst. |
| **V7** | Synchronized warp staging (`__syncwarp`) | 29 | 0 bytes | **254.8951 ms** | **Best-Performing Evaluated Kernel:** Inserts 3 warp barriers; restores load phasing; **12.02% kernel speedup**. |

### Technical Deep Dive: The V6 Regression & The V7 Fix

In **V4**, receive contributions were guarded by a transmit apodization branch:
```cuda
int apod_index = tx * N_ELEMENTS + element;
float apodization = tx_apodizations[apod_index];
if (apodization <= 0.0f) return 0.0f;
...
return interpolated * apodization;
```
In **V6**, this transmit apodization read and conditional return were eliminated because for the PICMUS dataset all transmit apodization weights equal 1.0. **V6 retains the temporal sample bounds check (`if (i0 < 0 || i1 >= N_SAMPLES) return 0.0f;`) unchanged.**

However, removing the apodization branch eliminated compiler stage boundaries between the four receive contribution phases. In Nsight Compute profiling and PTX/SASS inspection:
- **Measured Profiling Observations:** DRAM read traffic surged from $72.5\text{ MB}$ (V4) to $1,250.0\text{ MB}$ (V6, a $17.3\times$ inflation). L1 data cache hit rate dropped from $75.4\%$ to $21.8\%$, and Memory Input/Output (MIO) pipeline throttling caused **$44.5\%$ of active warp stall cycles**.
- **Instruction Scheduling Interpretation:** Supported by PTX and SASS disassembly, the compiler hoisted independent global memory loads (`LDG.E`) into an unphased burst across the loop body. This interpretation accounts for the observed MIO queue congestion and premature L1/L2 cache line eviction.

In **V7**, sequential execution phasing was restored by inserting three explicit warp synchronization barriers between the four `receive_contribution()` calls:
```cuda
local_sum += receive_contribution(tx, e0, rt0, ...);
__syncwarp(0xffffffffu);
local_sum += receive_contribution(tx, e1, rt1, ...);
__syncwarp(0xffffffffu);
local_sum += receive_contribution(tx, e2, rt2, ...);
__syncwarp(0xffffffffu);
local_sum += receive_contribution(tx, e3, rt3, ...);
```

#### PTX and SASS Instruction Verification
- **PTX Disassembly:** In `results/profiling/v7.ptx`, this source construct emits exactly **three** `bar.warp.sync -1;` instructions (lines 232, 258, and 283), where `-1` represents mask `0xffffffffu`. In `v4.ptx` and `v6.ptx`, there are **zero** `bar.` instructions.
- **SASS Disassembly:** In `results/profiling/v7_sass.txt`, hardware barriers and branch tokens are managed using `BSYNC`/`BSSY`. Total SASS `BSYNC` instructions increase from 11 in V6 to 15 in V7 (+4 compiler-scheduled synchronization points).
- **Result:** Under identical mobile power conditions, V7 dropped register pressure to **29 registers**, eliminated MIO throttle stalls, and delivered a **12.02% kernel speedup** (and **11.22% full-pipeline speedup**) over the V4 baseline.

---

## 5. Controlled Benchmark Results (Mobile Power Profile Baseline)

Benchmarked on an **NVIDIA GeForce RTX 3050 Laptop GPU** (Ampere `sm_86`, 16 SMs, 2048 CUDA Cores) operating under controlled mobile power constraints (observed clocks: Graphics $\approx 1730\text{ MHz}$, Memory $\approx 810\text{ MHz}$; 2 warm-up runs, 20 measured iterations via `cudaEventElapsedTime`):

| Implementation | Kernel Median (ms) | Kernel P95 (ms) | Pipeline Median (ms) | Pipeline P95 (ms) | Speedup vs V4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **V4 Baseline** | 289.7211 | 289.9589 | 310.1110 | 310.6784 | Baseline |
| **V7 Synchronized**| **254.8951** | **255.1807** | **275.3078** | **276.6326** | **+12.02% (Kernel)** <br> **+11.22% (Pipeline)** |

> [!NOTE]
> - **Full pipeline latency** includes PCIe Host-to-Device transfer of static parameters and RF data (~19.2 ms), kernel execution, and Device-to-Host transfer of the reconstructed $1024 \times 320$ image buffer (~0.57 ms).
> - V1, V2, V3, V5, and V6 are omitted from this controlled benchmark table because their historical test logs were obtained under differing operating profiles. See [`REFERENCE_AND_VALIDATION.md`](REFERENCE_AND_VALIDATION.md) for individual version analyses.

---

## 6. Numerical Validation & Accuracy

Numerical accuracy is verified across all $327,680$ pixels against the double/single-precision CPU reference generated from the PICMUS simulation dataset:

| Metric | CUDA V4 vs CPU Reference | CUDA V7 vs CPU Reference | CUDA V4 vs CUDA V7 |
| :--- | :---: | :---: | :---: |
| **Root-Mean-Square Error (RMSE)** | $1.15 \times 10^{-4}$ | $1.15 \times 10^{-4}$ | **0.000000** |
| **Maximum Absolute Error** | $3.29 \times 10^{-2}$ | $3.29 \times 10^{-2}$ | **0.000000** |
| **Relative RMSE** | $4.41 \times 10^{-4}$ | $4.41 \times 10^{-4}$ | **0.000000** |
| **Pearson Correlation ($r$)** | $> 0.99999$ | $> 0.99999$ | **1.000000** |
| **Bit-Identity Check** | — | — | **100% Bit-Identical** |

CUDA V4 and V7 produce **identical output bytes**, proving that warp synchronization optimizes instruction scheduling without altering numerical precision.

---

## 7. Build and Run Instructions

### Prerequisites

- **NVIDIA GPU:** Compute Capability $\ge 8.6$ (Ampere architecture, e.g., RTX 3050/3060/A2000, Jetson Orin).
- **CUDA Toolkit:** Version 11.8 or 12.x (`nvcc`).
- **Host Compiler:** C++17 compliant compiler (MSVC 2019/2022 on Windows, GCC 9+ on Linux).
- **Build System:** CMake 3.20+.
- **Python:** Version 3.9+ with packages from `requirements.txt`.

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Build with CMake
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
Compiled executables will be placed in `bin/` (`cuda_das_v1.exe` through `cuda_das_v7.exe`).

### Step 3: Direct Compilation with `nvcc`
To compile the best-performing V7 kernel directly:
```bash
nvcc -O3 -arch=sm_86 -Xptxas=-v src/kernels/cuda_das_v7.cu -o bin/cuda_das_v7.exe
```

### Step 4: Run Reconstruction & Validation
1. **Execute V7 Reconstruction:**
   ```bash
   ./bin/cuda_das_v7.exe
   ```
2. **Verify Bit Parity with V4:**
   ```bash
   python src/scripts/compare_v4_v7.py
   ```
3. **Generate B-Mode Visualizations & Error Maps:**
   ```bash
   python src/scripts/visualize_v7.py
   ```

---

## 8. Milestone Roadmap

- [x] **Milestone 1 — High-Throughput Coherent DAS Beamforming:** Full 75-transmission plane-wave compounding on GPU with verified bit-parity and sub-second execution on a mobile power profile.
- [ ] **Milestone 2 — Integrated In-Kernel B-Mode Processing:** GPU implementation of the analytic signal computation (Hilbert transform / envelope detection), logarithmic dynamic compression, and display clipping directly in device memory.
- [ ] **Milestone 3 — Embedded Zero-Copy & Streaming Architecture:** Integration of CUDA Unified Memory, pinned host-device streaming buffers, and deployment benchmarking on NVIDIA Jetson platforms.

---

## 9. References & Acknowledgments

- **PICMUS Benchmark:** H. Liebgott et al., *"Plane wave imaging challenge in medical ultrasound,"* IEEE International Ultrasonics Symposium (IUS), 2016.
- **Validation Framework:** CEA LIST Embedded Ultrasound Laboratory & `zea` beamforming library (`zeahub/picmus`).
- For complete technical documentation and audit logs, see [`REFERENCE_AND_VALIDATION.md`](REFERENCE_AND_VALIDATION.md) and [`data/README.md`](data/README.md).
