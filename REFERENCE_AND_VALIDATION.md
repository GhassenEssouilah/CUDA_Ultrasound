# Ultrasound Delay-and-Sum (DAS) Beamforming: Reference and Validation

**Document ID:** CEA-TECH-DAS-VAL-2026-11  
**Target Environment:** Embedded Ultrasound Acceleration (CEA Internship Project)  
**Date:** November 13, 2026  
**Status:** Approved Technical Baseline  

---

## 1. Project Objective and Dataset Description

### 1.1 Project Objective
This project develops and characterizes a high-throughput, power-efficient GPU acceleration pipeline for ultrasound Delay-and-Sum (DAS) beamforming tailored for edge and embedded computing platforms (e.g., embedded NVIDIA GPU architectures). The central goal is to transition ultrasound reconstruction from multi-second CPU execution to sub-second real-time imaging while maintaining strict numerical precision, low register pressure, and bounded memory bandwidth.

### 1.2 PICMUS Benchmark Dataset
The dataset utilized is from the **Plane Wave Imaging Challenge in Medical Ultrasound (PICMUS)** benchmark, presented at the IEEE International Ultrasonics Symposium (IUS) 2016. Specifically, this implementation processes the synthetic **numerical resolution and distortion phantom** (`resolution_distorsion_simu_dataset_rf.hdf5`) retrieved from the open-access `zeahub/picmus` repository.

The dataset provides raw Radio-Frequency (RF) echo matrices acquired from a simulated medium containing point scatterers and cyst targets designed to evaluate lateral resolution, axial resolution, and contrast ratio.

---

## 2. Acquisition Parameters and Grid Configuration

All physical dimensions, temporal parameters, and array coordinates are loaded from dataset metadata:

| Parameter | Symbol | Value | Physical Unit | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Transmissions** | $N_{\text{tx}}$ | 75 | — | Steered plane-wave transmits ($\theta \in [-16^\circ, +16^\circ]$) |
| **Temporal Samples** | $N_{\text{samples}}$ | 1,527 | — | Time samples acquired per receive channel |
| **Receive Elements** | $N_{\text{elements}}$ | 128 | — | Transducer elements participating in reception |
| **RF Data Matrix** | — | $75 \times 1,527 \times 128$ | `float32` | 14,659,200 total temporal samples (58.64 MB) |
| **Sampling Frequency** | $f_s$ | 20.832 | MHz | Receiver analog-to-digital sampling frequency |
| **Center Frequency** | $f_c$ | 5.208 | MHz | Nominal transducer carrier frequency |
| **Speed of Sound** | $c$ | 1540.0 | m/s | Assumed constant acoustic propagation velocity |
| **Array Pitch** | $p$ | 0.300 | mm | Center-to-center element spacing |
| **Element Width** | $w$ | 0.270 | mm | Individual element width |
| **Array Aperture** | — | 38.10 | mm | Physical probe span: $x \in [-19.05, +19.05]$ mm |

### Exploratory Image Grid Resolution and Aliasing Check
The Cartesian reconstruction grid is defined over the physical imaging plane ($y = 0$):
- **Lateral Range ($X$):** $[-19.20\text{ mm}, +19.20\text{ mm}]$ sampled at 320 points ($\Delta x = 0.12038\text{ mm}$).
- **Axial Range ($Z$):** $[5.00\text{ mm}, 50.00\text{ mm}]$ sampled at 1,024 points ($\Delta z = 0.04399\text{ mm}$).
- **Total Reconstruction Grid:** $320 \times 1,024 = 327,680\text{ pixels}$ (`flatgrid.bin`: $327,680 \times 3\text{ floats} = 3.93\text{ MB}$).

In exploratory notebook analysis (`Untitled3.ipynb`), spatial resolution was cross-checked using `zea.beamform.pixelgrid.aliasing_limits`, which computes idealized single-frequency spatial Nyquist limits:
$$\Delta x_{\text{max}} = \frac{c}{2 f_{\text{max}} \sin \theta_{\text{max}}} \approx 0.148\text{ mm}, \qquad \Delta z_{\text{max}} = \frac{c}{4 f_{\text{max}}} \approx 0.074\text{ mm}$$
The selected grid ($\Delta x \approx 0.120\text{ mm}$, $\Delta z \approx 0.044\text{ mm}$) provides a fine spatial discretization satisfying these exploratory guidelines.

---

## 3. CPU DAS Reference Methodology and Runtime

### 3.1 Algorithmic Formulation
The CPU reference implementation (`das_cpu`) is defined in `notebooks/Untitled3.ipynb`. It reconstructs the image using vectorized block chunking (chunk size = 4,096 pixels):

1. **Receive Travel Time ($\tau_{\text{rx}}$):** For pixel $\mathbf{p}_i = (x_i, 0, z_i)$ and receiving element $\mathbf{e}_j = (e_{x,j}, 0, e_{z,j})$:
   $$\tau_{\text{rx}}(i, j) = \frac{\|\mathbf{p}_i - \mathbf{e}_j\|_2}{c} = \frac{\sqrt{(x_i - e_{x,j})^2 + z_i^2}}{c}$$

2. **Transmit Delay ($\tau_{\text{tx}}$):** Under plane-wave transmission, wavefront arrival follows the minimum propagation delay across active transmit channels:
   $$\tau_{\text{tx}}(i) = \min_{j \in \text{active}} \left( \tau_{\text{rx}}(i, j) + t_0(tx, j) \right) - t_{\text{init}}(tx)$$

3. **Total Propagation Delay:**
   $$\tau_{\text{total}}(i, j) = \tau_{\text{tx}}(i) + t_{\text{peak}}(tx) + \tau_{\text{rx}}(i, j)$$

4. **Continuous Indexing & Sub-Sample Linear Interpolation:**
   $$s(i, j) = \tau_{\text{total}}(i, j) \times f_s, \qquad s_0 = \lfloor s \rfloor, \quad \alpha = s - s_0$$
   $$\text{RF}_{\text{interp}}(i, j) = (1 - \alpha) \cdot \text{RF}(tx, s_0, j) + \alpha \cdot \text{RF}(tx, s_0 + 1, j)$$

5. **Coherent Compounding:**
   $$I_{\text{DAS}}(\mathbf{p}_i) = \sum_{tx=0}^{74} \sum_{j=0}^{127} \text{RF}_{\text{interp}}(i, j) \cdot A_{\text{tx}}(tx, j)$$

### 3.2 Reference Execution Profile
- **Environment:** Intel Xeon / Colab Python 3.10, NumPy vectorized execution.
- **CPU Runtime:** **$135.730019\text{ seconds}$** ($2.26\text{ minutes}$) for 327,680 pixels.
- **Output Statistics:**
  - Range: $[-2431.3804, +2666.4521]$
  - Mean: $-0.003133$
  - Standard Deviation: $35.4747$
  - NaN / Inf Count: 0 / 0
- **Storage:** Persisted bit-identically in `data/picmus_cuda_input.npz` (`cpu_das_image`) and `data/binary/cpu_das_image.bin` ($1,310,720\text{ bytes}$).

---

## 4. Independent Cross-Software Verification (ZEA Library)

To establish that the custom vectorized Python implementation is mathematically sound, an **independent cross-software verification** was performed against the peer-reviewed **ZEA** ultrasound beamforming library (`zea.beamform.beamformer.tof_correction`).

- **Scope:** 512 evaluation pixels distributed uniformly across the entire physical field of view ($x \in [-19.2, +19.2]$ mm, $z \in [5, 50]$ mm).
- **Execution:** Identical input tensors ($t_0$, apodization, geometry, $t_{\text{peak}}$, $f_{\#}=0.0$).
- **Measured Error Metrics:**
  - Root Mean Square Error (RMSE): **$0.0002659989$**
  - Maximum Absolute Error: **$0.0020295382$**
  - Relative RMSE: **$0.0233207\%$**

*Technical Distinction:* This step constitutes an independent validation of algorithmic correctness against an established third-party library. It is strictly separate from the downstream CUDA-versus-CPU implementation fidelity checks.

---

## 5. CUDA V1–V7 Optimization and Architectural History

Seven distinct CUDA implementations were developed and audited:

| Version | Source File | Thread Mapping | Kernel Mechanism | Registers / Thread | Stack Frame / Local Mem | Status & Performance Impact |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **V1** | `cuda_das.cu` | 1 thread = 1 pixel | Monolithic kernel; redundant Euclidean distance recomputation twice per element per transmit. | 33 | 0 B | Baseline naive prototype. Compute-bound by $19,200\text{ sqrtf}$ evaluations/pixel. |
| **V2** | `cuda_das_v2.cu` | K1: 1 thread = 1 pair<br>K2: 1 thread = 1 pixel | Precomputes all $327,680 \times 128$ receive times into a 167.8 MB global memory buffer. | K1: 16<br>K2: 40 | 0 B | Avoids redundant distance math, but introduces massive global memory footprint and DRAM bandwidth pressure. |
| **V3** | `cuda_das_v3.cu` | 1 thread = 1 pixel | Allocates thread-local array `float receive_times[128]` (512 bytes) on the stack frame in local memory. | 31 | **512 B stack** (0 B spill) | **Local Memory Stack Frame:** Because the 128-element array is dynamically indexed and exceeds register budget, nvcc places it on the per-thread local-memory stack. ptxas reports `STACK: 512, SPILL: 0` (this is an array stack allocation, not register spilling). Causes high memory latency. |
| **V4** | `cuda_das_v4.cu` | **1 warp (32 threads) = 1 pixel** | Cooperative warp beamforming. Block size 256 (8 warps). Each lane handles 4 elements. Tree reductions via `__shfl_down_sync` + broadcast. Evaluates apodization branch. | **32** | **0 B** | **Warp-Cooperative Baseline Reference.** Fits entirely in registers; zero stack frame. Full 100% theoretical occupancy. |
| **V5** | `cuda_das_v5.cu` | 1 warp = 1 pixel | Replaces down-shuffle + broadcast with butterfly all-reduce via `__shfl_xor_sync`. | 34 | 0 B | Validates alternative reduction network. Identical performance to V4 (289.25 ms vs 289.72 ms). |
| **V6** | `cuda_das_v6.cu` | 1 warp = 1 pixel | Dataset specialization: removes transmit apodization reads/checks because all PICMUS transmit weights equal 1.0. | **37** | **0 B** | **Severe Regression ($20.7\times$ slowdown on battery):** Removing the conditional branch triggered an unphased load burst. DRAM traffic surged to $1,250.0\text{ MB}$, L1 hit rate plunged to $21.8\%$, and MIO throttle stalled $44.5\%$ of warp cycles. |
| **V7** | `cuda_das_v7.cu` | 1 warp = 1 pixel | Restores sequential load phasing by inserting `__syncwarp(0xffffffffu)` between the four `receive_contribution()` calls. | **29** | **0 B** | **Best-Performing Evaluated Kernel:** Emits `bar.warp.sync -1` / `BSYNC B0`. Drops register usage to 29 regs, restores cache hit rates, and achieves **12.02% kernel speedup over V4**. |

### 5.1 V6/V7 Profiling Observations vs Scheduling Hypotheses

To maintain engineering rigor, raw profiling measurements are distinguished from compiler scheduling hypotheses:

#### A. Measured Profiling Observations (Nsight Compute & cuobjdump)
- **Memory Traffic:** In V4, DRAM read traffic is $72.5\text{ MB}$. In V6, DRAM traffic surges to $1,250.0\text{ MB}$ ($17.3\times$ inflation). In V7, DRAM traffic returns to $72.1\text{ MB}$.
- **Cache Hit Rates:** L1 data cache hit rate is $75.4\%$ in V4, drops to $21.8\%$ in V6, and is restored to $73.0\%$ in V7. L2 cache hit rate drops from $90.7\%$ (V4) to $31.7\%$ (V6), and recovers to $88.2\%$ (V7).
- **Warp Stall Reasons:** Nsight Compute reports that in V6, **MIO Throttle** dominates warp stall reasons, accounting for **$44.5\%$ of active warp cycles** (1,418 average stall cycles per warp). In V4 and V7, MIO stalls are negligible ($< 1\%$).
- **SASS Disassembly:** In V4 (`v4_sass.txt`), loads are interleaved across predicated control blocks (`@P0`..`@P3`) with `BSSY`/`BSYNC` convergence barriers. In V6 (`v6_sass.txt`), all eight `LDG.E` loads are grouped consecutively in a single basic block. In V7 (`v7_sass.txt`), explicit `BSYNC B0` barrier instructions separate each pair of `LDG.E` loads.

#### B. Architectural Hypotheses
- **Compiler Hoisting Model:** It is hypothesized that removing the conditional apodization branch (`if (apodization <= 0.0f)`) eliminated compiler-enforced control-flow barriers. Without branches, nvcc aggressively unrolled the loop and hoisted all 8 global memory loads into an unphased burst.
- **Cache Thrashing Mechanism:** With 48 concurrent active warps per SM issuing 8 simultaneous 128-byte cache-line requests, 384 concurrent memory requests saturated the finite MIO queue and exceeded L1 cache capacity, inducing catastrophic cache eviction and re-fetching from DRAM. Inserting `__syncwarp()` forces the compiler and hardware to re-establish a phased (2 loads $\to$ math $\to$ sync $\to$ 2 loads) cadence.

---

## 6. Numerical Comparison and Implementation Fidelity

### 6.1 Cross-GPU Parity (Full Image: 327,680 Pixels)
Evaluated across the complete $320 \times 1,024$ output matrix:
- **V4, V5, V6, and V7 are 100% BIT-IDENTICAL**:
  - Maximum Absolute Difference: **$0.0$**
  - Mean Absolute Difference: **$0.0$**
  - RMSE: **$0.0$**
  - NaN / Inf Count: 0 / 0
- **V1, V2, V3 vs V4**:
  - Maximum Absolute Difference: $0.00024414$ ($2^{-12}$)
  - RMSE: $2.947 \times 10^{-6}$
  - *Root Cause:* Minor non-associative floating-point summation order differences between sequential loop accumulation and warp-level binary tree reductions.

### 6.2 GPU Implementation Fidelity vs CPU Reference
Evaluated against `data/binary/cpu_das_image.bin` across all 327,680 pixels:
- **Raw DAS RMSE:** **$0.00030968$**
- **Relative RMSE:** **$0.000873\%$**
- **Maximum Absolute Difference:** **$0.02478027$**
- **B-Mode Dynamic Range Difference** (post-Hilbert envelope, $-60\text{ dB}$ log scale):
  - Maximum Pixel Difference: **$0.003390\text{ dB}$**
  - Mean Pixel Difference: **$0.00002809\text{ dB}$**

---

## 7. Timing Conventions and Scientific Assumptions

### 7.1 Transmit Arrival Time and `initial_times`
- **Minimum Transmit Delay:** The time of flight from the virtual plane-wave source to pixel $\mathbf{p}_i$ is computed as:
  $$\tau_{\text{tx}}(i) = \min_{j \in \text{active}} \left( \tau_{\text{rx}}(i, j) + t_0(tx, j) \right) - t_{\text{init}}(tx)$$
  This formulation follows Fermat's principle: the earliest acoustic energy arriving at point $\mathbf{p}_i$ defines the leading wavefront.
- **Dataset Initial Times (`initial_times`):** `initial_times` is a per-transmission time offset array of shape $(75,)$ provided in the PICMUS HDF5 acquisition parameters (`tracks/track_0/scan/initial_times`). In the delay calculation, `initial_times[tx]` is subtracted from the candidate arrival time to align the time origin with the start of the recorded RF buffer for each transmission event. Its specific hardware synchronization origin is not documented in the dataset.

### 7.2 Origin of $t_{\text{peak}} = 1 / f_c$
- **Value:** $t_{\text{peak}} = \frac{1}{5,208,000\text{ Hz}} = 0.1920123\text{ }\mu\text{s}$.
- **Finding:** The PICMUS HDF5 file does not record a measured excitation pulse envelope peak. Inspection of the ZEA source code (`zea/parameters.py`) confirmed that when pulse metadata is absent, the library defaults to shifting the pulse by one carrier cycle ($1 / f_c$).
- **Classification:** **Algorithmic Modeling Assumption.** This value is an assumed analytical shift, not an empirical pulse calibration measured from the data.

### 7.3 Receive Aperture ($f$-Number)
- **Setting:** $f_{\#} = 0.0$ (Full Receive Aperture). All 128 elements contribute to every pixel.
- **Finding:** In `Untitled3.ipynb`, setting $f_{\#} = 0.0$ is our full-aperture engineering choice (`# Initial assumption: full receive aperture`).
- **Classification:** **In-Project Engineering Choice.** Full aperture was selected to maximize acoustic signal capture, simplify kernel control flow, and avoid dynamic aperture branching on GPU. It must not be cited as an official PICMUS competition requirement.

---

## 8. Benchmark Methodology and Controlled 25 W Results

### 8.1 Benchmark Methodology
- **Hardware Platform:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere sm_86, 16 SMs, 2048 CUDA cores, Driver / CUDA Toolkit 13.5, Windows 11).
- **Operating Conditions:** Controlled battery-powered profile operating under mobile power constraints (observed clocks: Graphics $\approx 1730\text{ MHz}$, Memory $\approx 810\text{ MHz}$).
- **Measurement Protocol:** 2 warm-up iterations, 20 measured iterations, high-precision CUDA event timing (`cudaEventElapsedTime`). Launch dimensions: 40,960 blocks $\times$ 256 threads (1 warp/pixel).

### 8.2 Primary Controlled Results (25 W Mobile Profile)

| Stage | Metric | CUDA DAS V4 (Baseline) | CUDA DAS V7 (Synchronized) | Controlled Improvement (V7 vs V4) |
| :--- | :--- | :---: | :---: | :---: |
| **Host-to-Device Transfer (H2D)** | Median | $19.3337\text{ ms}$ | $19.1917\text{ ms}$ | — |
| | P95 | $19.7755\text{ ms}$ | $20.4329\text{ ms}$ | — |
| **Kernel Latency** | Median | **$289.7211\text{ ms}$** | **$254.8951\text{ ms}$** | **$12.02\%$ latency reduction** ($1.137\times$ faster) |
| | P95 | $289.9589\text{ ms}$ | $255.1807\text{ ms}$ | **$12.00\%$ latency reduction** |
| **Device-to-Host Transfer (D2H)** | Median | $0.5771\text{ ms}$ | $0.5684\text{ ms}$ | — |
| | P95 | $0.7030\text{ ms}$ | $0.6562\text{ ms}$ | — |
| **Complete GPU Pipeline** | Median | **$310.1110\text{ ms}$** | **$275.3078\text{ ms}$** | **$11.22\%$ latency reduction** ($1.126\times$ faster) |
| | P95 | $310.6784\text{ ms}$ | $276.6326\text{ ms}$ | **$10.96\%$ latency reduction** |

### 8.3 CPU-to-GPU Speedup Summary
Baseline CPU reference execution: **$135.730019\text{ seconds}$** ($135,730.019\text{ ms}$).

- **CUDA DAS V4 (25 W):**
  - Kernel Speedup: $\frac{135,730.019\text{ ms}}{289.7211\text{ ms}} = \mathbf{468.49\times}$
  - Complete Pipeline Speedup: $\frac{135,730.019\text{ ms}}{310.1110\text{ ms}} = \mathbf{437.68\times}$
- **CUDA DAS V7 (25 W):**
  - Kernel Speedup: $\frac{135,730.019\text{ ms}}{254.8951\text{ ms}} = \mathbf{532.49\times}$
  - Complete Pipeline Speedup: $\frac{135,730.019\text{ ms}}{275.3078\text{ ms}} = \mathbf{493.01\times}$

### 8.4 Historical 80 W Power Figures (Status: NOT VERIFIED)
- **V4 Kernel (80 W AC):** $89.6\text{ ms}$ (Pipeline: $96.3\text{ ms}$) $\rightarrow$ **NOT VERIFIED**. This historical figure was provided in the initial task specification; no raw benchmark log or execution trace exists in this repository.
- **V7 Kernel (80 W AC Extrapolated):** $\approx 78.0\text{ ms}$ $\rightarrow$ **NOT VERIFIED (Unmeasured Projection)**. This value was derived mathematically by scaling $89.6\text{ ms} / 1.137 = 78.8\text{ ms} \approx 78\text{ ms}$; it is not an empirical benchmark.

---

## 9. Known Limitations and Assumptions

1. **No Physical `cpu_das_image.npy` File:** The standalone `.npy` file was not retrieved from Colab. Reference validation relies on the verified bit-identical arrays in `picmus_cuda_input.npz` and `data/binary/cpu_das_image.bin`.
2. **Acoustic Modeling Simplifications:** Constant sound speed ($c = 1540\text{ m/s}$) is assumed across the medium. Acoustic refraction through an acoustic lens or tissue heterogeneity is not modeled.
3. **Pulse Peak Modeling:** $t_{\text{peak}} = 1/f_c$ is an analytical shift, not an empirical pulse calibration.
4. **Full Aperture Selection:** $f_{\#} = 0.0$ disables dynamic receive apodization.

---

## 10. Current Status and Next Development Milestone

### 10.1 Current Status
The beamforming core is fully stabilized, numerically validated, and optimized through **CUDA DAS V7**. The compiler scheduling pathology identified in V6 is completely resolved.

### 10.2 Next Milestone: CUDA B-Mode Processing Pipeline
Currently, post-beamforming image formation (analytic signal envelope detection via Hilbert transform, dynamic range logarithmic compression, and grid display) is handled off-line in Python (`src/visualize_results.py` and `src/visualize_v7.py`).

The next engineering milestone is to implement **in-pipeline CUDA B-mode image generation**:
1. In-kernel or cuFFT-based Hilbert envelope extraction ($E(z, x) = |s(z, x) + j \mathcal{H}\{s(z, x)\}|$).
2. Vectorized logarithmic dynamic range scaling:
   $$I_{\text{dB}}(z, x) = 20 \log_{10}\left( \frac{E(z, x)}{\max E} + 10^{-12} \right), \qquad \text{clamped to } [-60\text{ dB}, 0\text{ dB}]$$
3. Direct output of 8-bit grayscale image frames ready for embedded display or video streaming.

---

## 11. Evidence Requiring Further Verification

The following items are explicitly marked as pending experimental confirmation:
1. **Empirical 80 W AC Benchmark for V4 and V7:** Execute controlled 20-run benchmarks of `cuda_das_v4.exe --full` and `cuda_das_v7.exe --full` on unthrottled AC power to replace historical figures and theoretical projections with measured logs.
2. **Measured Pulse Envelope Peak:** Perform Hilbert envelope extraction on the transmitted excitation waveform from the PICMUS simulation source to determine whether the physical pulse peak deviates from the assumed $1/f_c$ ($0.192\text{ }\mu\text{s}$) offset.
3. **Dynamic Aperture CUDA Evaluation:** Benchmark the memory divergence and occupancy impact of implementing dynamic receive apodization ($f_{\#} = 1.0\text{ to }1.75$) within the cooperative warp architecture.
