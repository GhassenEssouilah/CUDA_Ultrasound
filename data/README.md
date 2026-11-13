# Dataset Acquisition, Schema, and Binary Input Generation

This directory manages the input datasets, acoustic parameters, transducer geometries, and CPU reference reconstructions used by the CUDA ultrasound Delay-and-Sum (DAS) beamforming pipeline.

---

## 1. Dataset Origin & Description

The primary dataset is derived from the **Plane Wave Imaging Challenge in Medical Ultrasound (PICMUS)**, hosted at the 2016 IEEE International Ultrasonics Symposium (IUS).

- **Repository:** HuggingFace Hub [`zeahub/picmus`](https://huggingface.co/datasets/zeahub/picmus)
- **HDF5 Archive:** `database/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf/resolution_distorsion_simu_dataset_rf.hdf5`
- **Phantom Type:** Numerical simulation phantom containing point targets (wires) and anechoic cyst targets embedded in a homogeneous speckle-generating background. It is designed to rigorously benchmark lateral resolution, axial resolution, and cyst contrast ratio.

---

## 2. Binary Data Schema (`data/binary/`)

To achieve maximum I/O throughput and eliminate runtime parsing overhead in CUDA C++, all array inputs are stored as contiguous raw binary buffers (`.bin`). 

| Filename | Data Type | Dimensions | Elements | Size (Bytes) | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `rf_data.bin` | `float32` | $(75, 1527, 128)$ | 14,659,200 | 58,636,800 | Raw pre-beamformed RF echo samples $(N_{\text{tx}}, N_{\text{samples}}, N_{\text{elements}})$. |
| `flatgrid.bin` | `float32` | $(327680, 3)$ | 983,040 | 3,932,160 | Flattened 3D Cartesian coordinates $(x, y, z)$ in meters for all 327,680 reconstruction pixels. |
| `cpu_das_image.bin` | `float32` | $(1024, 320)$ | 327,680 | 1,310,720 | Reference beamformed RF envelope image produced by CPU DAS reference pipeline. |
| `t0_delays.bin` | `float32` | $(75, 128)$ | 9,600 | 38,400 | Transmit delay offsets for each element and steering angle (seconds). |
| `tx_apodizations.bin`| `float32` | $(75, 128)$ | 9,600 | 38,400 | Transmit apodization weights per element and transmission. |
| `probe_geometry.bin` | `float32` | $(128, 3)$ | 384 | 1,536 | Cartesian element coordinates $(x_e, y_e, z_e)$ in meters (pitch = 0.3 mm). |
| `initial_times.bin` | `float32` | $(75,)$ | 75 | 300 | Acquisition start time offsets for each plane wave transmission (seconds). |
| `t_peak.bin` | `float32` | $(75,)$ | 75 | 300 | Transmit pulse center timing offset ($t_{\text{peak}} = 1/f_c \approx 1.92 \times 10^{-7}\text{ s}$). Assumed analytical convention generated via ZEA default ($1/f_c$); not an empirical pulse-peak parameter from PICMUS HDF5 metadata. |
| `grid_shape.bin` | `int32` | $(2,)$ | 2 | 8 | Reconstruction grid dimensions: `[1024, 320]` ($N_z, N_x$). |
| `sampling_frequency.bin`| `float32` | $(1,)$ | 1 | 4 | Receiver ADC sampling frequency: $f_s$ = 20,832,000 Hz. |
| `sound_speed.bin` | `float32` | $(1,)$ | 1 | 4 | Acoustic speed of sound in medium: $c = 1540.0$ m/s. |

### Memory Layout Note
In `rf_data.bin`, the array is arranged in contiguous C-order with shape `(75, 1527, 128)`, corresponding to `(transmission, sample, receive element)`:
$$\text{Linear Offset} = (tx \times N_{\text{samples}} + sample) \times N_{\text{elements}} + element$$
The fastest-varying inner dimension is the 128 transducer receive elements (unit stride). Adjacent receive channels reside contiguously in memory, enabling coalesced memory access across threads in a warp.

---

## 3. Data Generation Pipeline

The generation of binary files follows a two-stage reproducible pipeline:

```
HuggingFace Hub (zeahub/picmus)
       │
       ▼ (download & CPU beamforming)
notebooks/01_cpu_das_reference_pipeline.ipynb
       │
       ▼ (generates)
data/picmus_cuda_input.npz  (~42.3 MB compressed)
       │
       ▼ (extracts & verifies)
python src/scripts/export_data.py
       │
       ▼ (outputs)
data/binary/*.bin  (11 flat binary buffers)
```

### Step 1: Download & CPU Reference Generation
Run `notebooks/01_cpu_das_reference_pipeline.ipynb`:
1. Downloads `resolution_distorsion_simu_dataset_rf.hdf5` using `huggingface_hub.snapshot_download`.
2. Extracts probe geometry, RF echoes, transmit angles, and acquisition timings.
3. Computes the CPU reference beamformed image ($1024 \times 320$) using vectorised Delay-and-Sum.
4. Saves all packaged inputs to `data/picmus_cuda_input.npz`.

### Step 2: Binary Export & Integrity Verification
Run the export script:
```bash
python src/scripts/export_data.py
```
This script:
- Reads `data/picmus_cuda_input.npz`.
- Ensures all arrays are C-contiguous `float32` (or `int32` for `grid_shape`).
- Writes out each array to `data/binary/<name>.bin`.
- Re-reads each exported binary file and verifies bit-for-bit equality against the NumPy array.

### Step 3: Inspect Exported Data
To verify tensor shapes and first element values at any time:
```bash
python src/scripts/inspect_data.py
```

---

## 4. Storage & Version Control Policy (`.gitignore`)

To adhere to Git repository size best practices and avoid GitHub's 50 MB / 100 MB file limits:
- `data/binary/rf_data.bin` (58.6 MB) is **excluded** from Git tracking.
- `data/*.npz` (42.3 MB) is **excluded** from Git tracking.
- Parameter and geometry buffers (`probe_geometry.bin`, `flatgrid.bin`, `t0_delays.bin`, `initial_times.bin`, etc.) are tracked in Git to allow instantaneous validation without redownloading the full RF dataset.
- The full raw dataset can be reconstructed deterministically at any time using the workflow outlined above.

