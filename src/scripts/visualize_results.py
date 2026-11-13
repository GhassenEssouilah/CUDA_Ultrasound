
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "data" / "binary"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------
# IMAGE DIMENSIONS
# ---------------------------------------------------------

HEIGHT = 1024
WIDTH = 320
N_PIXELS = HEIGHT * WIDTH

# ---------------------------------------------------------
# LOAD CPU AND CUDA V4 RESULTS
# ---------------------------------------------------------

cpu_file = BINARY / "cpu_das_image.bin"
cuda_file = RESULTS / "cuda_das_v4_image.bin"

if not cpu_file.exists():
    raise FileNotFoundError(f"Missing CPU reference: {cpu_file}")

if not cuda_file.exists():
    raise FileNotFoundError(f"Missing CUDA V4 output: {cuda_file}")

cpu = np.fromfile(cpu_file, dtype=np.float32)
cuda = np.fromfile(cuda_file, dtype=np.float32)

if cpu.size != N_PIXELS or cuda.size != N_PIXELS:
    raise ValueError("Incorrect number of pixels in input files.")

cpu = cpu.reshape(HEIGHT, WIDTH)
cuda = cuda.reshape(HEIGHT, WIDTH)

# ---------------------------------------------------------
# ANALYTIC SIGNAL (FFT-BASED HILBERT TRANSFORM)
# ---------------------------------------------------------

def analytic_signal(signal, axis=0):
    n = signal.shape[axis]

    spectrum = np.fft.fft(signal, axis=axis)
    h = np.zeros(n)

    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2

    shape = [1] * signal.ndim
    shape[axis] = n

    return np.fft.ifft(
        spectrum * h.reshape(shape), axis=axis
    )

# ---------------------------------------------------------
# B-MODE PROCESSING
# ---------------------------------------------------------

def make_bmode(image, reference_max):
    envelope = np.abs(analytic_signal(image, axis=0))
    envelope_normalized = envelope / (reference_max + 1e-12)

    bmode = 20 * np.log10(envelope_normalized + 1e-12)

    return np.clip(bmode, -60, 0)

# Common normalization reference: CPU envelope maximum
cpu_envelope = np.abs(analytic_signal(cpu, axis=0))
reference_max = np.max(cpu_envelope)

cpu_bmode = make_bmode(cpu, reference_max)
cuda_bmode = make_bmode(cuda, reference_max)

# ---------------------------------------------------------
# ERROR ANALYSIS
# ---------------------------------------------------------

raw_error = np.abs(cuda - cpu)
bmode_error = np.abs(cuda_bmode - cpu_bmode)

rmse = np.sqrt(np.mean((cuda - cpu) ** 2))
relative_rmse = (
    100 * rmse / np.sqrt(np.mean(cpu ** 2))
)

print("\n========== CPU vs CUDA V4 ==========")

print(f"Image dimensions: {WIDTH} x {HEIGHT}")
print(f"Total pixels: {N_PIXELS}")

print("\n--- Raw DAS error ---")
print(f"RMSE: {rmse:.8f}")
print(f"Maximum absolute error: {raw_error.max():.8f}")
print(f"Mean absolute error: {raw_error.mean():.8f}")
print(f"Relative RMSE: {relative_rmse:.6f} %")

print("\n--- B-mode error ---")
print(f"Maximum difference: {bmode_error.max():.6f} dB")
print(f"Mean difference: {bmode_error.mean():.8f} dB")

# ---------------------------------------------------------
# IMAGE COORDINATES
# ---------------------------------------------------------

extent = [-19.2, 19.2, 50.0, 5.0]

# ---------------------------------------------------------
# VISUAL COMPARISON
# ---------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

images = [
    (cpu_bmode, "CPU DAS - B-mode", -60, 0),
    (cuda_bmode, "CUDA V4 DAS - B-mode", -60, 0),
    (bmode_error, "Absolute B-mode difference (dB)", 0, 1),
    (raw_error, "Absolute raw DAS error", 0, None),
]

for ax, (data, title, vmin, vmax) in zip(axes.flat, images):

    cmap = (
        "gray"
        if "B-mode" in title and "difference" not in title
        else "inferno"
    )

    im = ax.imshow(
        data,
        extent=extent,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title)
    ax.set_xlabel("Lateral position (mm)")
    ax.set_ylabel("Depth (mm)")

    fig.colorbar(
        im,
        ax=ax,
        label="dB" if "dB" in title else "Amplitude"
    )

fig.suptitle(
    "PICMUS Ultrasound - CPU DAS vs CUDA V4",
    fontsize=15
)

fig.tight_layout()

output = FIGURES / "cpu_vs_cuda_v4_comparison.png"

fig.savefig(output, dpi=200, bbox_inches="tight")

print(f"\nComparison saved: {output}")

plt.show()
