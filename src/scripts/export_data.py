
import numpy as np
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_DIR / "data" / "picmus_cuda_input.npz"
OUTPUT_DIR = PROJECT_DIR / "data" / "binary"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original dataset
with np.load(INPUT_FILE) as data:

    # Arrays to export as float32 binary files
    arrays = [
        "rf_data",
        "probe_geometry",
        "t0_delays",
        "tx_apodizations",
        "initial_times",
        "t_peak",
        "flatgrid",
        "cpu_das_image",
        "sampling_frequency",
        "sound_speed",
    ]

    print("========== EXPORTING CUDA DATA ==========")

    for name in arrays:
        array = np.ascontiguousarray(data[name], dtype=np.float32)

        output_file = OUTPUT_DIR / f"{name}.bin"
        array.tofile(output_file)

        # Verify the exported file
        loaded = np.fromfile(output_file, dtype=np.float32)

        if not np.array_equal(array.reshape(-1), loaded):
            raise RuntimeError(f"Verification failed: {name}")

        print(
            f"{name:22s} "
            f"Shape: {str(array.shape):20s} "
            f"Size: {output_file.stat().st_size / 1024**2:.2f} MB "
            f"[OK]"
        )

    # Export grid dimensions as int32
    grid_shape = np.asarray(data["grid_shape"], dtype=np.int32)

    grid_file = OUTPUT_DIR / "grid_shape.bin"
    grid_shape.tofile(grid_file)

    print("\ngrid_shape             ", grid_shape.tolist(), "[OK]")

print("\nAll binary files exported and verified!")
print("Output directory:", OUTPUT_DIR)
