
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Validation outputs (4096 pixels)
files = {
    "V1": "cuda_das_test.bin",
    "V2": "cuda_das_optimized_test.bin",
    "V4": "cuda_das_v4_test.bin",
    "V5": "cuda_das_v5_test.bin",
    "V6": "cuda_das_v6_test.bin",
    "V7": "cuda_das_v7_test.bin",
}

reference = np.fromfile(
    RESULTS / files["V4"],
    dtype=np.float32
)

print("=" * 70)
print("CUDA DAS - VALIDATION OUTPUT COMPARISON")
print("=" * 70)

print("Reference: V4")
print("Reference pixels:", reference.size)

for version, filename in files.items():

    path = RESULTS / filename

    if not path.exists():
        print(f"\n{version}: FILE NOT FOUND")
        continue

    data = np.fromfile(path, dtype=np.float32)

    print(f"\n{version}")
    print("-" * 40)
    print("Pixels:", data.size)

    if data.shape != reference.shape:
        print("Different output size - comparison skipped")
        continue

    difference = data - reference

    print("Bit-identical:", np.array_equal(data, reference))
    print("Maximum absolute difference:",
          np.max(np.abs(difference)))
    print("Mean absolute difference:",
          np.mean(np.abs(difference)))
    print("RMSE:",
          np.sqrt(np.mean(difference ** 2)))

print("\n" + "=" * 70)
print("VALIDATION COMPARISON COMPLETED")
print("=" * 70)
