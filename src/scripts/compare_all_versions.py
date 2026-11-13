
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

files = {
    "V1": "cuda_das_image.bin",
    "V2": "cuda_das_optimized_image.bin",
    "V3": "cuda_das_v3_image.bin",
    "V4": "cuda_das_v4_image.bin",
    "V5": "cuda_das_v5_image.bin",
    "V6": "cuda_das_v6_image.bin",
    "V7": "cuda_das_v7_image.bin",
}

reference_name = "V4"
reference = np.fromfile(
    RESULTS / files[reference_name],
    dtype=np.float32
)

print("=" * 70)
print("CUDA DAS - FULL IMAGE NUMERICAL COMPARISON")
print("=" * 70)

print(f"\nReference: {reference_name}")
print(f"Reference pixels: {reference.size}")

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
        print("Cannot compare: different output sizes")
        continue

    difference = np.abs(data - reference)

    print("Bit-identical:", np.array_equal(data, reference))
    print("Maximum absolute difference:", np.max(difference))
    print("Mean absolute difference:", np.mean(difference))
    print("RMSE:", np.sqrt(np.mean((data - reference) ** 2)))
    print("NaN values:", np.isnan(data).sum())
    print("Infinite values:", np.isinf(data).sum())

print("\n" + "=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)
