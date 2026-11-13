from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
v4 = np.fromfile(RESULTS / "cuda_das_v4_image.bin", dtype=np.float32)
v_other = np.fromfile(RESULTS / f"cuda_das_v7_image.bin", dtype=np.float32)

print("Bit-identical:", np.array_equal(v4, v_other))
print("Max difference:", np.max(np.abs(v4 - v_other)))
