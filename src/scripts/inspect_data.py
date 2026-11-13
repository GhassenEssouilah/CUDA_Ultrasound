import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "data/picmus_cuda_input.npz"
if not path.exists(): path = Path("data/picmus_cuda_input.npz")

if not path.exists():
    raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

with np.load(path) as data:

    print("\n========== PICMUS DATASET ==========")

    for key in data.files:
        array = data[key]

        print(f"\n{key}")
        print("Shape:", array.shape)
        print("Data type:", array.dtype)
        print("Memory:", round(array.nbytes / 1024**2, 2), "MB")

    print("\nDataset loaded successfully!")