
from pathlib import Path
import re

targets = {
    "v4": ["16d0", "1ba0", "1d00"],
    "v6": ["1450", "1920", "1a80"],
}

for version, offsets in targets.items():

    ROOT = Path(__file__).resolve().parents[2]
    p = ROOT / f"results/profiling/{version}_sass.txt"
    if not p.exists(): p = ROOT / f"results/{version}_sass.txt"
    lines = p.read_text(
        errors="ignore"
    ).splitlines()

    print("\n" + "=" * 65)
    print(f"{version.upper()} CALL TARGETS")
    print("=" * 65)

    for target in offsets:

        print(f"\n--- Target 0x{target} ---")

        start = None

        for i, line in enumerate(lines):
            match = re.search(r"/\*([0-9a-fA-F]+)\*/", line)

            if match and match.group(1).lower() == target:
                start = i
                break

        if start is None:
            print("Target not found.")
            continue

        for line in lines[start:start + 25]:
            print(line.strip())
