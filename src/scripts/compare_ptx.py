
from pathlib import Path
import re
from collections import Counter

versions = ["v4", "v6", "v7"]

for version in versions:
    ROOT = Path(__file__).resolve().parents[2]
    path = ROOT / f"results/profiling/{version}.ptx"
    if not path.exists():
        path = ROOT / f"results/{version}.ptx"

    print("\n" + "=" * 60)
    print(f" {version.upper()} PTX ANALYSIS")
    print("=" * 60)

    if not path.exists():
        print(f"Missing file: {path}")
        continue

    text = path.read_text(errors="ignore")

    # Find the entry containing our kernel name
    lines = text.splitlines()

    start = None

    for i, line in enumerate(lines):
        if ".entry" in line and f"das_kernel_{version}" in line:
            start = i
            break

    if start is None:
        print("Kernel not found.")
        print("\nAvailable kernel entries:")

        for line in lines:
            if ".entry" in line:
                print(line.strip())

        continue

    # Extract kernel until its closing brace
    kernel_lines = []
    opened = False
    depth = 0

    for line in lines[start:]:
        kernel_lines.append(line)

        depth += line.count("{")
        depth -= line.count("}")

        if "{" in line:
            opened = True

        if opened and depth == 0:
            break

    kernel = "\n".join(kernel_lines)

    print("\n--- Kernel information ---")
    print("Kernel entry:", lines[start].strip())
    print("PTX lines:", len(kernel_lines))

    print("\n--- Instruction counts ---")

    instructions = re.findall(
        r"^\s*([a-zA-Z][a-zA-Z0-9_.]*)\s+",
        kernel,
        re.M
    )

    counts = Counter(instructions)

    for instruction, count in counts.most_common():
        print(f"{instruction:25s} {count}")

    print("\n--- Register declarations ---")

    for line in kernel_lines:
        if ".reg " in line:
            print(line.strip())

    print("\n--- Memory operations ---")

    for line in kernel_lines:
        if re.search(r"\b(ld|st)\.(local|global)\b", line):
            print(line.strip())

    print("\n--- Warp shuffle operations ---")

    for line in kernel_lines:
        if "shfl" in line:
            print(line.strip())
