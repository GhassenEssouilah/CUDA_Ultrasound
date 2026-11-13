
from pathlib import Path
from collections import Counter
import re

for version in ["v4", "v6", "v7"]:

    ROOT = Path(__file__).resolve().parents[2]
    path = ROOT / f"results/profiling/{version}_sass.txt"
    if not path.exists():
        path = ROOT / f"results/{version}_sass.txt"
    text = path.read_text(errors="ignore")

    print("\n" + "=" * 65)
    print(f" {version.upper()} SASS ANALYSIS")
    print("=" * 65)

    lines = text.splitlines()

    # Find the exact kernel header
    start = None

    for i, line in enumerate(lines):
        if f"Function : _Z13das_kernel_{version}" in line:
            start = i
            break

    if start is None:
        print("Kernel not found.")
        continue

    # Keep everything after the kernel header until the next function
    kernel_lines = []

    for line in lines[start + 1:]:
        if "Function :" in line:
            break
        kernel_lines.append(line)

    kernel = "\n".join(kernel_lines)

    # Extract instructions from cuobjdump SASS format
    instructions = re.findall(
        r"/\*([0-9a-fA-F]+)\*/\s+([A-Z][A-Z0-9_.]*)\s+(.*?);",
        kernel
    )

    counts = Counter(opcode for address, opcode, operands in instructions)

    print("Total instructions:", len(instructions))

    print("\n--- Instruction frequencies ---")

    for opcode, count in counts.most_common():
        print(f"{opcode:25s} {count}")

    print("\n--- Important instruction groups ---")

    groups = {
        "Memory": ("LD", "ST"),
        "Warp": ("SHFL",),
        "Floating point": ("FADD", "FMUL", "FFMA", "MUFU"),
        "Integer": ("IMAD", "IADD", "IMUL"),
        "Control flow": ("BRA", "BRX", "JMP"),
    }

    for group, prefixes in groups.items():
        total = sum(
            count for opcode, count in counts.items()
            if opcode.startswith(prefixes)
        )
        print(f"{group:20s}: {total}")

    print("\n--- First 20 instructions ---")

    for address, opcode, operands in instructions[:20]:
        print(f"{address:>6s}  {opcode:20s} {operands}")
