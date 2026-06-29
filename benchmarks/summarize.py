"""Aggregate benchmarks/results_*.json (written by bench.py) into one table."""

import glob
import json

files = sorted(glob.glob("benchmarks/results_*.json"))
data = {}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    data[d["backend"]] = dict(d["results"])

backends = list(data.keys())
cases = list(next(iter(data.values())).keys()) if data else []

header = f"{'case':>16s} | " + " | ".join(f"{b:>14s}" for b in backends)
print(header)
print("-" * len(header))
for case in cases:
    row = f"{case:>16s} | "
    row += " | ".join(f"{data[b].get(case, float('nan')):14.3f}" for b in backends)
    print(row)

if "pytorch" in data:
    print()
    print(f"{'case':>16s} | " + " | ".join(f"{b:>14s}" for b in backends if b != "pytorch"))
    for case in cases:
        ref = data["pytorch"].get(case)
        row = f"{case:>16s} | "
        cells = []
        for b in backends:
            if b == "pytorch":
                continue
            v = data[b].get(case)
            cells.append(f"{v / ref:13.2f}x" if ref else "n/a")
        row += " | ".join(cells)
        print(row)
