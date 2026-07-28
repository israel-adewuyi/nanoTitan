import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "assets" / "gemm_benchmark.csv").open() as f:
    rows = list(csv.DictReader(f))

plt.bar([r["implementation"] for r in rows], [float(r["tflops"]) for r in rows])
plt.ylabel("TFLOP/s")
plt.tight_layout()
plt.savefig(ROOT / "assets" / "gemm_benchmark.png", dpi=150)
