"""Serial-CPU per-step time of the polygon forward by category (profile_forward_cpu.py)
against SPARTA's serial step -> examples/mesh/figures/forward_step_breakdown_cpu.png"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "profile", "forward_cpu_serial.json")))
sparta = min(
    float(line.split()[3])
    for line in open(os.path.join(HERE, "bench_sparta.txt")).readlines()[1:]
    if line.startswith("serial")
)
order = sorted(d["categories"].items(), key=lambda t: -t[1])
fig, ax = plt.subplots(figsize=(10, 4.2))
y = np.arange(len(order))[::-1]
ax.barh(y, [v for _c, v in order], color="#b8461c")
for yi, (_c, v) in zip(y, order, strict=True):
    ax.text(v, yi, f" {v:.1f} ms", va="center", fontsize=7.5)
ax.axvline(sparta, color="#2a78d6", lw=1.5)
ax.text(
    sparta,
    y[0] + 0.6,
    f" SPARTA's whole step, 1 core: {sparta:.2f} ms",
    color="#2a78d6",
    fontsize=7.5,
)
ax.set_yticks(y)
ax.set_yticklabels([c for c, _v in order], fontsize=7.5)
ax.set_xlabel("ms per step, one core")
ax.set_title(
    f"polygon circle forward on one CPU core (XLA:CPU): {d['wall_ms_per_step']:.1f} ms/step "
    f"vs SPARTA {sparta:.2f} ms/step",
    fontsize=8.5,
)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(
    os.path.join(HERE, "..", "..", "examples", "mesh", "figures", "forward_step_breakdown_cpu.png"),
    dpi=150,
)
