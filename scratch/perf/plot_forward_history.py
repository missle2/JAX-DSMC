"""Forward time of the SPARTA circle case (1000 steps from empty) across the speed-up
changes, against SPARTA serial, sparta_jax and the 5x-SPARTA target.
-> examples/mesh/figures/forward_speedup_history.png

Milestones (all on this laptop, best of 2-3 timed runs of scratch/perf/bench_forward.py):
  2026-09-28  before        bench_forward_*_polygon.json / the 2026-09-28 summary
  2026-09-28  batch         epsilon skip, packed sort, culls (bench_forward_*_batch.json)
  2026-09-30  merge 1       Halley inflow, level-set search, tracker compactions
                            (scratch/perf/merged/bench_*.log)
  2026-10-01  merge 2       routed (SPARTA-structured) tracker on CPU, counter draws, NTC
                            and RNG roll-ins (JAX-DSMC-sparta final session)
  2026-10-01  sort          carried partition sort, revival at slots, one-pass cell
                            temperature (scratch/perf/bench_forward_{gpu,cpu_serial}.json)

    python3 scratch/perf/plot_forward_history.py
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "examples", "mesh", "figures", "forward_speedup_history.png")

MILESTONES = ["09-28\nbefore", "09-28\nbatch", "09-30\nmerge 1", "10-01\nmerge 2", "10-01\nsort"]
# seconds per 1000 steps; the last column is read from the current benchmark files
HISTORY = {
    ("cpu", "polygon"): [67.6, 27.9, 15.76, 8.73],
    ("cpu", "levelset"): [98.3, 59.2, 35.17, 13.06],
    ("gpu", "polygon"): [15.1, 9.4, 5.12, 4.50],
    ("gpu", "levelset"): [20.8, 17.7, 7.54, 6.99],
}
SPARTA_SERIAL = 0.518  # spa_serial, best of 5, 2026-10-01
SPARTA_JAX = {"cpu": 1.295, "gpu": 1.410}  # sparta_jax, same session

SERIES = [  # (platform, body, label, colour) -- categorical slots 1-4, fixed order
    ("cpu", "polygon", "1 CPU core, polygon", "#2a78d6"),
    ("cpu", "levelset", "1 CPU core, level set", "#eb6834"),
    ("gpu", "polygon", "GPU, polygon", "#1baf7a"),
    ("gpu", "levelset", "GPU, level set", "#eda100"),
]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def current():
    files = {"cpu": "bench_forward_cpu_serial.json", "gpu": "bench_forward_gpu.json"}
    out = {}
    for platform, name in files.items():
        for row in json.load(open(os.path.join(HERE, name))):
            out[(platform, row["body"])] = row["seconds"]
    return out


def _spread(ends, gap=0.07):
    """Label heights for the line ends: at least ``gap`` decades apart (log axis)."""
    ordered = sorted(ends)
    out, last = [], -np.inf
    for value, label in ordered:
        level = max(np.log10(value), last + gap)
        out.append((value, 10**level, label))
        last = level
    return out


def main():
    now = current()
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = list(range(len(MILESTONES)))
    ends = []
    for platform, body, label, colour in SERIES:
        y = HISTORY[(platform, body)] + [now[(platform, body)]]
        ax.plot(x, y, color=colour, lw=2, marker="o", ms=6, label=label)
        ends.append((y[-1], label))
    for value, text_y, label in _spread(ends):
        ax.annotate(
            f"{label}  {value:.1f} s",
            (x[-1], value),
            xytext=(x[-1] + 0.12, text_y),
            textcoords="data",
            va="center",
            fontsize=8,
            color=INK,
        )
    refs = [
        (5 * SPARTA_SERIAL, "5x SPARTA serial (target)"),
        (SPARTA_JAX["cpu"], "sparta_jax, 1 CPU core"),
        (SPARTA_SERIAL, "SPARTA serial"),
    ]
    for value, label in refs:
        ax.axhline(value, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(-0.15, value * 1.06, f"{label}  {value:.2f} s", fontsize=7.5, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 130)
    ax.set_xlim(-0.2, len(MILESTONES) + 0.9)
    ax.set_xticks(x, MILESTONES, fontsize=8, color=INK)
    ax.set_ylabel("seconds per 1000 steps (log scale)", color=INK)
    ax.set_title(
        "SPARTA circle case, 1000 steps from empty: main-code forward across the speed-ups",
        fontsize=9.5,
        color=INK,
    )
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
