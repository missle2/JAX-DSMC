"""Bar chart: forward throughput (particle moves per second) of the SPARTA circle case --
SPARTA on this laptop's CPUs (scratch/perf/bench_sparta.txt, best of 3 per rank count, and
the reference logs of examples/mesh/results/sparta) against this code on the GPU and the
CPU (scratch/perf/bench_forward_{gpu,cpu}.json).  -> examples/mesh/figures/performance/forward_vs_sparta.png

    python3 scratch/perf/plot_forward_vs_sparta.py
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(ROOT, "examples", "mesh", "figures", "performance", "forward_vs_sparta.png")


def sparta_rows():
    best = {}
    with open(os.path.join(HERE, "bench_sparta.txt")) as fh:
        next(fh)
        for line in fh:
            build, ranks, _rep, secs, moves, _np = line.split()
            key = "SPARTA serial" if build == "serial" else f"SPARTA MPI x{ranks}"
            rate = float(moves) / float(secs)
            best[key] = max(best.get(key, 0.0), rate)
    ref = []
    folder = os.path.join(ROOT, "examples", "mesh", "results", "sparta")
    for name in sorted(os.listdir(folder)):
        if name.startswith("log.circle_ar."):
            text = open(os.path.join(folder, name)).read()
            secs = float(re.search(r"Loop time of ([\d.]+)", text).group(1))
            moves = float(re.search(r"Particle moves\s+=\s+(\d+)", text).group(1))
            ref.append(moves / secs)
    rows = [("SPARTA serial\n(reference logs)", max(ref), "#8a8a86")] if ref else []
    order = ["SPARTA serial"] + [f"SPARTA MPI x{r}" for r in (1, 2, 4, 8, 16)]
    rows += [(k.replace(" MPI", "\nMPI"), best[k], "#8a8a86") for k in order if k in best]
    return rows


def jax_rows():
    rows = []
    for platform, name, colour in (
        ("gpu", "GPU", "#2a78d6"),
        ("cpu", "CPU all cores", "#eb6834"),
        ("cpu_serial", "CPU 1 core", "#b8461c"),
    ):
        path = os.path.join(HERE, f"bench_forward_{platform}.json")
        if not os.path.exists(path):
            continue
        for r in json.load(open(path)):
            label = f"JAX-DSMC {name}\n{r['body']}"
            rows.append((label, r["moves_per_second"], colour))
    return rows


def main():
    rows = sparta_rows() + jax_rows()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    labels = [r[0] for r in rows]
    rates = [r[1] / 1e6 for r in rows]
    bars = ax.bar(range(len(rows)), rates, color=[r[2] for r in rows])
    for b, rate in zip(bars, rates, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, rate * 1.08, f"{rate:.2g}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("particle moves per second  [millions]")
    ax.set_title(
        "SPARTA circle case (Ar, 20x20, ~48k particles, 1000 steps from empty): "
        "forward throughput on this laptop (float64)",
        fontsize=9,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(OUT)


if __name__ == "__main__":
    main()
