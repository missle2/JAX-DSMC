"""Tables and figures from the P3 scale-up artifacts (``scratch/bench/run_p3.sh``).

python3 -m examples.p3_replication.report [--plot-only]
"""

import argparse
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
PAPER_INLET = (0.15171883, 0.15157001)
PAPER_INLET_SE = (0.00012830, 0.00023059)
PAPER_COUETTE = (3.0283, 3.0011)
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def load(name):
    path = os.path.join(RESULTS, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _pm(mean, se, digits=4):
    if se is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def table_inlet():
    names = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(RESULTS, "p1_*.json")))
    if not names:
        return None
    lines = [
        "| run | N₀ | retained | partners | neighbors | recon | seeds | G_L (AD) | G_R (AD) | G_L (FD) | G_R (FD) | s/seed |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        d = load(name)
        c = d["config"]
        ad, fd = d["adjoint_normal_temperature"], d["finite_difference_normal_temperature"]
        se_ad = ad["se"] or [None, None]
        se_fd = fd["se"] or [None, None]
        lines.append(
            f"| {name[3:]} | {c['n_initial']} | {c['retained']} | {c['partners']} | {c['neighbors']} | {c['reconstruction']} | {d['seeds']} "
            f"| {_pm(ad['mean'][0], se_ad[0])} | {_pm(ad['mean'][1], se_ad[1])} | {_pm(fd['mean'][0], se_fd[0], 3)} | {_pm(fd['mean'][1], se_fd[1], 3)} "
            f"| {d['elapsed_seconds'] / d['seeds']:.0f} |"
        )
    lines.append(
        f"\npaper (N0 = 2e6, 32 000 retained/cell, 8 seeds): G_L = {PAPER_INLET[0]} ± {PAPER_INLET_SE[0]}, G_R = {PAPER_INLET[1]} ± {PAPER_INLET_SE[1]}"
    )
    return "\n".join(lines)


def table_walls():
    names = sorted(
        os.path.basename(p)[:-5] for p in glob.glob(os.path.join(RESULTS, "p[23]_*.json"))
    )
    if not names:
        return None
    lines = [
        "| run | case | particles | seeds | adjoint | deterministic FD | regularised FD | objective |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        d = load(name)
        c = d["config"]

        def fmt(block):
            m, se = block["mean"], block["se"] or [None] * len(block["mean"])
            return ", ".join(_pm(a, b, 3) for a, b in zip(m, se, strict=True))

        lines.append(
            f"| {name} | {c['observable']} | {c['particles']} | {d['seeds']} | {fmt(d['adjoint'])} | {fmt(d['deterministic_fd'])} | {fmt(d['regularized_fd'])} | {d['objective']['mean']:.4f} |"
        )
    lines.append(f"\npaper Couette (Test 4): adjoint {PAPER_COUETTE[0]}, FD {PAPER_COUETTE[1]}")
    return "\n".join(lines)


def figure_inlet(path):
    names = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(RESULTS, "p1_*.json")))
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    for side, color, label in ((0, SERIES[0], "G_L adjoint"), (1, SERIES[1], "G_R adjoint")):
        ys = [load(n)["adjoint_normal_temperature"]["mean"][side] for n in names]
        es = [(load(n)["adjoint_normal_temperature"]["se"] or [0, 0])[side] for n in names]
        ax.errorbar(
            range(len(names)),
            ys,
            yerr=es,
            fmt="o",
            color=color,
            ms=5,
            capsize=3,
            lw=1.5,
            label=label,
        )
        ax.axhline(PAPER_INLET[side], color=color, lw=1, ls="--", label=f"paper {label.split()[0]}")
    for side, color in ((0, SERIES[2]), (1, SERIES[3])):
        ys = [load(n)["finite_difference_normal_temperature"]["mean"][side] for n in names]
        es = [
            (load(n)["finite_difference_normal_temperature"]["se"] or [0, 0])[side] for n in names
        ]
        ax.errorbar(
            np.arange(len(names)) + 0.15,
            ys,
            yerr=es,
            fmt="s",
            color=color,
            ms=4,
            capsize=2,
            lw=1,
            alpha=0.8,
            label=f"fixed-count FD, side {side}",
        )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[3:] for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("∂J₃/∂T_normal")
    ax.grid(color="#e6e5e1", lw=0.6)
    ax.legend(fontsize=7, ncol=3)
    ax.set_title(
        "P3 Test 3 inlet gradient — JAX backend, one budget varied per run (base: N₀=4k, R=512, P=8, K=32, ballistic)",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(FIGURES, exist_ok=True)
    made = figure_inlet(os.path.join(FIGURES, "p3_inlet_convergence.png"))
    if made:
        print(f"wrote {made}")
    if args.plot_only:
        return
    for title, table in (
        ("P3 Test 3 inlet", table_inlet()),
        ("P3 walls (Tests 4 / 1)", table_walls()),
    ):
        if table:
            print(f"\n### {title}\n\n{table}")


if __name__ == "__main__":
    main()
