#!/usr/bin/env python3
"""Figures of the scaling study from results.csv (bench/parse_logs.py).

    python3 bench/plot.py bench/results/results.csv --out bench/figures

Figures (PNG + the plotted numbers as CSV next to each, the table view):
  device_<deck>.png     single device: Mpsps per device vs particles per device
  node_<deck>.png       single node: Mpsps per node vs particles per node
  strong_<deck>.png     Perlmutter strong scaling: Mpsps per node vs nodes, one panel per size
                        (ideal scaling is a horizontal line, as on sparta.github.io/bench.html)
  weak_<deck>.png       Perlmutter weak scaling: Mpsps per node vs nodes, per size per node
  portability.png       best Mpsps per device on each platform, per deck and code
  ratio.png             sparta_jax / SPARTA throughput per platform and deck (single device)

Encoding: colour = platform (fixed order, never re-assigned when a filter drops a platform);
code = line style and marker (SPARTA/Kokkos solid circles, sparta_jax dashed triangles), so
neither identity rests on colour alone.  Values are the median over reps.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# validated categorical order (dataviz palette, light): fixed per platform
PLATFORM_ORDER = ["pm-gpu", "alp-a100", "alp-h200", "alp-l40", "alp-rtx6000", "alp-mi100", "pm-cpu"]
PLATFORM_COLOR = dict(
    zip(PLATFORM_ORDER, ["#2a78d6", "#6250d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"], strict=True)
)
PLATFORM_NAME = {
    "pm-gpu": "A100 (Perlmutter)", "alp-a100": "A100 (Alpine)", "alp-h200": "H200", "alp-l40": "L40",
    "alp-rtx6000": "RTX PRO 6000", "alp-mi100": "MI100", "pm-cpu": "EPYC 7763 node",
}
CODE_STYLE = {"sparta": dict(ls="-", marker="o"), "jax": dict(ls="--", marker="^")}
CODE_NAME = {"sparta": "SPARTA/Kokkos", "jax": "sparta_jax"}
TEXT = "#2b2b2a"
MUTED = "#6b6a66"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"
DECKS = ("free", "collide", "sphere")


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "lines.markersize": 6, "font.size": 10, "legend.frameon": False,
    })


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("particles", "procs", "nodes", "steps"):
            r[k] = int(float(r[k])) if r.get(k) not in (None, "") else 1
        for k in ("mpsps", "loop_time"):
            r[k] = float(r[k])
        r["platform"] = next((p for p in PLATFORM_ORDER if r["machine"].startswith(p)), r["machine"])
    return rows


def median_by(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r["mpsps"])
    return {k: float(np.median(v)) for k, v in g.items()}


def series_plot(ax, data, xlabel, ylabel, logx=True, end_labels=True):
    """data: {(platform, code): [(x, y), ...]} -> lines, legend, direct end labels (<= 4)."""
    keys = sorted(data, key=lambda k: (PLATFORM_ORDER.index(k[0]) if k[0] in PLATFORM_ORDER else 99, k[1]))
    for k in keys:
        pts = sorted(data[k])
        x, y = zip(*pts, strict=True)
        ax.plot(x, y, color=PLATFORM_COLOR.get(k[0], MUTED), label=f"{PLATFORM_NAME.get(k[0], k[0])}, {CODE_NAME[k[1]]}",
                markeredgecolor=SURFACE, markeredgewidth=1.0, **CODE_STYLE[k[1]])
        if end_labels and len(keys) <= 4:
            ax.annotate(f"{PLATFORM_NAME.get(k[0], k[0])}, {CODE_NAME[k[1]]}", (x[-1], y[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=TEXT, va="center")
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def save(fig, out, name, table):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(out, name + ".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["platform", "code", "x", "value"])
        for (p, c), pts in sorted(table.items()):
            for x, y in sorted(pts):
                w.writerow([p, c, x, f"{y:.6g}"])


def legend_side(fig, axes):
    """One legend for the figure, right of the plots (never over the axis labels)."""
    seen = {}
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(ll, hh)
    fig.tight_layout()
    fig.legend(list(seen.values()), list(seen.keys()), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)


def fig_single(rows, out, per_node):
    """Per device (procs == 1) or per node (one node, all devices)."""
    for deck in DECKS:
        sel = [r for r in rows if r["deck"] == deck and r["nodes"] == 1 and ((r["procs"] == 1) != per_node)]
        if not sel:
            continue
        med = median_by(sel, lambda r: (r["platform"], r["code"], r["particles"], r["procs"]))
        data = defaultdict(list)
        for (p, c, n, procs), v in med.items():
            data[(p, c)].append((n if per_node else n / procs, v if per_node else v / procs))
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        series_plot(ax, data, "particles per node" if per_node else "particles per device",
                    "million particle-steps / s" + (" per node" if per_node else " per device"))
        ax.set_title(f"in.{deck}: {'one node' if per_node else 'one device'}", loc="left", fontsize=11)
        legend_side(fig, [ax])
        save(fig, out, f"{'node' if per_node else 'device'}_{deck}", data)


def fig_scaling(rows, out, kind):
    """Strong (fixed total) or weak (fixed per node) on the multi-node platforms."""
    multi = [r for r in rows if r["platform"] in ("pm-gpu", "pm-cpu")]
    for deck in DECKS:
        sel = [r for r in multi if r["deck"] == deck]
        if not sel:
            continue
        nodes_by_size = defaultdict(set)
        for r in sel:
            key = r["particles"] if kind == "strong" else r["particles"] // max(1, r["nodes"])
            nodes_by_size[key].add(r["nodes"])
        sizes = sorted(s for s, ns in nodes_by_size.items() if len(ns) >= 3)
        if not sizes:
            continue
        fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 4.0), squeeze=False)
        axes = axes[0]
        table = defaultdict(list)
        for ax, size in zip(axes, sizes, strict=True):
            data = defaultdict(list)
            for r in sel:
                key = r["particles"] if kind == "strong" else r["particles"] // max(1, r["nodes"])
                if key != size:
                    continue
                data[(r["platform"], r["code"], r["nodes"])].append(r["mpsps"] / r["nodes"])
            lines = defaultdict(list)
            for (p, c, n), v in data.items():
                lines[(p, c)].append((n, float(np.median(v))))
                table[(p, c)].append((f"{size}:{n}", float(np.median(v))))
            series_plot(ax, lines, "nodes", "million particle-steps / s per node", end_labels=False)
            ax.set_xscale("log", base=2)
            ax.set_title(f"{size / 1e6:g}M particles" + (" total" if kind == "strong" else " per node"), loc="left", fontsize=10)
        fig.suptitle(f"in.{deck}: {kind} scaling on Perlmutter (flat = ideal)", x=0.01, ha="left", fontsize=11)
        legend_side(fig, list(axes))
        save(fig, out, f"{kind}_{deck}", table)


def fig_portability(rows, out):
    """Best single-device throughput per platform; grouped bars per deck, hatched for JAX."""
    sel = [r for r in rows if r["procs"] == 1 and r["nodes"] == 1]
    if not sel:
        return
    best = defaultdict(float)
    for r in sel:
        k = (r["platform"], r["code"], r["deck"])
        best[k] = max(best[k], r["mpsps"])
    plats = [p for p in PLATFORM_ORDER if any(k[0] == p for k in best)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=False)
    table = defaultdict(list)
    for ax, deck in zip(axes, DECKS, strict=True):
        x = np.arange(len(plats))
        for j, code in enumerate(("sparta", "jax")):
            vals = [best.get((p, code, deck), 0.0) for p in plats]
            for p, v in zip(plats, vals, strict=True):
                table[(p, code)].append((deck, v))
            ax.bar(x + (j - 0.5) * 0.4, vals, width=0.38, color=[PLATFORM_COLOR[p] for p in plats],
                   hatch=None if code == "sparta" else "//", edgecolor=SURFACE, linewidth=2, label=CODE_NAME[code])
        ax.set_xticks(x, [PLATFORM_NAME[p] for p in plats], rotation=30, ha="right")
        for xi, p in zip(x, plats, strict=True):  # direct values: the colours are below 3:1 on the surface
            for j, code in enumerate(("sparta", "jax")):
                v = best.get((p, code, deck), 0.0)
                if v:
                    ax.text(xi + (j - 0.5) * 0.4, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_title(f"in.{deck}", loc="left", fontsize=10)
        ax.set_ylabel("best million particle-steps / s per device")
        ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(facecolor=MUTED, edgecolor=SURFACE, label=CODE_NAME["sparta"]),
                        Patch(facecolor=MUTED, edgecolor=SURFACE, hatch="//", label=CODE_NAME["jax"])],
               loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.suptitle("Single-device throughput (solid: SPARTA/Kokkos, hatched: sparta_jax)", x=0.01, ha="left", fontsize=11)
    save(fig, out, "portability", table)


def fig_ratio(rows, out):
    sel = [r for r in rows if r["procs"] == 1 and r["nodes"] == 1]
    best = defaultdict(float)
    for r in sel:
        best[(r["platform"], r["code"], r["deck"])] = max(best[(r["platform"], r["code"], r["deck"])], r["mpsps"])
    plats = [p for p in PLATFORM_ORDER if best.get((p, "jax", "collide")) or best.get((p, "jax", "free"))]
    if not plats:
        return
    M = np.full((len(plats), len(DECKS)), np.nan)
    for i, p in enumerate(plats):
        for j, d in enumerate(DECKS):
            a, b = best.get((p, "jax", d)), best.get((p, "sparta", d))
            if a and b:
                M[i, j] = a / b
    fig, ax = plt.subplots(figsize=(5.2, 0.6 * len(plats) + 1.4))
    # diverging around 1 (log): orange = sparta_jax slower, blue = faster, grey midpoint
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    cmap = LinearSegmentedColormap.from_list("div", ["#c2491d", "#f0c9b5", "#e6e5e0", "#b7d3f6", "#1c5cab"])
    lim = max(4.0, 1.25 * np.nanmax(np.maximum(M, 1 / M))) if np.isfinite(M).any() else 4.0
    ax.imshow(M, cmap=cmap, norm=LogNorm(1 / lim, lim), aspect="auto")
    for i in range(len(plats)):
        for j in range(len(DECKS)):
            if np.isfinite(M[i, j]):
                strong = abs(np.log(M[i, j])) > 0.8 * np.log(lim)  # dark cell: light text
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9, color=SURFACE if strong else TEXT)
    ax.set_xticks(range(len(DECKS)), [f"in.{d}" for d in DECKS])
    ax.set_yticks(range(len(plats)), [PLATFORM_NAME[p] for p in plats])
    ax.grid(False)
    ax.set_title("sparta_jax / SPARTA throughput, best single device", loc="left", fontsize=10)
    save(fig, out, "ratio", {(p, "ratio"): [(d, M[i, j]) for j, d in enumerate(DECKS)] for i, p in enumerate(plats)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--out", default="bench/figures")
    a = p.parse_args()
    style()
    rows = load(a.csv)
    fig_single(rows, a.out, per_node=False)
    fig_single(rows, a.out, per_node=True)
    fig_scaling(rows, a.out, "strong")
    fig_scaling(rows, a.out, "weak")
    fig_portability(rows, a.out)
    fig_ratio(rows, a.out)
    print(f"figures in {a.out}")


if __name__ == "__main__":
    main()
