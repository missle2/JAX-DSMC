"""Fitgate analysis (2026-10-31): every grad-lambda variant x collision-field basis against the
central differences of examples/mesh/results/fitgate/*.jsonl.

Per (gate, basis, variant): the seed mean of the adjoint over the mean FD at step h (and h / 2),
its 2-standard-error band, the paired (adjoint - FD) t-statistic and the per-seed spread
(one realisation's noise, what an optimiser step sees).  Writes

* examples/mesh/figures/fitgate/forest.png -- adjoint / FD per variant, one panel per gate;
* examples/mesh/figures/fitgate/spread.png -- bias against per-seed spread, the candidates;
* examples/mesh/figures/fitgate/summary.md -- the tables (per gate, and the scorecard).

    PYTHONPATH=. python3 scratch/fitgate/analyze.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path("examples/mesh/results/fitgate")
OUT = Path("examples/mesh/figures/fitgate")
TERMS = ("pathwise_x", "pathwise_v", "pathwise_tally", "score", "timing", "silhouette", "order")
GATES = (
    ("sq_spec", "squash M0.32 Kn0.5 specular"),
    ("sq_diff", "squash M0.32 Kn0.5 diffuse"),
    ("c10_spec", "circle M10 Kn0.5 specular"),
    ("c10_diff", "circle M10 Kn0.5 diffuse"),
    ("c10_diff_dt2", "circle M10 diffuse, dt/2"),
    ("c10_diff_f2", "circle M10 diffuse, N/2"),
    ("af_drag_radius", "airfoil Kn0.39 diffuse, radius"),
    ("af_drag_mode2", "airfoil Kn0.39 diffuse, cos2θ"),
)
BASES = {"base": "#2a78d6", "pool1": "#eb6834", "pool2": "#1baf7a"}
# the estimators worth carrying forward (the rest: the scorecard table only)
CANDIDATES = (
    "one_pool_plane",
    "along_pool_plane",
    "cell_pool_plane",
    "cell_along_pool_plane",
    "cell_along_plane",
    "cellxv_pool_plane",
)


def load():
    """``{(gate, basis): [row per seed]}``, the last row of a repeated seed winning."""
    by = collections.defaultdict(dict)
    for path in sorted(RESULTS.glob("*.jsonl")):
        if "smoke" in path.name:
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            by[(row["gate"], row["basis"])][row["seed"]] = row
    return {k: [v[s] for s in sorted(v)] for k, v in by.items()}


def total(row, variant):
    return sum(row["terms"][variant][k] for k in TERMS) + row["volume"]


def stats(rows, variant):
    """Adjoint / FD_h, its 2 s.e., adjoint / FD_h/2, paired t, per-seed sd / |FD|, n."""
    n = len(rows)
    a = np.array([total(r, variant) for r in rows])
    fd = np.array([r["fd"][0] for r in rows])
    fd2 = np.array([r["fd"][1] for r in rows])
    m = fd.mean()
    d = a - fd
    return {
        "ratio": a.mean() / m,
        "ratio_2se": 2 * a.std(ddof=1) / np.sqrt(n) / abs(m),
        "ratio_h2": a.mean() / fd2.mean(),
        "t": d.mean() / (d.std(ddof=1) / np.sqrt(n)),
        "spread": a.std(ddof=1) / abs(m),
        "n": n,
    }


def fd_band(rows):
    fd = np.array([r["fd"][0] for r in rows])
    fd2 = np.array([r["fd"][1] for r in rows])
    return {
        "se": fd.std(ddof=1) / np.sqrt(len(fd)) / abs(fd.mean()),
        "h2_over_h": fd2.mean() / fd.mean(),
        "h2_se": fd2.std(ddof=1) / np.sqrt(len(fd2)) / abs(fd.mean()),
    }


def forest(data, variants):
    fig, axes = plt.subplots(2, 4, figsize=(17, 10), sharey=True)
    y = np.arange(len(variants))
    for ax, (gate, title) in zip(axes.flat, GATES, strict=True):
        bases = [b for b in BASES if (gate, b) in data]
        rows0 = data[(gate, bases[0])]
        band = fd_band(rows0)
        ax.axvspan(1 - 2 * band["se"], 1 + 2 * band["se"], color="0.88", zorder=0)
        ax.axvline(1.0, color="0.45", lw=1, zorder=1)
        ax.axvline(band["h2_over_h"], color="0.45", lw=1, ls=":", zorder=1)
        for j, b in enumerate(bases):
            off = (j - (len(bases) - 1) / 2) * 0.25
            st = [stats(data[(gate, b)], v) for v in variants]
            x = np.array([s["ratio"] for s in st])
            e = np.array([s["ratio_2se"] for s in st])
            ax.errorbar(np.clip(x, 0.3, 1.7), y + off, xerr=e, fmt="o", ms=4, lw=1.2,
                        color=BASES[b], label=b, zorder=3)  # fmt: skip
            for k in np.where((x < 0.3) | (x > 1.7))[0]:
                ax.annotate(f"{x[k]:.2f}", (np.clip(x[k], 0.3, 1.7), y[k] + off), fontsize=7,
                            xytext=(3, -3), textcoords="offset points", color="0.3")  # fmt: skip
        ax.set_xlim(0.3, 1.7)
        ax.set_title(f"{title}\n(n = {len(rows0)}; FD ±2 s.e. ±{200 * band['se']:.1f} %)",
                     fontsize=9)  # fmt: skip
        ax.set_xlabel("adjoint / central difference (step h)", fontsize=8)
        ax.grid(axis="x", color="0.93", lw=0.6)
        ax.tick_params(labelsize=7)
        if gate == "af_drag_radius":
            ax.legend(fontsize=7, loc="lower right", title="collision field", title_fontsize=7)
    for ax in axes[:, 0]:
        ax.set_yticks(y, variants, fontsize=7)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Shape-adjoint fits against central differences (mean over seeds ±2 s.e.; "
                 "grey: FD's own ±2 s.e.; dotted: FD at h/2)", fontsize=11)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "forest.png", dpi=150)
    plt.close(fig)


def spread_plot(data):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    markers = dict(zip(CANDIDATES, "osD^vP", strict=True))
    for ax, (gate, title) in zip(axes.flat, GATES, strict=True):
        for b in BASES:
            if (gate, b) not in data:
                continue
            for v in CANDIDATES:
                s = stats(data[(gate, b)], v)
                ax.errorbar(s["spread"], s["ratio"], yerr=s["ratio_2se"], fmt=markers[v],
                            ms=6, color=BASES[b], mfc="white" if b != "base" else BASES[b],
                            lw=1)  # fmt: skip
        ax.axhline(1.0, color="0.45", lw=1)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("per-seed spread (sd / |FD|)", fontsize=8)
        ax.set_ylabel("adjoint / FD", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(color="0.93", lw=0.6)
    handles = [plt.Line2D([], [], ls="", marker=markers[v], color="0.3", label=v)
               for v in CANDIDATES]  # fmt: skip
    handles += [plt.Line2D([], [], ls="", marker="s", color=c, label=b) for b, c in BASES.items()]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=8)
    fig.suptitle("Bias against noise, candidate fits (filled: base field; open: pooled)",
                 fontsize=11)  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "spread.png", dpi=150)
    plt.close(fig)


def tables(data, variants):
    lines = ["# Fitgate: adjoint vs central differences", "",
             "Ratio = seed-mean adjoint / seed-mean FD at step h, ±2 s.e. of the adjoint mean; "
             "t = paired (adjoint - FD) mean / s.e.; spread = per-seed sd / |FD|.", ""]  # fmt: skip
    lines += [
        "## FD references",
        "",
        "| gate | n | FD ±2 s.e. | FD(h/2)/FD(h) |",
        "|---|---|---|---|",
    ]
    for gate, _ in GATES:
        rows = data[(gate, "base")]
        band = fd_band(rows)
        lines.append(f"| {gate} | {len(rows)} | ±{200 * band['se']:.1f} % | "
                     f"{band['h2_over_h']:.3f} ± {2 * band['h2_se']:.3f} |")  # fmt: skip
    for gate, title in GATES:
        bases = [b for b in BASES if (gate, b) in data]
        lines += ["", f"## {gate} ({title})", "",
                  "| variant | " + " | ".join(f"{b} ratio | t | spread" for b in bases) + " |",
                  "|---|" + "---|---|---|" * len(bases)]  # fmt: skip
        for v in variants:
            cells = []
            for b in bases:
                s = stats(data[(gate, b)], v)
                cells += [f"{s['ratio']:.3f} ± {s['ratio_2se']:.3f}", f"{s['t']:+.1f}",
                          f"{s['spread']:.3f}"]  # fmt: skip
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
    # scorecard: over the gates each (variant, basis) has
    lines += ["", "## Scorecard (base field over all 8 gates; pool1 over its 5: no c10_diff run)", "",
              "| variant | basis | gates | RMS(ratio - 1) | max abs(ratio - 1) | RMS t | "
              "median spread | airfoil spread (r, m2) |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for b in ("base", "pool1"):
        for v in variants:
            sts = {g: stats(data[(g, b)], v) for g, _ in GATES if (g, b) in data}
            dev = np.array([s["ratio"] - 1 for s in sts.values()])
            ts = np.array([s["t"] for s in sts.values()])
            spr = np.array([s["spread"] for s in sts.values()])
            af = " / ".join(f"{sts[g]['spread']:.2f}" for g in ("af_drag_radius", "af_drag_mode2"))
            lines.append(f"| {v} | {b} | {len(sts)} | {np.sqrt(np.mean(dev**2)):.3f} | "
                         f"{np.abs(dev).max():.3f} | {np.sqrt(np.mean(ts**2)):.1f} | "
                         f"{np.median(spr):.3f} | {af} |")  # fmt: skip
    # the shared terms of one candidate
    lines += [
        "",
        "## Terms of cell_pool_plane (base) / mean FD",
        "",
        "| gate | " + " | ".join(TERMS) + " | volume |",
        "|---|" + "---|" * (len(TERMS) + 1),
    ]
    for gate, _ in GATES:
        rows = data[(gate, "base")]
        m = np.mean([r["fd"][0] for r in rows])
        vals = [np.mean([r["terms"]["cell_pool_plane"][k] for r in rows]) / m for k in TERMS]
        vals.append(np.mean([r["volume"] for r in rows]) / m)
        lines.append(f"| {gate} | " + " | ".join(f"{x:+.3f}" for x in vals) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    variants = list(data[("sq_spec", "base")][0]["terms"])
    forest(data, variants)
    spread_plot(data)
    tables(data, variants)
    print(f"wrote {OUT}/forest.png, spread.png, summary.md")


if __name__ == "__main__":
    main()
