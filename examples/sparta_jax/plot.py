"""Timing figure: seconds per 1000 steps of the circle deck, SPARTA vs sparta_jax vs the main code,
and the per-stage split on one CPU core.  Reads examples/sparta_jax/results/*.json and the numbers
in TIMES below (SPARTA and main-code timings measured outside this script).

    python3 examples/sparta_jax/plot.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def main():
    t = load("timings.json")
    rows = t["rows"]  # [label, family, seconds]
    color = {"SPARTA (C++)": BLUE, "sparta_jax (JAX)": ORANGE, "main code (JAX)": AQUA}
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 1]}, facecolor="#fcfcfb"
    )
    y = list(range(len(rows)))[::-1]
    for yi, (label, fam, sec) in zip(y, rows, strict=True):
        ax.barh(yi, sec, color=color[fam], height=0.62, edgecolor="#fcfcfb", linewidth=2)
        ax.text(sec * 1.08, yi, f"{sec:.2f} s", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows], color=INK, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 20)
    ax.set_xlabel("seconds per 1000 steps (log scale)", color=INK2)
    ax.set_title(
        "Circle deck, 1000 steps from empty (~37 M particle moves)", color=INK, fontsize=11
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color.values()]
    ax.legend(handles, list(color), frameon=False, loc="upper right", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # per-stage split, one CPU core, ms per step
    st = t["stages_cpu1"]  # {code: {stage: ms}}
    stages = ["move", "sort", "collide", "emit/inflow", "other"]
    scol = dict(zip(stages, (BLUE, ORANGE, AQUA, YELLOW, MAGENTA), strict=True))
    codes = list(st)
    for i, code in enumerate(codes):
        left = 0.0
        for s in stages:
            w = st[code].get(s, 0.0)
            if w <= 0:
                continue
            bx.barh(i, w, left=left, color=scol[s], height=0.6, edgecolor="#fcfcfb", linewidth=2)
            if w > 0.06 * max(sum(v.values()) for v in st.values()):
                bx.text(left + w / 2, i, f"{w:.1f}", ha="center", va="center", fontsize=8,
                        color=INK)  # fmt: skip
            left += w
        bx.text(left + 0.3, i, f"{left:.1f} ms", va="center", color=INK2, fontsize=9)
    bx.set_yticks(range(len(codes)), codes, fontsize=9, color=INK)
    bx.invert_yaxis()
    bx.set_xlabel("ms per step, one CPU core (profiled, steady state)", color=INK2)
    bx.set_title("Where a step goes (1 core)", color=INK, fontsize=11)
    bx.legend(
        [plt.Rectangle((0, 0), 1, 1, color=scol[s]) for s in stages], stages, frameon=False,
        fontsize=8, loc="upper right",
    )  # fmt: skip
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.grid(axis="x", color=GRID, linewidth=0.8)
    bx.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(RES, "timings.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
