"""Timing figures of the circle deck.

* ``results/timings.png``: seconds per 1000 steps, SPARTA vs sparta_jax vs the main code, and the
  per-stage split on one CPU core (``results/timings.json``; the SPARTA and main-code numbers in it
  were measured outside this script).
* ``results/step_breakdown.png``: the profiled step of sparta_jax on one CPU core
  (``results/profile_cpu1_final.json``, from ``profile.py``), stage by stage against SPARTA's own
  timers, and the XLA ops each stage executes per step.

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


STAGE_LABELS = {
    "move_walk": "move: cell walk + surface collisions (near particles)",
    "move_dense": "move: straight move of every particle",
    "move_select": "move: pick the near particles (lane compaction)",
    "sort": "sort (partition sort by cell)",
    "emit": "emit/face (inflow)",
    "collide": "collide (VSS NTC, 0 attempts at this Kn)",
    "other": "other (step glue, outside the named stages)",
}
SPARTA_STAGE = {"move_walk": "move", "sort": "sort", "emit": "emit/inflow", "collide": "collide"}


def breakdown():
    """Where one sparta_jax step goes on one CPU core, against SPARTA serial."""
    prof = load("profile_cpu1_final.json")
    sparta = load("timings.json")["stages_cpu1"]["SPARTA serial (x1.26 to steady state)"]
    stages = dict(prof["stages"])
    order = [s for s in STAGE_LABELS if s in stages]
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1.6, 1]}, facecolor="#fcfcfb"
    )
    y = list(range(len(order)))[::-1]
    h = 0.38
    for yi, s in zip(y, order, strict=True):
        ms = stages[s]["ms_per_step"]
        ax.barh(yi + h / 2, ms, height=h, color=ORANGE, edgecolor="#fcfcfb", linewidth=1.5)
        ax.text(ms + 0.01, yi + h / 2, f"{ms:.2f}", va="center", fontsize=8, color=INK2)
        if s in SPARTA_STAGE:  # SPARTA's move timer covers all three move scopes
            ref = sparta[SPARTA_STAGE[s]]
            ax.barh(yi - h / 2, ref, height=h, color=BLUE, edgecolor="#fcfcfb", linewidth=1.5)
            note = " (whole move)" if s == "move_walk" else ""
            ax.text(ref + 0.01, yi - h / 2, f"{ref:.2f}{note}", va="center", fontsize=8,
                    color=INK2)  # fmt: skip
    ax.set_yticks(y, [STAGE_LABELS[s] for s in order], fontsize=9, color=INK)
    ax.set_xlabel("ms per step, one CPU core (steady state, ~47.6k particles)", color=INK2)
    total = sum(v["ms_per_step"] for v in stages.values())
    ax.set_title(
        f"sparta_jax step {total:.2f} ms profiled ({prof['wall_ms_per_step']:.2f} ms wall, "
        f"1.30 ms untraced) vs SPARTA serial {sum(sparta.values()):.2f} ms",
        color=INK, fontsize=10,
    )  # fmt: skip
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, BLUE)],
        ["sparta_jax (XLA:CPU)", "SPARTA serial (own timers, x1.26 to steady state)"],
        frameon=False, fontsize=8, loc="lower right",
    )  # fmt: skip

    ops = [stages[s]["ops_per_step"] for s in order]
    bx.barh(y, ops, height=0.6, color=ORANGE, edgecolor="#fcfcfb", linewidth=1.5)
    for yi, s, n in zip(y, order, ops, strict=True):
        us = stages[s]["ms_per_step"] * 1e3 / max(n, 1e-9)
        bx.text(n + 2, yi, f"{n:.0f} ops, {us:.1f} us/op", va="center", fontsize=8, color=INK2)
    bx.set_yticks(y, [""] * len(order))
    bx.set_xlim(0, max(ops) * 1.55)
    bx.set_xlabel("XLA ops executed per step", color=INK2)
    bx.set_title(f"{prof['ops_per_step']:.0f} ops per step", color=INK, fontsize=10)
    for a in (ax, bx):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(axis="x", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(RES, "step_breakdown.png")
    fig.savefig(out, dpi=150)
    print(out)


THREAD_COUNTS = (1, 2, 4, 8, 16)


def threads():
    """Seconds per 1000 steps against the hardware threads the process may use (``taskset``;
    ``results/bench_cpu_threads_<n>.json``, ``scratch/sparta_jax/threads.sh``), with XLA's
    intra-op threading off on the whole machine for comparison."""
    sec = [load(f"bench_cpu_threads_{n}.json")["seconds"] for n in THREAD_COUNTS]
    off = load("bench_cpu_all_eigen_off.json")["seconds"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor="#fcfcfb")
    ax.plot(THREAD_COUNTS, sec, "o-", color=ORANGE, lw=2, ms=7, label="sparta_jax, XLA threading on")
    ax.plot([16], [off], "s", color=BLUE, ms=8, label="16 threads, XLA intra-op threading off")
    for n, t in zip(THREAD_COUNTS, sec, strict=True):
        ax.annotate(f"{t:.2f}", (n, t), xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)  # fmt: skip
    ax.set_xscale("log", base=2)
    ax.set_xticks(THREAD_COUNTS, [str(n) for n in THREAD_COUNTS])
    ax.set_ylim(0, max(sec) * 1.2)
    ax.set_xlabel("hardware threads available (taskset; 8 cores x 2)", color=INK2)
    ax.set_ylabel("seconds per 1000 steps", color=INK2)
    ax.set_title("More threads only add hand-offs: ~3 us ops, 435 per step", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(RES, "cpu_threads.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
    breakdown()
    threads()
