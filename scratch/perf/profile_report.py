"""Figures and a summary of the per-stage GPU profile (scratch/perf/profile/<tag>/<stage>/
events.npz from profile_extract.py) and the regression timings (scratch/perf/regress/*.json).

  examples/mesh/figures/performance/profile_stages.png     wall time vs GPU-busy time per stage, with
                                               kernels per step and the mean kernel length
  examples/mesh/figures/performance/profile_breakdown.png  GPU time per stage by code path
  examples/mesh/figures/performance/speedup_history.png    per-stage wall time of one steady seed across
                                               the speed-up changes (same seeds, same GPU)
  scratch/perf/profile/<tag>/summary.json

    python3 scratch/perf/profile_report.py gpu_after
"""

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "..", "..", "examples", "mesh", "figures", "performance")
STAGES = [
    ("forward", "forward (1000 steps, captured 300)", 1000),
    ("replay", "traced replay", 300),
    ("backward", "value backwards (x2)", 600),
    ("shape", "shape gradient", 300),
]
TRACER_CAP = 2_000_000  # the GPU tracer's event buffer: a stage at the cap is truncated

# code-path categories: first match wins (substrings of the XLA op path)
CATEGORIES = {
    "forward": [
        ("checkpoint/while/body", "lane passes 1..K (while loop)"),
        ("cond/branch", "hit search (cull branches)"),
        ("argsort", "cell sort"),
        ("checkpoint", "step body: pass 0, collide, inject, tallies"),
    ],
    "replay": [
        ("while/body/while/body", "lane passes 1..K (fixed trip count)"),
        ("cond/branch", "hit search (cull branches)"),
        ("while/body", "step body: pass 0, rewards, checks"),
    ],
    "backward": [
        ("qki", "fit: normal equations (einsum)"),
        ("jit(solve)", "fit: LU solve"),  # before 2026-10-02
        ("spd_solve", "fit: Cholesky solve (all columns)"),
        ("fit_rhs", "fit: right-hand sides"),
        ("top_k", "fit: k nearest (top_k)"),
        ("while/body", "partner draws, tables, gathers"),
    ],
    "shape": [
        ("transpose", "pull-back (one batched transpose)"),
        ("jvp(while)", "linearised lane passes 1..K"),
        ("qki", "grad-lambda fits"),
        ("top_k", "grad-lambda fits"),
        ("jit(solve)", "grad-lambda fits"),
        ("spd_solve", "grad-lambda fits"),
        ("fit_rhs", "grad-lambda fits"),
        ("jvp(cond)", "linearised hit search"),
        ("while/body", "step body: linearised pass 0, cotangents"),
    ],
}
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#c9c9c4"


def category(stage, op):
    for key, name in CATEGORIES[stage]:
        if key in op:
            return name
    return "other"


def load(tag):
    root = os.path.join(HERE, "profile", tag)
    walls = json.load(open(os.path.join(root, "stages.json")))["stages"]
    out = {}
    for slug, name, steps in STAGES:
        ev = np.load(os.path.join(root, slug, "events.npz"))["events"]
        tables = json.load(open(os.path.join(root, slug, "events.json")))
        ops = tables["op"]
        busy = float(ev[:, 1].sum()) / 1e9
        cats = {}
        for o, d in zip(ev[:, 3].astype(int), ev[:, 1], strict=True):
            c = category(slug, ops[o])
            cats[c] = cats.get(c, 0.0) + d / 1e9
        out[slug] = {
            "label": name,
            "wall": walls[name],
            "gpu_busy": busy,
            "kernels": int(ev.shape[0]),
            "kernels_per_step": ev.shape[0] / steps,
            "mean_kernel_us": busy / ev.shape[0] * 1e6,
            "truncated": bool(ev.shape[0] >= TRACER_CAP),
            "categories": dict(sorted(cats.items(), key=lambda t: -t[1])),
        }
    return root, out


def fig_stages(rows, path):
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    names = [r["label"] + (" *" if r["truncated"] else "") for r in rows.values()]
    y = np.arange(len(rows))[::-1]
    busy = np.array([r["gpu_busy"] for r in rows.values()])
    wall = np.array([r["wall"] for r in rows.values()])
    ax.barh(y, busy, color=BLUE, label="GPU busy")
    ax.barh(
        y, np.maximum(wall - busy, 0.0), left=busy, color=GREY, label="GPU idle (launch / sync)"
    )
    for yi, r in zip(y, rows.values(), strict=True):
        ax.text(
            r["wall"] + 0.8, yi,
            f"{r['wall']:.1f} s wall, {r['gpu_busy']:.1f} s busy; "
            f"{r['kernels_per_step']:.0f} kernels/step, {r['mean_kernel_us']:.1f} us mean",
            va="center", fontsize=7.5,
        )  # fmt: skip
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("seconds (one optimiser seed, optc settings at fnum x 4, RTX 3050 Ti, float64)")
    ax.set_xlim(0, wall.max() * 1.9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    if any(r["truncated"] for r in rows.values()):
        ax.set_title(
            "* the GPU tracer's ~2 M event cap truncated this stage: busy time is a lower bound",
            fontsize=7.5, loc="left",
        )  # fmt: skip
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_breakdown(rows, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6))
    for ax, (slug, r) in zip(axes.ravel(), rows.items(), strict=True):
        cats = list(r["categories"].items())[:6]
        labels = [c[0] for c in cats][::-1]
        vals = [c[1] for c in cats][::-1]
        ax.barh(range(len(vals)), vals, color=BLUE)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f} s ({v / r['gpu_busy'] * 100:.0f}%)", va="center", fontsize=7)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlim(0, max(vals) * 1.45)
        mark = " (truncated)" if r["truncated"] else ""
        ax.set_title(f"{r['label']}: {r['gpu_busy']:.1f} s GPU{mark}", fontsize=8.5)
        ax.set_xlabel("GPU seconds", fontsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_history(path):
    versions = [
        ("base", "before"),
        ("fix1", "1: compile once"),
        ("fix3", "3: fits at hit slots"),
        ("lean", "5a: lean record"),
        ("items4to7_local", "4-7: cull, lean 2,\nbounded fits, 1 VJP"),
        ("batch_fwd", "09-28: forward\nbatch"),
        ("merged", "09-30: inflow, level\nset, compaction"),
        ("final", "10-01: tracker,\nsort, bookkeeping"),
        ("columns", "10-02: two-column\nbackward"),
    ]
    parts = [("forward_replay", "forward + replay + host", ORANGE), ("backward", "value backwards", GREY),
             ("shape", "shape gradient", BLUE)]  # fmt: skip
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(versions))
    bottom = np.zeros(len(versions))
    totals = []
    for key, label, colour in parts:
        vals = []
        for tag, _l in versions:
            vals.append(json.load(open(os.path.join(HERE, "regress", f"{tag}.json")))[1][key])
        ax.bar(x, vals, bottom=bottom, color=colour, label=label, width=0.6)
        bottom += np.array(vals)
    totals = bottom
    for xi, t in zip(x, totals, strict=True):
        ax.text(xi, t + 5, f"{t:.0f} s", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([v[1] for v in versions], fontsize=7.5)
    ax.set_ylabel("seconds per steady seed")
    ax.set_title(
        "one optimiser seed (optc settings, fnum x 4, 15 k slots), RTX 3050 Ti", fontsize=9
    )
    ax.legend(fontsize=7.5, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(tag):
    root, rows = load(tag)
    with open(os.path.join(root, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    fig_stages(rows, os.path.join(FIGURES, "profile_stages.png"))
    fig_breakdown(rows, os.path.join(FIGURES, "profile_breakdown.png"))
    fig_history(os.path.join(FIGURES, "speedup_history.png"))
    for slug, r in rows.items():
        print(slug, {k: v for k, v in r.items() if k != "categories"})
        for c, v in r["categories"].items():
            print(f"    {v:7.2f} s  {c}")


if __name__ == "__main__":
    main(sys.argv[1])
