#!/usr/bin/env python3
"""Preliminary single-GPU analysis from synced results (cluster_logs/<site>/results).

    python3 bench/prelim_gpu.py [--results cluster_logs/alpine/results] [--out bench/figures/prelim]

Reads every platform's rows.jsonl and, for SPARTA, the timer breakdown of each run's log
(Move / Coll / Sort / Comm / Modify / Output sections of the timed run).  Writes:
  single_gpu.png / .csv   Mpsps per GPU vs particles: SPARTA as run, SPARTA without its Output
                          section, sparta_jax (median over reps; per-deck tuned JAX variant)
  sparta_sections.png/.csv SPARTA's time per step by section, 16.4M particles per GPU
"Without Output" is an estimate of what the stats-100 GPU protocol removes: the device-to-host
particle copy before every stats line (stats 10 in the CPU decks).  It is not a measured run.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLATFORMS = ["alp-h200", "alp-rtx6000", "alp-a100", "alp-l40", "alp-mi100"]
NAME = {"alp-h200": "H200", "alp-rtx6000": "RTX PRO 6000", "alp-a100": "A100 40GB", "alp-l40": "L40",
        "alp-mi100": "MI100", "pm-gpu": "A100 (Perlmutter)"}
DECKS = ["free", "collide", "sphere"]
# dataviz reference palette (light), validated: slot 1 blue = SPARTA, slot 2 orange = sparta_jax
BLUE, ORANGE = "#257de0", "#f47230"
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"
SECTIONS = ["Move", "Coll", "Sort", "Comm", "Modify", "Output"]
SEC_COLOR = {"Move": "#257de0", "Coll": "#00b894", "Sort": "#6a4ce2", "Comm": "#f2a600", "Modify": "#ee7fac",
             "Output": "#f47230"}
SEC_RE = re.compile(r"^(Move|Coll|Sort|Comm|Modify|Output)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)")


def sections(path):
    """Seconds per section of the LAST run (the timed one) in a SPARTA log."""
    out, cur = {}, {}
    try:
        for ln in open(path):
            if ln.startswith("Loop time"):
                cur = {}
                out = cur
            m = SEC_RE.match(ln)
            if m:
                cur[m.group(1)] = float(m.group(3))  # avg time
    except OSError:
        return {}
    return out


def load(results):
    data = defaultdict(lambda: defaultdict(list))  # (plat, deck, series) -> particles -> [mpsps]
    secs = defaultdict(list)  # (plat, deck) -> [sections at 16.384M]
    for plat in PLATFORMS:
        f = os.path.join(results, plat, "rows.jsonl")
        if not os.path.exists(f):
            continue
        cfg = {}
        cf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{plat}.json")
        if os.path.exists(cf):
            cfg = json.load(open(cf)).get("jax", {})
        for ln in open(f):
            r = json.loads(ln)
            if r.get("procs") != 1 or r["variant"] in ("validate",) or r.get("steps") != 1000:
                continue
            d, n = r["deck"], r["particles"]
            if r["code"] == "sparta" and r["variant"] == "kk":
                data[(plat, d, "sparta")][n].append(r["mpsps"])
                s = sections(os.path.join(results, plat, "logs", r.get("log", "")))
                if s:
                    t = sum(s.values())
                    keep = t - s.get("Output", 0.0)
                    data[(plat, d, "sparta_noout")][n].append(r["mpsps"] * t / keep)
                    if n == 16_384_000:
                        secs[(plat, d)].append({k: v / r["steps"] * 1e3 for k, v in s.items()})  # ms/step
            elif r["code"] == "jax":
                want = cfg.get("decks", {}).get(d, cfg).get("tag")
                if want and r["variant"] == want:
                    data[(plat, d, "jax")][n].append(r["mpsps"])
    return data, secs


def med(xs):
    return statistics.median(xs)


def fig_single(data, out):
    plats = [p for p in PLATFORMS if any((p, d, "sparta") in data or (p, d, "jax") in data for d in DECKS)]
    fig, axes = plt.subplots(len(plats), 3, figsize=(12, 2.6 * len(plats)), sharex=True, squeeze=False)
    rows = []
    styles = {
        "sparta": dict(color=BLUE, ls="-", marker="o", mfc=BLUE, label="SPARTA/Kokkos (as run, stats 10)"),
        "sparta_noout": dict(color=BLUE, ls="--", marker="o", mfc="white", label="SPARTA without Output (estimate)"),
        "jax": dict(color=ORANGE, ls="-", marker="s", mfc=ORANGE, label="sparta_jax (tuned)"),
    }
    for i, p in enumerate(plats):
        for j, d in enumerate(DECKS):
            ax = axes[i][j]
            ymax = 0
            for s, st in styles.items():
                pts = sorted((n, med(v)) for n, v in data.get((p, d, s), {}).items())
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ymax = max(ymax, max(ys))
                ax.plot(xs, ys, lw=2, ms=5, mec=st["color"], mew=1.5, color=st["color"], ls=st["ls"],
                        marker=st["marker"], mfc=st["mfc"], label=st["label"], zorder=3)
                k = max(range(len(ys)), key=ys.__getitem__)
                ax.annotate(f"{ys[k]:.0f}", (xs[k], ys[k]), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7.5, color=TEXT)
                for x, y in pts:
                    rows.append([NAME[p], d, s, x, round(y, 2)])
            ax.set_xscale("log")
            ax.set_ylim(0, ymax * 1.18 if ymax else 1)
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color("#bbbbbb")
            ax.tick_params(colors=TEXT2, labelsize=8)
            if i == 0:
                ax.set_title(d, fontsize=11, color=TEXT)
            if j == 0:
                ax.set_ylabel(f"{NAME[p]}\nMpsps per GPU", fontsize=9, color=TEXT)
            if i == len(plats) - 1:
                ax.set_xlabel("particles per GPU", fontsize=9, color=TEXT)
    h, lab = axes[0][1].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Single GPU throughput, Alpine (preliminary: CPU-deck stats protocol)", y=1.025, fontsize=12,
                 color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out, "single_gpu.png"), dpi=150, bbox_inches="tight")
    with open(os.path.join(out, "single_gpu.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows([["gpu", "deck", "series", "particles", "mpsps_median"]] + rows)


def fig_sections(secs, out):
    keys = [(p, d) for d in ("free", "collide", "sphere") for p in PLATFORMS if (p, d) in secs]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(keys) + 1.4))
    rows = []
    for y, (p, d) in enumerate(keys):
        s = {k: med([x.get(k, 0.0) for x in secs[(p, d)]]) for k in SECTIONS}
        left = 0.0
        for k in SECTIONS:
            if s[k] <= 0:
                continue
            ax.barh(y, s[k], left=left, height=0.62, color=SEC_COLOR[k], edgecolor="white", linewidth=2,
                    label=k if y == 0 or k not in [t.get_label() for t in ax.patches] else None)
            left += s[k]
        ax.text(left, y, f"  {left:.2f} ms", va="center", fontsize=8, color=TEXT)
        rows.append([NAME[p], d] + [round(s[k], 4) for k in SECTIONS] + [round(left, 4)])
    ax.set_yticks(range(len(keys)), [f"{d} · {NAME[p]}" for p, d in keys], fontsize=8.5, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("SPARTA time per step at 16.4M particles (ms), by timer section", fontsize=9, color=TEXT)
    ax.grid(True, axis="x", color=GRID)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEC_COLOR[k]) for k in SECTIONS]
    ax.legend(handles, SECTIONS, ncol=6, frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sparta_sections.png"), dpi=150, bbox_inches="tight")
    with open(os.path.join(out, "sparta_sections.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows([["gpu", "deck"] + [f"{k}_ms" for k in SECTIONS] + ["total_ms"]] + rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="cluster_logs/alpine/results")
    p.add_argument("--out", default="bench/figures/prelim")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data, secs = load(a.results)
    fig_single(data, a.out)
    fig_sections(secs, a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
