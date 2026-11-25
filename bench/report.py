#!/usr/bin/env python3
"""Tables for the SPARTA developers, laid out like sparta.github.io/bench.html.

    python3 bench/report.py bench/results/results.csv --logs bench/results --out bench/report

Writes ``<out>/index.md`` (and ``index.html``) with, per benchmark (free, collide, sphere):
  * "best" tables -- one device, one node, strong scaling, weak scaling -- in million
    particle-steps / s (per node for the scaling tables, so ideal scaling is a constant);
  * the detailed rows: machine, code/variant, size, nodes, procs, Mpsps, the launch command
    and the log file (copied into ``<out>/logs``).
Values are medians over reps; the best variant per (machine, code) is the one with the highest
median.  Also copies bench/env build recipes and bench/configs/*.json (the tuned settings).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
from collections import defaultdict

import numpy as np

DECKS = ("free", "collide", "sphere")


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["particles"] = int(float(r["particles"]))
        r["procs"] = int(float(r["procs"] or 1))
        r["nodes"] = int(float(r.get("nodes") or 1))
        r["mpsps"] = float(r["mpsps"])
    return rows


def size(n):
    return f"{n / 1e6:g}M" if n >= 1e6 else f"{n / 1e3:g}K"


def med_rows(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["machine"], r["code"], r["variant"], r["deck"], r["particles"], r["nodes"], r["procs"])].append(r)
    out = []
    for k, rs in g.items():
        m = dict(rs[0])
        m["mpsps"] = float(np.median([r["mpsps"] for r in rs]))
        m["reps"] = len(rs)
        out.append(m)
    return out


def table(title, header, body):
    lines = [f"#### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines) + "\n"


def section(deck, rows):
    rs = [r for r in rows if r["deck"] == deck]
    out = [f"## in.{deck}\n"]
    # best per (machine, code): one device
    for title, sel, per in (
        ("One device (best size)", lambda r: r["procs"] == 1 and r["nodes"] == 1, "device"),
        ("One node (best size)", lambda r: r["nodes"] == 1 and r["procs"] > 1, "node"),
    ):
        best = {}
        for r in rs:
            if sel(r):
                k = (r["machine"], r["code"])
                if k not in best or r["mpsps"] > best[k]["mpsps"]:
                    best[k] = r
        body = [[m, c, b["variant"], size(b["particles"]), b["procs"], f"{b['mpsps']:.1f}", f"`{b['log']}`"]
                for (m, c), b in sorted(best.items())]
        if body:
            out.append(table(title, ["machine", "code", "variant", "size", "procs", f"Mpsps per {per}", "log"], body))
    for kind in ("strong", "weak"):
        series = defaultdict(dict)
        for r in rs:
            if r["machine"] not in ("pm-gpu", "pm-cpu") and not r["machine"].startswith("pm-"):
                continue
            key = r["particles"] if kind == "strong" else r["particles"] // max(1, r["nodes"])
            series[(r["machine"], r["code"], key)][r["nodes"]] = r["mpsps"] / r["nodes"]
        nodes = sorted({n for s in series.values() for n in s})
        body = []
        for (m, c, key), s in sorted(series.items()):
            if len(s) < 2:
                continue
            body.append([m, c, size(key)] + [f"{s[n]:.1f}" if n in s else "" for n in nodes])
        if body:
            out.append(table(f"{kind.capitalize()} scaling, Mpsps per node ({'total' if kind == 'strong' else 'per node'} size)",
                             ["machine", "code", "size"] + [f"{n} nodes" for n in nodes], body))
    detail = [[r["machine"], r["code"], r["variant"], size(r["particles"]), r["nodes"], r["procs"], f"{r['mpsps']:.1f}",
               r.get("reps", 1), f"`{html.escape(r.get('command', '') or '')[:120]}`", f"`{r['log']}`"]
              for r in sorted(rs, key=lambda r: (r["machine"], r["code"], r["particles"], r["nodes"]))]
    out.append(table("All runs", ["machine", "code", "variant", "size", "nodes", "procs", "Mpsps", "reps", "command", "log"], detail))
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--logs", default=None)
    p.add_argument("--out", default="bench/report")
    a = p.parse_args()
    rows = med_rows(load(a.csv))
    os.makedirs(a.out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    doc = [
        "# SPARTA benchmarks: SPARTA/Kokkos and sparta_jax\n",
        "Units: million particle-steps per second (particle moves / loop time), as bench.html. "
        "Decks: SPARTA's bench/in.free, in.collide, in.sphere unchanged except the timed run length "
        "(GPU free/collide: 1000 steps, as SPARTA's *.gpu.steps decks) and the seed.  Median over reps.\n",
        "Builds and settings: `env/` (build scripts), `configs/` (tuned runtime settings per platform).\n",
    ]
    for deck in DECKS:
        doc.append(section(deck, rows))
    md = "\n".join(doc)
    with open(os.path.join(a.out, "index.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(a.out, "index.html"), "w") as fh:
        fh.write("<html><body><pre style='font-family:monospace'>" + html.escape(md) + "</pre></body></html>")
    if a.logs:
        dst = os.path.join(a.out, "logs")
        os.makedirs(dst, exist_ok=True)
        names = {r["log"] for r in load(a.csv)}
        for root, _s, files in os.walk(a.logs):
            for f in files:
                if f in names:
                    shutil.copy2(os.path.join(root, f), dst)
    for sub in ("env", "configs"):
        src = os.path.join(here, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(a.out, sub), dirs_exist_ok=True)
    print(f"report -> {a.out}/index.md ({len(rows)} median rows)")


if __name__ == "__main__":
    main()
