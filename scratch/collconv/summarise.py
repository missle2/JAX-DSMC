"""Summaries of sweep.py rows: per setting and variant, the assembled gradient
(tally + x + v + volume + order + score + silhouette + timing) as mean +- se, its paired
difference from the first setting, and its gap to an FD reference.

    python3 scratch/collconv/summarise.py FILE.jsonl [--fd VALUE SE] [--variants a,b,..]
"""

import argparse
import json

import numpy as np

DEFAULT = "along_pool_plane"


def stat(a):
    a = np.asarray(a, float)
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
    return a.mean(), se


def grad(run, variant):
    t = run["variants"][variant]
    return (run["tally"] + t["x"] + t["v"] + run["volume"] + run["order"] + run["score"]
            + t["sil"] + run["timing"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--fd", type=float, nargs=2, default=None)
    p.add_argument("--variants", default=None)
    a = p.parse_args()
    rows = [json.loads(line) for f in a.files for line in open(f)]
    settings = list(rows[0]["runs"])
    rows = [r for r in rows if all(s in r["runs"] for s in settings)]
    variants = a.variants.split(",") if a.variants else list(rows[0]["runs"][settings[0]]["variants"])
    fd_rows = [np.mean(r["fd"]) for r in rows if r["fd"]]
    print(f"n={len(rows)} config={rows[0]['config']}")
    if fd_rows:
        m, se = stat(fd_rows)
        print(f"  paired FD {rows[0]['fd_steps']}: {m:.4e} +- {se:.1e}")
    ref_fd = a.fd
    base = settings[0]
    for s in settings:
        secs = np.mean([r["runs"][s]["seconds"] for r in rows])
        print(f" {s}  ({secs:.0f} s)  tally {stat([r['runs'][s]['tally'] for r in rows])[0]:.4e}"
              f"  volume {stat([r['runs'][s]['volume'] for r in rows])[0]:.3e}"
              f"  order {stat([r['runs'][s]['order'] for r in rows])[0]:.2e}")
        for v in variants:
            g = np.array([grad(r["runs"][s], v) for r in rows])
            x = np.array([r["runs"][s]["variants"][v]["x"] for r in rows])
            vv = np.array([r["runs"][s]["variants"][v]["v"] for r in rows])
            d = g - np.array([grad(r["runs"][base], v) for r in rows])
            line = (f"   {v:22s} x {x.mean(): .3e} v {vv.mean(): .3e}  grad {g.mean():.4e} +- "
                    f"{stat(g)[1]:.1e}")
            if s != base:
                line += f"  - {base} {d.mean()/abs(g.mean())*100:+.2f}% +- {stat(d)[1]/abs(g.mean())*100:.2f}%"
            if ref_fd:
                gap = g.mean() / ref_fd[0] - 1
                err = np.hypot(stat(g)[1] / ref_fd[0], ref_fd[1] / ref_fd[0])
                line += f"  vs FD {gap*100:+.1f}% +- {err*100:.1f}%"
            print(line)


if __name__ == "__main__":
    main()
