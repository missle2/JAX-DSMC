"""Merge squash sweep rows across files by seed (the forward is deterministic per seed, so
settings run in different jobs are paired) and summarise each setting's assembled gradient
(variant along_pool_plane): mean +- se, paired % against a reference setting, and the
gap to the per-seed FD (ppc_f4.jsonl's h = 0.2) over the seeds the setting has.

    python3 scratch/cellfit/pair.py REF_SETTING FILE.jsonl ... [--variant V] [--fd FILE]
"""

import argparse
import json
from collections import defaultdict

import numpy as np

FD = "scratch/collconv/ppc_f4.jsonl"


def grad(run, v):
    t = run["variants"][v]
    return (run["tally"] + t["x"] + t["v"] + run["volume"] + run["order"] + run["score"]
            + t["sil"] + run["timing"])


def se(a):
    a = np.asarray(a, float)
    return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("files", nargs="+")
    p.add_argument("--variant", default="along_pool_plane")
    p.add_argument("--fd", default=FD)
    a = p.parse_args()
    runs = defaultdict(dict)
    for f in a.files:
        for line in open(f):
            r = json.loads(line)
            runs[r["seed"]].update(r["runs"])
    fd = {}
    for line in open(a.fd):
        r = json.loads(line)
        if r["fd"]:
            fd[r["seed"]] = r["fd"][0]
    names = sorted({n for r in runs.values() for n in r}, key=lambda n: (len(n), n))
    print(f"variant {a.variant}; reference {a.ref}")
    print(f"{'setting':34s} {'n':>3s} {'grad':>10s} {'se':>8s} {'s/seed':>7s}  "
          f"{'vs ref (paired)':>18s}  {'vs FD (paired seeds)':>22s}")
    for n in names:
        seeds = sorted(s for s in runs if n in runs[s])
        g = np.array([grad(runs[s][n], a.variant) for s in seeds])
        secs = np.median([runs[s][n]["seconds"] for s in seeds])
        line = f"{n:34s} {len(seeds):3d} {g.mean():10.4e} {se(g):8.1e} {secs:7.0f}  "
        both = [s for s in seeds if a.ref in runs[s]]
        if n != a.ref and len(both) >= 2:
            d = np.array([grad(runs[s][n], a.variant) - grad(runs[s][a.ref], a.variant)
                          for s in both])
            m = np.mean([grad(runs[s][a.ref], a.variant) for s in both])
            line += f"{d.mean()/m*100:+7.2f}% +- {se(d)/m*100:4.2f}% ({len(both)})"
        else:
            line += " " * 24
        fs = [s for s in seeds if s in fd]
        if len(fs) >= 2:
            gf = np.array([grad(runs[s][n], a.variant) for s in fs])
            f = np.array([fd[s] for s in fs])
            d = gf - f
            line += f"  {d.mean()/f.mean()*100:+6.1f}% +- {se(d)/f.mean()*100:3.1f}% ({len(fs)})"
        print(line)
    if fd:
        f = np.array(list(fd.values()))
        print(f"FD h=0.2 over {len(f)} seeds: {f.mean():.4e} +- {se(f):.1e}")


if __name__ == "__main__":
    main()
