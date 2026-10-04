"""Paired summaries of the bias-study rows (fm_exact / kn_gate jsonl).

    python3 scratch/bias/summarise.py FILE.jsonl [...]
"""

import json
import sys

import numpy as np


def stat(a):
    a = np.asarray(a, float)
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
    return f"{a.mean(): .4e} +- {se:.1e}"


def fm(rows):
    fd = np.array([np.mean(r["fd"]) for r in rows])
    tally = np.array([r["tally"] for r in rows])
    ex = {k: np.array([r["exact"][k] for r in rows]) for k in rows[0]["exact"]}
    print(f"  n={len(rows)}  FD {stat(fd)}")
    print(f"  JVP objective / forward objective - 1: "
          f"{stat([r['jvp_objective'] / r['objective'] - 1 for r in rows])}")
    for k, v in ex.items():
        print(f"  exact {k:12s} {stat(v)}   minus FD {stat(v - fd)}")
    for h in ("h400", "h600", "full"):
        if f"gz_{h}" in ex:
            print(f"  horizon part ({h} - trunc): {stat(ex[f'gz_{h}'] - ex['gz_trunc'])}")
    print(f"  grazing cut-off part (trunc gz - exact): {stat(ex['gz_trunc'] - ex['exact_trunc'])}")
    print(f"  tally {stat(tally)}  timing {stat([r['timing'] for r in rows])}")
    for n in rows[0]["variants"]:
        xv = np.array([r["variants"][n]["x"] + r["variants"][n]["v"] for r in rows])
        tim = np.array([r["timing"] for r in rows])
        print(f"   {n:16s} x+v {stat(xv)}   grad - FD {stat(tally + xv - fd)}"
              f"   +timing - FD {stat(tally + xv + tim - fd)}")


def identity_check(rows):
    """The along-flight identity against the fitted x-slope: the x part of a fit variant
    minus that of the same fit with the parallel component replaced by the identity (with
    and without the -L* lambda term).  The identity that is right reads ~0."""
    pairs = (("one", "along"), ("one_pool", "along_pool"), ("one_pool_plane", "along_pool_plane"))
    for fit, ident in pairs:
        if f"{ident}_free" not in rows[0]["variants"]:
            continue
        xf = np.array([r["variants"][fit]["x"] for r in rows])
        xi = np.array([r["variants"][ident]["x"] for r in rows])
        x0 = np.array([r["variants"][f"{ident}_free"]["x"] for r in rows])
        print(f"   identity check {fit:15s}: fit - (T + L*) {stat(xf - xi)}   "
              f"fit - (T only) {stat(xf - x0)}")


def kn(rows):
    fd = np.array([np.mean(r["fd"]) for r in rows])
    fz = np.array([np.mean(r["fd_frozen"]) for r in rows])
    vol = np.array([r["volume"] for r in rows])
    tally = np.array([r["tally"] for r in rows])
    print(f"  n={len(rows)} virtual={rows[0]['virtual']} config={rows[0]['config']}")
    print(f"  FD {stat(fd)}  FD frozen {stat(fz)}  FD - frozen {stat(fd - fz)}")
    print(f"  adjoint volume term {stat(vol)}   (FD-frozen) - volume {stat(fd - fz - vol)}")
    print(f"  order term {stat([r.get('order', 0.0) for r in rows])}")
    print(f"  tally {stat(tally)}  score {stat([r['score'] for r in rows])}  "
          f"silhouette {stat([r['silhouette'] for r in rows])}  "
          f"timing {stat([r['timing'] for r in rows])}")
    for n in rows[0]["variants"]:
        x = np.array([r["variants"][n]["x"] for r in rows])
        v = np.array([r["variants"][n]["v"] for r in rows])
        sil = np.array([r["variants"][n].get("sil", r["silhouette"]) for r in rows])
        score = np.array([r["score"] for r in rows])
        order = np.array([r.get("order", 0.0) for r in rows])
        g = tally + x + v + vol + score + sil + order
        tim = np.array([r["timing"] for r in rows])
        print(f"   {n:16s} x {stat(x)} v {stat(v)}  grad {stat(g)}  grad - FD {stat(g - fd)}"
              f"  +timing - FD {stat(g + tim - fd)}")
    identity_check(rows)


for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path)]
    print(path)
    for s in sorted({r["s"] for r in rows}):
        sub = [r for r in rows if r["s"] == s]
        print(f" s = {s}")
        (fm if "exact" in sub[0] else kn)(sub)
