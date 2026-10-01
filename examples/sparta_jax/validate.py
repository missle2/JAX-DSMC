"""Validate sparta_jax against SPARTA on the circle deck (in.circle_ar), several seeds each.

Observables (all from the same deck, 1000 steps from an empty box):
  * stats: particle count every 10 steps, and the run totals SPARTA prints (surface collisions,
    surface checks, boundary exits, boundary collides);
  * compute surf + fix ave/surf over steps 701-1000: per-line hit count and force, total drag;
  * compute grid + fix ave/grid over steps 701-1000: per-cell n, nrho, u, v, temp.
Agreement is judged against the seed-to-seed scatter of both codes (z-scores).

    run_locked.sh gpu python3 examples/sparta_jax/validate.py --seeds 8
Reference SPARTA runs: examples/sparta_jax/sparta_ref (seeds 101-108, spa_serial, in.circle_ar).
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from sparta_jax import geometry as G  # noqa: E402

REF = os.path.join(HERE, "sparta_ref")
BOLTZ = G.BOLTZ


def last_block(path):
    """Rows of the last dump block of a SPARTA grid/surf dump file."""
    lines = open(path).read().splitlines()
    k = max(
        i
        for i, ln in enumerate(lines)
        if ln.startswith("ITEM: SURFS") or ln.startswith("ITEM: CELLS")
    )
    return np.array([[float(t) for t in ln.split()] for ln in lines[k + 1 :] if ln.strip()])


def sparta_log(path):
    txt = open(path).read()
    rows = re.findall(
        r"^\s+(\d+)\s+[\d.e+-]+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
        txt,
        re.M,
    )
    st = np.array(rows, dtype=np.int64)
    st = st[st[:, 0] > 0]
    tot = {}
    for key, pat in (
        ("moves", r"Particle moves\s+=\s+(\d+)"),
        ("nscheck", r"SurfColl checks\s+=\s+(\d+)"),
        ("nscoll", r"SurfColl occurs\s+=\s+(\d+)"),
        ("nexit", r"Boundary exits\s+=\s+(\d+)"),
        ("nbound", r"Boundary collides\s+=\s+(\d+)"),
    ):
        tot[key] = int(re.search(pat, txt).group(1))
    return st, tot


def load_sparta():
    out = {"np": [], "tot": [], "surf": [], "grid": []}
    for log in sorted(glob.glob(os.path.join(REF, "log.circle_ar.*"))):
        seed = log.rsplit(".", 1)[1]
        st, tot = sparta_log(log)
        out["np"].append(st[:, 1])
        out["steps"] = st[:, 0]
        out["tot"].append(tot)
        s = last_block(os.path.join(REF, f"surf_ave.circle_ar.{seed}.txt"))
        s = s[np.argsort(s[:, 0])]
        # columns: id v1x v1y v2x v2y area n press px py pz shx shy shz fx fy fz ke
        out["surf"].append({"n": s[:, 6], "fx": s[:, 14], "fy": s[:, 15]})
        g = last_block(os.path.join(REF, f"grid_ave.circle_ar.{seed}.txt"))
        g = g[np.argsort(g[:, 0])]
        # columns: id xc yc n nrho u v w temp
        out["grid"].append(
            {"n": g[:, 3], "nrho": g[:, 4], "u": g[:, 5], "v": g[:, 6], "temp": g[:, 8]}
        )
    return out


def run_jax(seeds, sort):
    import jax
    import jax.numpy as jnp

    from sparta_jax import sim

    cfg, consts, geo = sim.make(tally=True, sort=sort)
    out = {"np": [], "tot": [], "surf": [], "grid": [], "platform": jax.devices()[0].platform}
    for seed in seeds:
        t0 = time.perf_counter()
        st = sim.empty_state(cfg)
        st, stats = sim.run(st, seed, jnp.int32(0), consts, cfg, 1000, tally_from=jnp.int32(700))
        s = sim.as_numpy_stats(stats)
        assert int(s["overflow"].sum()) == 0 and int(s["emit_fail"].sum()) == 0
        out["np"].append(s["np"][9::10])
        out["tot"].append(
            {
                "moves": sim.moves(stats),
                "nscheck": int(s["nscheck"].sum()),
                "nscoll": int(s["nscoll"].sum()),
                "nexit": int(s["nexit"].sum()),
                "nbound": int(s["nbound"].sum()),
                "nstuck": int(s["nstuck"].sum()),
                "nattempt": int(s["nattempt"].sum()),
            }
        )
        t = st.tally
        ns = float(t.nsample)
        surf = np.asarray(t.surf)
        scale = cfg.fnum / cfg.dt / ns
        out["surf"].append(
            {"n": surf[:, 0] / ns, "fx": surf[:, 1] * scale, "fy": surf[:, 2] * scale}
        )
        g = np.asarray(t.grid)
        n = g[:, 0]
        nn = np.where(n > 0, n, 1.0)
        vol = np.where(geo.volume > 0, geo.volume, 1.0)
        u, v = g[:, 1] / nn, g[:, 2] / nn
        # compute grid 'temp' is m <|v|^2> / (3 k), streaming part included (compute_grid.cpp)
        temp = cfg.mass / (3 * BOLTZ) * g[:, 4] / nn
        out["grid"].append(
            {
                "n": n / ns,
                "nrho": n / ns * cfg.fnum / vol,
                "u": np.where(n > 0, u, 0.0),
                "v": np.where(n > 0, v, 0.0),
                "temp": np.where(n > 0, temp, 0.0),
            }
        )
        print(f"seed {seed}: {time.perf_counter() - t0:.1f} s, np {int(s['np'][-1])}", flush=True)
    return out, geo


def mean_sem(a):
    a = np.asarray(a, float)
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))


def compare(j, s, geo):
    res = {}
    # particle count curve
    mj, ej = mean_sem(j["np"])
    ms, es = mean_sem(s["np"])
    z = (mj - ms) / np.sqrt(ej**2 + es**2 + 1e-12)
    res["np"] = {
        "steps": s["steps"].tolist(),
        "jax_mean": mj.tolist(),
        "sparta_mean": ms.tolist(),
        "jax_sem": ej.tolist(),
        "sparta_sem": es.tolist(),
        "final_jax": [float(mj[-1]), float(ej[-1])],
        "final_sparta": [float(ms[-1]), float(es[-1])],
        "max_abs_z": float(np.max(np.abs(z))),
        "frac_abs_z_gt_3": float(np.mean(np.abs(z) > 3)),
        "rms_z": float(np.sqrt(np.mean(z**2))),
    }
    # run totals
    tot = {}
    for key in ("moves", "nscheck", "nscoll", "nexit", "nbound"):
        a = [t[key] for t in j["tot"]]
        b = [t[key] for t in s["tot"]]
        (mja, eja), (msa, esa) = mean_sem(a), mean_sem(b)
        tot[key] = {
            "jax": [float(mja), float(eja)],
            "sparta": [float(msa), float(esa)],
            "rel_diff": float(mja / msa - 1),
            "z": float((mja - msa) / np.sqrt(eja**2 + esa**2)),
        }
    res["totals"] = tot
    # drag and per-line loads
    for q in ("fx", "fy"):
        a = [x[q].sum() for x in j["surf"]]
        b = [x[q].sum() for x in s["surf"]]
        (mja, eja), (msa, esa) = mean_sem(a), mean_sem(b)
        res["force_" + q] = {
            "jax": [float(mja), float(eja)],
            "sparta": [float(msa), float(esa)],
            "z": float((mja - msa) / np.sqrt(eja**2 + esa**2)),
        }
    for q in ("n", "fx", "fy"):
        a, b = np.array([x[q] for x in j["surf"]]), np.array([x[q] for x in s["surf"]])
        (mja, eja), (msa, esa) = mean_sem(a), mean_sem(b)
        zz = (mja - msa) / np.sqrt(eja**2 + esa**2 + 1e-300)
        res["surf_" + q] = {
            "rms_z": float(np.sqrt(np.mean(zz**2))),
            "max_abs_z": float(np.max(np.abs(zz))),
        }
    # grid fields over flow cells with particles
    flow = geo.volume > 0
    for q in ("nrho", "u", "v", "temp"):
        a, b = np.array([x[q] for x in j["grid"]]), np.array([x[q] for x in s["grid"]])
        (mja, eja), (msa, esa) = mean_sem(a), mean_sem(b)
        den = np.sqrt(eja**2 + esa**2)
        ok = flow & (den > 0)
        zz = (mja - msa)[ok] / den[ok]
        res["grid_" + q] = {
            "cells": int(ok.sum()),
            "rms_z": float(np.sqrt(np.mean(zz**2))),
            "max_abs_z": float(np.max(np.abs(zz))),
            "frac_abs_z_gt_3": float(np.mean(np.abs(zz) > 3)),
            "mean_rel_diff": float(np.mean((mja - msa)[ok]) / np.mean(np.abs(msa[ok]))),
        }
        res["grid_" + q + "_fields"] = {"jax": mja.tolist(), "sparta": msa.tolist()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--sort", default="partition")
    args = ap.parse_args()
    s = load_sparta()
    j, geo = run_jax(list(range(1001, 1001 + args.seeds)), args.sort)
    res = compare(j, s, geo)
    res["platform"] = j["platform"]
    res["n_seeds"] = {"jax": len(j["np"]), "sparta": len(s["np"])}
    with open(os.path.join(HERE, "results", "validate.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    brief = {k: v for k, v in res.items() if not k.endswith("_fields") and k != "np"}
    brief["np"] = {k: v for k, v in res["np"].items() if not isinstance(v, list) or len(v) < 5}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
