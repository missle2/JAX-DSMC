"""The route-(b) verification experiments on P3's own model (review section 5.2).

    python3 -m examples.p3_replication.route_b_experiments two-step  --seeds 240 --a-seeds 16
    python3 -m examples.p3_replication.route_b_experiments compare   --n0 2000 --seeds 16
    python3 -m examples.p3_replication.route_b_experiments scale     --n0 16000 --seeds 32
    python3 -m examples.p3_replication.route_b_experiments fd        --n0 16000 --seeds 1600 --workers 10
    python3 -m examples.p3_replication.route_b_experiments recon-bias --retained 512 --n0 16000

* ``two-step``: N0 = 20 000, two steps, strong collisions (q0 = 4); route (b) with and
  without interpolation (``exact_last``), route (a) at full retention, fixed-count CRN FD
  at three steps.  Writes ``results/route_b/arbiter_two_step.json``.
* ``compare``: 20 steps at the paper's setting, small N0; (a) at R = 512 and (b) on the
  same trajectories with the FD.  Writes ``results/route_b/compare_ab_N{n0}_R{R}.json``.
* ``scale``: (b) alone at large N0 (the interpolation-free answer for the transcribed
  model).  Writes ``results/route_b/route_b_N{n0}.json`` (``--seed-start`` appends a
  second file ``_b``).
* ``fd``: the fixed-count CRN finite difference at large N0 in parallel workers.
  Writes ``results/route_b/fd_N{n0}.json``.
* ``recon-bias``: affine reconstruction forced on the exactly known collisionless field.

``route_b_report.py`` turns the artifacts into the agreement figure.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from jax_dsmc import ensemble_inflow as ei
from jax_dsmc import ensemble_particle as ep

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results", "route_b")
T4 = np.full((2, 3), 4.0)


def _fd(cfg, seed, counts, h=0.1):
    fd = []
    for side in (0, 1):
        plus, minus = T4.copy(), T4.copy()
        plus[side, 0] += h
        minus[side, 0] -= h
        fd.append(
            (
                ep.forward(cfg, plus, seed=seed, injection_counts=counts).objective
                - ep.forward(cfg, minus, seed=seed, injection_counts=counts).objective
            )
            / (2 * h)
        )
    return fd


def two_step(args):
    cfg = ei.Config(
        n_initial=20000,
        cells=20,
        dt=0.05,
        steps=2,
        retained=4000,
        source_samples=100000,
        partners=8,
        neighbors=32,
        q0=4.0,
        reconstruction="ballistic",
    )
    counts = (5000, 5000)
    rows = []
    for seed in range(args.seeds):
        tb = ep.forward(cfg, T4, seed=seed, injection_counts=counts)
        row = {
            "seed": seed,
            "b_exact": ep.backward(tb, exact_last=True)[:, 0].tolist(),
            "b": ep.backward(tb)[:, 0].tolist(),
            "fd": {str(h): _fd(cfg, seed, counts, h) for h in (0.2, 0.1, 0.05)},
        }
        if seed < args.a_seeds:
            row["a"] = ei.backward(
                ei.forward(cfg, T4, seed=seed, injection_counts=counts), seed=seed
            )[:, 0].tolist()
        rows.append(row)
        print(
            f"seed {seed}: b_exact={np.round(row['b_exact'], 4)} fd0.1={np.round(row['fd']['0.1'], 4)}",
            flush=True,
        )
        _dump("arbiter_two_step.json", {"config": cfg.__dict__, "rows": rows})


def compare(args):
    cfg = ei.Config(
        n_initial=args.n0,
        cells=40,
        dt=0.025,
        steps=20,
        retained=args.retained,
        source_samples=100000,
        partners=8,
        neighbors=32,
        reconstruction="ballistic",
    )
    counts = ei.nominal_counts(cfg, T4)
    rows = []
    for seed in range(args.seeds):
        t0 = time.time()
        tb = ep.forward(cfg, T4, seed=seed, injection_counts=counts)
        gb = ep.backward(tb)[:, 0]
        tb_time = time.time() - t0
        t0 = time.time()
        ta = ei.forward(cfg, T4, seed=seed, injection_counts=counts)
        ga = ei.backward(ta, seed=1000 + seed)[:, 0]
        ta_time = time.time() - t0
        assert ta.objective == tb.objective
        rows.append(
            {
                "seed": seed,
                "a": ga.tolist(),
                "b": gb.tolist(),
                "fd": _fd(cfg, seed, counts),
                "objective": tb.objective,
                "s_a": ta_time,
                "s_b": tb_time,
            }
        )
        print(
            f"seed {seed}: a={ga.round(4)} b={gb.round(4)} ({ta_time:.0f}s / {tb_time:.1f}s)",
            flush=True,
        )
        _dump(
            f"compare_ab_N{args.n0}_R{args.retained}.json",
            {"config": cfg.__dict__, "counts": counts, "fd_step": 0.1, "rows": rows},
        )


def scale(args):
    cfg = ei.Config(
        n_initial=args.n0,
        cells=40,
        dt=0.025,
        steps=20,
        retained=8,
        source_samples=8,
        partners=1,
        neighbors=32,
    )
    counts = ei.nominal_counts(cfg, T4)
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        t0 = time.time()
        tb = ep.forward(cfg, T4, seed=seed, injection_counts=counts)
        g = ep.backward(tb)[:, 0]
        rows.append(
            {
                "seed": seed,
                "b": g.tolist(),
                "fd": _fd(cfg, seed, counts),
                "objective": tb.objective,
                "seconds": time.time() - t0,
            }
        )
        print(f"seed {seed}: b={g.round(4)} ({time.time() - t0:.0f}s)", flush=True)
        _dump(
            f"route_b_N{args.n0 // 1000}k{'_b' if args.seed_start else ''}.json",
            {
                "n_initial": args.n0,
                "counts": counts,
                "normalization": tb.normalization,
                "rows": rows,
            },
        )


_FD_CFG = {}


def _fd_worker(seed):
    return {"seed": seed, "fd": _fd(_FD_CFG["cfg"], seed, _FD_CFG["counts"])}


def fd(args):
    cfg = ei.Config(
        n_initial=args.n0, cells=40, dt=0.025, steps=20, retained=8, source_samples=8, partners=1
    )
    _FD_CFG.update(cfg=cfg, counts=ei.nominal_counts(cfg, T4))
    rows = []
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(_fd_worker, range(args.seeds), chunksize=4):
            rows.append(row)
            if len(rows) % 50 == 0:
                f = np.array([r["fd"] for r in rows])
                print(
                    f"{len(rows)} seeds: fd {f.mean(0).round(4)} +- {(f.std(0, ddof=1) / np.sqrt(len(f))).round(4)}",
                    flush=True,
                )
                _dump(f"fd_N{args.n0 // 1000}k.json", {"n_initial": args.n0, "rows": rows})
    _dump(f"fd_N{args.n0 // 1000}k.json", {"n_initial": args.n0, "rows": rows})


def recon_bias(args):
    cfg = ei.Config(
        n_initial=args.n0,
        cells=40,
        dt=0.025,
        steps=20,
        retained=args.retained,
        source_samples=1995,
        partners=1,
        neighbors=32,
        q0=0.0,
    )
    exact, affine = [], []
    original = ei._reconstruct
    for seed in range(3):
        tr = ei.forward(cfg, T4, seed=seed, injection_counts=ei.nominal_counts(cfg, T4))
        ei._reconstruct = original
        exact.append(ei.backward(tr, seed=seed)[:, 0])
        ei._reconstruct = _forced_affine
        affine.append(ei.backward(tr, seed=seed)[:, 0])
    ei._reconstruct = original
    exact, affine = np.array(exact), np.array(affine)
    print(
        f"R={args.retained} N0={args.n0}: exact {exact.mean(0).round(4)}  affine {affine.mean(0).round(4)}  ratio {(affine.mean(0) / exact.mean(0)).round(3)}"
    )


def _forced_affine(targets, values, cfg, normalization, remaining, terminal_reward, scale):
    del normalization, remaining, terminal_reward
    return ei.LocalAffine(targets, values, cfg.neighbors, scale)


def _dump(name, payload):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("two-step")
    s.add_argument("--seeds", type=int, default=240)
    s.add_argument("--a-seeds", type=int, default=16)
    s = sub.add_parser("compare")
    s.add_argument("--n0", type=int, default=2000)
    s.add_argument("--retained", type=int, default=512)
    s.add_argument("--seeds", type=int, default=16)
    s = sub.add_parser("scale")
    s.add_argument("--n0", type=int, default=16000)
    s.add_argument("--seeds", type=int, default=32)
    s.add_argument("--seed-start", type=int, default=0)
    s = sub.add_parser("fd")
    s.add_argument("--n0", type=int, default=16000)
    s.add_argument("--seeds", type=int, default=1600)
    s.add_argument("--workers", type=int, default=10)
    s = sub.add_parser("recon-bias")
    s.add_argument("--n0", type=int, default=16000)
    s.add_argument("--retained", type=int, default=512)
    args = p.parse_args(argv)
    {"two-step": two_step, "compare": compare, "scale": scale, "fd": fd, "recon-bias": recon_bias}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
