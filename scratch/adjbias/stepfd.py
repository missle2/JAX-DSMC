"""Is the descent the free M 10 runs keep predicting real (2026-10-25)?  In their slide phase
(ox_adam_fix iterates 150-250) the adjoint predicts ``g . step ~ -0.01`` per iterate (~4 % of
J, the same with independent seeds) and the realised change is ~0, with the predicted decrease
at the needle nose in ``pathwise_tally`` (scratch/adjbias/drift_pipeline.py).  Here, at
iterate K of a run, along the step the run actually took (``delta = coef_{K+1} - coef_K``)
and its parts:

* ``full``: the step;
* ``nose``: the step on the knots within ``NOSE`` m of the tip (tapered to 0 over ``TAPER`` m);
* ``rest``: ``full - nose``;
* ``shift``: a rigid ``SHIFT`` m downstream translation (vec_analysis' surface-fitted one);

the adjoint's ``g . delta`` per seed against the paired central difference
``(J(c + eps delta) - J(c - eps delta)) / (2 eps)`` (same keys, forward only) at
``eps = 0.5, 1`` (``shift`` at 1).  FD ~ adjoint at both eps: the gradient is right and the
runs' unrealised decrease is the step's curvature (overshoot); FD ~ 0: the gradient is wrong
along that part.  ``--refine 2`` repeats it on the body resampled onto knots ``h / 2`` (the
same shapes: ``coef``, ``coef +- eps delta`` resampled; samples doubled, same hit reach):
does a nose error shrink with the representation's resolution?

    PYTHONPATH=. python3 scratch/adjbias/stepfd.py NAME --run TAG --iterate K [--seeds 8]
        [--refine 1|2] -> scratch/adjbias/cluster/stepfd_NAME.json (cluster group stepfd)
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

sys.path.insert(0, "scratch/adjbias")
import drift_pipeline as DP  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward  # noqa: E402

NOSE = 1.0
TAPER = 0.5
SHIFT = 0.04
EPS = {"full": (0.5, 1.0), "nose": (0.5, 1.0), "rest": (0.5, 1.0), "shift": (1.0,)}


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--run", required=True)
    p.add_argument("--iterate", type=int, required=True)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--refine", type=int, default=1)
    return p.parse_args(argv)


def directions(ls, coef0, coef1, lo, hi):
    """``{name: coefficient direction}`` on the run's knots."""
    delta = coef1 - coef0
    kx = np.asarray(O.L.knots(ls)).reshape(*coef0.shape, 2)
    tip = float(kx[..., 0][coef0 < 0].min())
    u = np.clip((kx[..., 0] - tip - NOSE) / TAPER, 0.0, 1.0)
    w = 0.5 * (1.0 + np.cos(np.pi * u))  # 1 up to tip + NOSE, 0 beyond tip + NOSE + TAPER
    t, _quad = DP.fitted_translation(ls, kx[coef0 < 0].min(0) - 0.5, kx[coef0 < 0].max(0) + 0.5)
    return {"full": delta, "nose": delta * w, "rest": delta * (1 - w), "shift": SHIFT * t}, tip


def resample(coef, src, dst):
    """The spline ``coef`` on ``src``'s knots, as coefficients on ``dst``'s (linear)."""
    s = dataclasses.replace(src, coef=jnp.asarray(coef))
    return np.asarray(O.L.interpolate(O.L.value(s, O.L.knots(dst)).reshape(dst.coef.shape)))


def forward_objective(cfg, ls, key):
    value0 = EM.nominal(cfg, "radius")
    _s0, _f, diag, _rec, tallies, s = EM.forward(cfg, "radius", value0, key, False, ls)
    EM._check(diag)
    scale_r = float(s["params"].fnum[0]) / (cfg["window"] * float(s["params"].dt))
    return EM.objective_value(cfg, tallies, s["static"].D, scale_r)


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    run_dir = Path(O.RESULTS) / args.run
    cfg = dict(O.default_config(), **json.load(open(run_dir / "history.json"))["config"])
    cfg["restart_steps"] = 0  # every forward from the empty box: the pairs share keys only
    ensemble_backward.configure("virtual", 4, 8, "staged")
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    coef0 = np.load(run_dir / f"coef_{args.iterate}.npy")
    coef1 = np.load(run_dir / f"coef_{args.iterate + 1}.npy")
    ls = dataclasses.replace(ls0, coef=jnp.asarray(coef0))
    dirs, tip = directions(ls, coef0, coef1, lo, hi)
    _a, g_area = O.area_and_gradient(ls, lo, hi)
    area_change = {k: float(np.sum(g_area * v)) for k, v in dirs.items()}
    if args.refine > 1:
        fine_cfg = dict(cfg, levelset_h=cfg["levelset_h"] / args.refine,
                        levelset_samples=cfg["levelset_samples"] * args.refine)  # fmt: skip
        _g, fine0 = O.initial_levelset(fine_cfg)
        base = resample(coef0, ls, fine0)
        # each shape resampled, so the finer run sees the same bodies; the map is linear
        dirs = {k: resample(coef0 + v, ls, fine0) - base for k, v in dirs.items()}
        cfg, ls = fine_cfg, dataclasses.replace(fine0, coef=jnp.asarray(base))
    out = {"run": args.run, "iterate": args.iterate, "refine": args.refine, "h": ls.h,
           "tip_x": tip, "nose": NOSE, "taper": TAPER, "shift": SHIFT, "eps": EPS,
           "area_change": area_change,
           "history_predicted": None, "seeds": []}  # fmt: skip
    rows = json.load(open(run_dir / "history.json"))["rows"]
    k = args.iterate
    out["history_predicted"] = rows[k]["predicted_decrease"]
    out["history_realised"] = rows[k + 1]["objective"] - rows[k]["objective"]
    path = Path(f"scratch/adjbias/cluster/stepfd_{args.name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(args.seeds):
        t0 = time.time()
        key = jax.random.key(cfg["seed_start"] + i)
        st = O.seed_terms(cfg, ls, [i])
        g, j0 = st["gradient"][0], float(st["objective"][0])
        row = {"seed": i, "J": j0, "adjoint": {}, "fd": {}}
        for name, v in dirs.items():
            row["adjoint"][name] = float(np.sum(g * v))
            for eps in EPS[name]:
                plus = dataclasses.replace(ls, coef=ls.coef + eps * jnp.asarray(v))
                minus = dataclasses.replace(ls, coef=ls.coef - eps * jnp.asarray(v))
                jp, jm = forward_objective(cfg, plus, key), forward_objective(cfg, minus, key)
                row["fd"][f"{name}_{eps}"] = {"plus": jp, "minus": jm,
                                              "slope": (jp - jm) / (2 * eps)}  # fmt: skip
        row["seconds"] = time.time() - t0
        out["seeds"].append(row)
        path.write_text(json.dumps(out, indent=1))
        print(
            i,
            {n: round(row["adjoint"][n], 5) for n in dirs},
            {n: round(f["slope"], 5) for n, f in row["fd"].items()},
            flush=True,
        )
    summary = {}
    for name in dirs:
        a = np.array([r["adjoint"][name] for r in out["seeds"]])
        summary[f"adjoint_{name}"] = [float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))]
        for eps in EPS[name]:
            f = np.array([r["fd"][f"{name}_{eps}"]["slope"] for r in out["seeds"]])
            d = a - f
            summary[f"fd_{name}_{eps}"] = [float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))]
            summary[f"diff_{name}_{eps}"] = [
                float(d.mean()),
                float(d.std(ddof=1) / np.sqrt(len(d))),
            ]
    out["summary"] = summary
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
