"""Data of the paper's 2-D shape figures as CSV for pgfplots: the body's zero contour at chosen
iterates of a run (examples/mesh/results/optimize/<tag>/coef_<k>.npy), segments separated by a
``nan, nan`` row (``unbounded coords=jump``), and the run's history (objective, its running
mean, area):

* ``shape_<name>_<k>.csv`` (``x, y``) for each iterate ``k`` (``last``: the last saved one);
* ``run_<name>.csv`` (``iterate, value, mean, area``): value is ``L / D`` for a lift-to-drag
  run and ``J / J_0`` otherwise, ``J_0`` the reference run's first objective where one is
  named (the airfoils: the NACA 0012's drag at the same Knudsen number, paper_fwd).

    PYTHONPATH=. python3 examples/paper/figures/export_shapes.py --out docs/paper/figures/data
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from examples.mesh import optimize as O
from examples.paper.figures.export_histories import running

ROOT = Path(O.RESULTS)
# name -> (tag, iterates[, reference tag])
RUNS = {
    # 261: the end of the best 25-iterate mean (3.37; pieces broke off after ~300)
    "ld2d": ("circle_long_kn0.5_M10_S1_ox_adam_ldhz0_floor_sm8_free_g32_w500", (0, 100, 200, 261)),
    "naca": ("paper_fwd_naca0012_kn0.5", (0,)),
    **{f"af{kn}": (f"paper_airfoil_kn{kn}", (0,), f"paper_fwd_naca0012_kn{kn}")
       for kn in ("0.2", "0.5", "1.0")},
}  # fmt: skip


def _last(out):
    k = max(int(p.stem.split("_")[1]) for p in out.glob("coef_*.npy"))
    return k


def _write_polyline(path, segments):
    rows = []
    for seg in segments:
        rows += [*seg.tolist(), [np.nan, np.nan]]
    np.savetxt(path, np.array(rows), delimiter=",", header="x,y", comments="", fmt="%.5g")


def export(name, tag, iterates, out, ref=None):
    run = ROOT / tag
    hist = json.loads((run / "history.json").read_text())
    cfg = dict(O.default_config(), **hist["config"])
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    last = _last(run)
    for k in (*iterates, "last"):
        kk = last if k == "last" else min(int(k), last)
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(run / f"coef_{kk}.npy")))
        segs = [s for s in O.surface_polyline(ls, lo, hi) if len(s)]
        _write_polyline(out / f"shape_{name}_{k}.csv", segs)
        print(f"{name} iterate {kk}: {len(segs)} segment(s)")
    rows = hist["rows"]
    j = np.array([r["objective"] for r in rows], float)
    n = len(j) if np.isfinite(j).all() else int(np.argmin(np.isfinite(j)))
    j = j[:n]
    lift = cfg["objective"] == "lift_drag"
    j0 = j[0]
    if ref:
        j0 = json.loads((ROOT / ref / "history.json").read_text())["rows"][0]["objective"]
    value = -j if lift else j / j0
    area = np.array([r["area"] for r in rows[:n]])
    np.savetxt(
        out / f"run_{name}.csv",
        np.c_[np.arange(n), value, running(value), area],
        delimiter=",",
        header="iterate,value,mean,area",
        comments="",
        fmt="%.6g",
    )
    print(f"{name}: {n} iterates, last mean {running(value)[-1]:.4g}, saved coef {last}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=list(RUNS))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        tag, iterates, *ref = RUNS[name]
        if (ROOT / tag / "history.json").exists():
            export(name, tag, iterates, args.out, *ref)
        else:
            print(f"missing {tag}")


if __name__ == "__main__":
    main()
