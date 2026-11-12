"""The correction features (:data:`jax_dsmc.collide.relax.FEATURES`) over coarse cells of the
round-1 cases, as the operator sees them: thinned reference snapshots on the coarse grid.

* ``phi_scale`` -- each feature's RMS over the cells of the four training cases: the fixed
  standardisation of the linear and MLP corrections (an L1 penalty then weighs every term by
  its observed spread, not by a nominal guess).
* the features' correlation matrix over the training cells (uncentred, on the standardised
  features -- the Gram matrix a Lasso on them sees): near-collinear terms are the ones whose
  selection order an L1 path cannot be trusted on.
* per case (training and test): quantiles of each feature -- the coverage the test and
  circle cells are measured against.

    PYTHONPATH=. python3 -m examples.learned_collision.features [--windows 32]

Writes ``examples/learned_collision/results/features_<tag>.json``.
"""

from __future__ import annotations

import argparse
import functools
import json

import jax
import numpy as np

from examples.learned_collision import flows, operators
from examples.learned_collision import windows2d as w2
from examples.learned_collision.evaluate import load_ref
from examples.learned_collision.shear_wave import RESULTS
from jax_dsmc import sort
from jax_dsmc.collide import relax

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _cell_features(state, case, periodic):
    table = sort.build_table(state, case.grid)
    _, _, n, u, _, pi, theta_c = relax.local_state(state, table, case.grid, periodic)
    _, n_s, _, theta_s = relax.safe_cells(n, pi, theta_c)
    phi = relax.cell_features(n, n_s, u, theta_s, case.grid, case.sp, case.params, periodic)
    return phi, n


def case_features(name, ref, meta, windows, rng):
    """``f64[cells, 9]``: the features of the cells holding two or more particles, over
    ``windows`` thinned snapshots of case ``name`` (unscaled by any training spread)."""
    case = w2.make_case(name, meta)
    periodic = operators.PERIODIC[case.kind]
    feats = jax.jit(functools.partial(_cell_features, case=case, periodic=periodic))
    seeds, snaps, pools = ref["pool_x"].shape[:3]
    out = []
    for _ in range(windows):
        s, t, k = (int(rng.integers(m)) for m in (seeds, snaps, pools))
        state, _, over = w2.pool_state(ref["pool_x"][s, t, k], ref["pool_v"][s, t, k],
                                       ref["pool_alive"][s, t, k], case,
                                       jax.random.PRNGKey(int(rng.integers(2**31 - 1))))  # fmt: skip
        if int(over):
            raise RuntimeError(f"{name}: thinning overflow")
        phi, n = (np.asarray(a) for a in feats(state))
        out.append(phi[n >= 2.0])
    return np.concatenate(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--windows", type=int, default=32, help="snapshots per case")
    ap.add_argument("--refs-tag", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    refs, per_case, train = {}, {}, []
    for name, (ref_name, _, role) in flows.CASES.items():
        if ref_name not in refs:
            refs = {ref_name: load_ref(ref_name, args.refs_tag)}  # one reference in memory
        phi = case_features(name, *refs[ref_name], args.windows, rng)
        per_case[name] = {"role": role, "cells": int(phi.shape[0]),
                          "quantiles": np.quantile(phi, QUANTILES, axis=0).tolist(),
                          "rms": np.sqrt(np.mean(phi**2, axis=0)).tolist()}  # fmt: skip
        if role == "train":
            # equal weight per case whatever its cell count
            train.append(phi[rng.choice(len(phi), size=min(len(phi), 20_000), replace=False)])
        print(f"{name} ({role}): {phi.shape[0]} cells, median "
              f"{np.round(np.median(phi, axis=0), 3).tolist()}", flush=True)  # fmt: skip
    pooled = np.concatenate(train)
    scale = np.sqrt(np.mean(pooled**2, axis=0))
    z = pooled / scale
    gram = (z.T @ z) / len(z)
    out = {"features": list(relax.FEATURES), "phi_scale": scale.tolist(),
           "correlation": gram.tolist(), "quantiles_at": list(QUANTILES), "cases": per_case}  # fmt: skip
    (RESULTS / f"features_{args.tag}.json").write_text(json.dumps(out))
    print("phi_scale", np.round(scale, 4).tolist())
    print("correlation (uncentred, standardised):")
    for name, row in zip(relax.FEATURES, gram, strict=True):
        print(f"  {name:10s} " + " ".join(f"{v:5.2f}" for v in row))


if __name__ == "__main__":
    main()
