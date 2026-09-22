"""The pathwise (fixed-key ``jax.grad``) gradient on every ensemble-benchmark control.

    python3 -m examples.gradient_bench.pathwise            # every base ensemble artifact
    python3 -m examples.gradient_bench.pathwise --only relax_diameter

For each base ``results/ensemble_*.json`` artifact this differentiates the *same* forward at
the *same* seeds with plain reverse-mode AD of the objective -- every discrete decision
(cell membership, attempt count, pair choice, accept bit, accommodation branch, hit set)
frozen at its realised value, exactly the mode-A estimator of ``examples.shock_adjoint``
-- and writes ``results/pathwise_<name>.json`` with one row per seed.  The report's
``figure_pathwise_vs_ensemble`` joins it by seed with the ensemble artifact's gradient and
common-random-number FD, so the three estimators of one control are compared on identical
realisations.

What the comparison shows (review section 4, ``shock_adjoint/README``): a control that
enters through a reparameterised draw (a wall's speed or temperature, a reservoir
temperature) has a pathwise route and AD is exact for it; a control that enters only
through decisions (VSS diameter / omega through the accept probability, the inlet density
through the emitted count, the accommodation coefficient through its Bernoulli branch, a
wall position through the hit set) has ``dJ/dtheta = 0`` pathwise -- not noise, a
systematic answer of zero for a nonzero sensitivity -- and a control that enters both ways
(the inlet temperature: the Maxwellian draws AND the flux-weighted count) is biased by the
decision share.  The ensemble adjoint recovers all of them.  A decision-only control's
pathwise gradient is identically zero, so ``--seeds 8`` is enough to record it; the
reparameterised controls take seconds per hundred seeds at the benchmark sizes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import jax
import jax.numpy as jnp

from examples import ensemble_couette, ensemble_inlet_shock, ensemble_relaxation
from examples.gradient_bench import RESULTS
from examples.gradient_convergence import inlet_shock_geometry
from jax_dsmc.validation import require_valid

VARIANT = re.compile(r"_(virt\d+x\d+(?:st)?|lin|nocoll|exactlast)")
COUETTE_KEYS = ("particles", "cells", "steps", "attempts", "dt", "epsilon", "majorant")
RELAX_KEYS = ("particles", "steps", "attempts", "dt", "majorant")


def _couette_value(c, key, parameter, cfg):
    forward = ensemble_couette._compiled_forward(**cfg, capture=False)
    _, out, diag, _, _ = forward(ensemble_couette.controls(parameter, c), key)
    return jnp.mean(out.v[:, 1] ** 2), diag


def _relax_value(c, key, parameter, cfg):
    forward = ensemble_relaxation._compiled_forward(**cfg, capture=False)
    _, out, diag, _, _ = forward(ensemble_relaxation.controls(parameter, c), key)
    return jnp.mean(out.v[:, 0] ** 2), diag


def _inlet_value(c, key, parameter, cfg, geometry, objective):
    forward = ensemble_inlet_shock.compiled_forward(
        geometry, cfg["cells"], cfg["attempts"], cfg["steps"], cfg["majorant"], False
    )
    _, out, diag, _ = forward(ensemble_inlet_shock.controls(parameter, c), key)
    value = ensemble_inlet_shock.objective_value(
        objective, out.x, out.v, out.alive, geometry["length"], cfg["particles"]
    )
    return value, diag


def value_function(name, config):
    """``(f(c, key) -> (J, diag), family)`` of one ensemble artifact from its config."""
    parameter = config["parameter"]
    if name.startswith("ensemble_couette_"):
        cfg = {k: config[k] for k in COUETTE_KEYS}
        return lambda c, key: _couette_value(c, key, parameter, cfg), "couette"
    if name.startswith("ensemble_relax_"):
        cfg = {k: config[k] for k in RELAX_KEYS}
        return lambda c, key: _relax_value(c, key, parameter, cfg), "relax"
    cfg = {k: config[k] for k in ("particles", "cells", "steps", "attempts", "dt", "majorant")}
    geometry = inlet_shock_geometry(cfg["particles"], cfg["cells"], cfg["dt"])
    objective = config.get("objective", "energy-window")
    return lambda c, key: _inlet_value(c, key, parameter, cfg, geometry, objective), "inlet"


def base_artifacts():
    names = sorted(
        os.path.basename(p)[:-5] for p in glob.glob(os.path.join(RESULTS, "ensemble_*.json"))
    )
    return [n for n in names if not VARIANT.search(n) and not n.startswith("ensemble_mesh")]


def run(name, force=False, seeds=None):
    with open(os.path.join(RESULTS, f"{name}.json"), encoding="utf-8") as fh:
        art = json.load(fh)
    out = os.path.join(RESULTS, f"pathwise_{name[9:]}.json")
    if os.path.exists(out) and not force:
        with open(out, encoding="utf-8") as fh:
            prev = json.load(fh)
        if len(prev["rows"]) >= min(len(art["rows"]), seeds or len(art["rows"])):
            print(f"{name}: pathwise artifact complete ({len(prev['rows'])} seeds)")
            return
    fn, family = value_function(name, art["config"])
    grad = jax.jit(jax.value_and_grad(fn, has_aux=True))
    seed_start = art["config"]["seed_start"]
    rows = []
    t0 = time.time()
    for r in art["rows"][:seeds]:
        key = jax.random.key(seed_start + r["seed"])
        (value, diag), g = grad(jnp.float64(1.0), key)
        require_valid(diag)
        rows.append({"seed": r["seed"], "objective": float(value), "gradient": float(g)})
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "estimator": "pathwise: fixed-key jax.grad of the objective, decisions frozen",
                "ensemble_artifact": name,
                "family": family,
                "config": art["config"],
                "rows": rows,
                "elapsed": time.time() - t0,
            },
            fh,
            indent=1,
        )
        fh.write("\n")
    g = jnp.array([r["gradient"] for r in rows])
    print(
        f"{name}: {len(rows)} seeds, pathwise {float(g.mean()):.5g} ± "
        f"{float(g.std(ddof=1) / jnp.sqrt(len(rows))):.2g}  ({time.time() - t0:.0f} s) -> {out}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default=None, help="substring of the artifact names to run")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seeds", type=int, default=None, help="cap on the seeds (default: all)")
    args = p.parse_args(argv)
    for name in base_artifacts():
        if args.only and args.only not in name:
            continue
        run(name, args.force, args.seeds)


if __name__ == "__main__":
    main()
