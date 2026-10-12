"""The split per-cell backward fit on a real, complicated cut-cell geometry: an optc level-set
iterate (circle_coarse_S{S}_{opt}/coef_<k>.npy) in the transitional regime (Kn 0.5, so
the backward's collision term and its fits matter) with ONE-SIDED backward fits
(``backward_normals``).  Per seed and backward setting the coefficient gradient dJ/dcoef
(examples/mesh/optimize.seed_terms' sum), paired across settings (same forward), and CRN
central FDs of J along fixed coefficient directions ``d`` against ``g . d``: ``shift``
(``d = 1``, the surface moved along its normal everywhere: per seed a sum of ~5000 noisy
coefficients, so poor signal-to-noise) and ``mode2`` (the ``cos 2 theta`` mode about the
case centre, ensemble_mesh's ``mode2`` control: a smooth, localised direction).

    PYTHONPATH=. python3 scratch/cellfit/geom_gate.py OUT.jsonl RUN_DIR ITER SEED0 SEED1 '{..}'
overrides: settings [{"basis": .., "k": .., ["sided": bool]}] (sided: one-sided backward fits,
default the config's backward_normals), fd_h [h, ..] ([] = none), directions [shift, mode2],
virtual [S, K] (default [4, 8])
and cfg keys (knudsen 0: free-molecular, the regime the optc runs optimised in).
"""

import dataclasses
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward  # noqa: E402
from jax_dsmc.mesh import levelset as L  # noqa: E402

DEFAULT_SETTINGS = [{"basis": "quadratic_full", "k": 32}]


def config(over):
    cfg = OPT.default_config()
    cfg.update(fnum_scale=4, grid_scale=1, levelset_h=0.0625, levelset_samples=16,
               grazing=0.05, knudsen=0.5, a_max=64, backward_normals=True)
    cfg.update(over)
    return cfg


def gradient(cfg, ls, key, setting, virtual):
    ensemble_backward.configure("virtual", *virtual, "staged", setting["basis"])
    cfg = dict(cfg, backward_normals=setting.get("sided", cfg["backward_normals"]))
    t0 = time.time()
    all_terms, info, s = EM.shape_terms(
        cfg, key, ls, "radius", setting["k"], cfg["timing_band"], (cfg["variant"],)
    )
    t = all_terms[cfg["variant"]]
    g = t.pathwise + t.score + t.silhouette + t.timing
    if t.order is not None:
        g = g + t.order
    if s["volume"] is not None:
        g = g + s["volume"]
    return np.asarray(g), info, time.time() - t0


def direction(name, cfg, ls):
    """Coefficient direction ``d`` (``f64`` of the coefficient shape)."""
    if name == "shift":
        return jnp.ones_like(ls.coef)
    if name == "mode2":
        centre = jnp.asarray(EM.F.CASES[cfg["case"]]["centre"], dtype=jnp.float64)
        mode = EM._mode_shape(L.knots(ls), centre)
        return L.interpolate(mode.reshape(ls.coef.shape))
    raise ValueError(name)


def objective(cfg, ls, key):
    _, _, diag, _, tallies, s = EM.forward(cfg, "radius", EM.nominal(cfg, "radius"), key,
                                           False, ls)
    EM._check(diag)
    scale_r = float(s["params"].fnum[0]) / (cfg["window"] * float(s["params"].dt))
    return EM.objective_value(cfg, tallies, s["static"].D, scale_r)


def main():
    path, run, it = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed0, seed1 = int(sys.argv[4]), int(sys.argv[5])
    over = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
    settings = over.pop("settings", DEFAULT_SETTINGS)
    hs = over.pop("fd_h", [0.05])
    names = over.pop("directions", ["shift", "mode2"])
    virtual = over.pop("virtual", [4, 8])
    cfg = config(over)
    _grid, ls = OPT.initial_levelset(cfg)
    ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{run}/coef_{it}.npy")))
    dirs = {n: direction(n, cfg, ls) for n in names}
    for i in range(seed0, seed1):
        key = jax.random.key(cfg["seed_start"] + i)
        row = {"run": run, "iterate": it, "seed": i, "config": over, "runs": {}}
        for st in settings:
            g, info, secs = gradient(cfg, ls, key, st, virtual)
            tag = f"{st['basis']}_k{st['k']}" + ("" if st.get("sided", True) else "_twosided")
            row["runs"][tag] = {
                "gradient": g.tolist(),
                "directional": {n: float(jnp.sum(g * d)) for n, d in dirs.items()},
                "objective": info["objective"], "seconds": secs,
                "seconds_backward": info["seconds_backward"],
                "lambda_rms": info["lambda_rms"],
            }  # fmt: skip
            proj = " ".join(f"g.{n} {float(jnp.sum(g * d)):+.4e}" for n, d in dirs.items())
            print(f"seed {i} {tag:44s} {proj} lambda_rms {info['lambda_rms']} "
                  f"|g| {np.linalg.norm(g):.4e} ({secs:.0f} s, backward "
                  f"{info['seconds_backward']:.0f} s)", flush=True)
        row["fd"], row["fd_h"] = {}, hs
        for n, d in dirs.items():
            fd = []
            for h in hs:
                up = objective(cfg, dataclasses.replace(ls, coef=ls.coef + h * d), key)
                dn = objective(cfg, dataclasses.replace(ls, coef=ls.coef - h * d), key)
                fd.append((up - dn) / (2 * h))
            row["fd"][n] = fd
            print(f"seed {i} FD {n} {fd}", flush=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
