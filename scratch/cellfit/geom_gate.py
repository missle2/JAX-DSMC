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
overrides: settings [{"basis": .., "k": .., ["sided": bool], ["grazing": eps], ["horizon": M]}]
(sided: one-sided backward fits, default the config's backward_normals; grazing: the hit
derivative's cut-off -- derivative only, the forward is unchanged, so paired; horizon: the
same forward captured for its last M of warmup + steps steps), fd_h [h, ..] ([] = none),
directions [shift, mode2], virtual [S, K] (default [4, 8]), smooth_shape sigma (knots; 0 =
off: the iterate's coefficients Gaussian-filtered, reinitialised and shifted back to its
area -- the sub-cell features removed, FD and adjoint both on the smoothed body) and cfg
keys (knudsen 0: free-molecular, the regime the optc runs optimised in).  Each run also
records every term's projection on the directions (``terms``).

Since 2026-10-17 (scratch/thingate): direction ``nose`` (a Gaussian bump of sigma
:data:`NOSE_SIGMA` on the coefficients at the body's most upstream point: the surface
there moved alone), ``maps`` DIR (every (seed, setting)'s per-coefficient terms to
``DIR/it<k>_seed<i>_<setting>.npz``, float32) and the carried value's rms / max / 99.99th
percentile per run (``lambda_*``).  The optimiser's own iterate is reproduced exactly by its
config (``case``, ``boundary``, ``mach``, ...), ``virtual``, the shipped basis and seeds
``0 .. S-1`` (keys ``seed_start + i``, the optimiser's common random numbers).
"""

import dataclasses
import json
import os
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
NOSE_SIGMA = 0.5  # m: one DSMC cell


def config(over):
    cfg = OPT.default_config()
    cfg.update(fnum_scale=4, grid_scale=1, levelset_h=0.0625, levelset_samples=16,
               grazing=0.05, knudsen=0.5, a_max=64, backward_normals=True)
    cfg.update(over)
    return cfg


TERMS = ("pathwise_tally", "pathwise_x", "pathwise_v", "score", "silhouette", "timing", "order")


def setting_cfg(cfg, setting):
    cfg = dict(cfg, backward_normals=setting.get("sided", cfg["backward_normals"]))
    if "grazing" in setting:
        cfg["grazing"] = float(setting["grazing"])
    if "horizon" in setting:
        total = cfg["warmup"] + cfg["steps"]
        cfg.update(warmup=total - int(setting["horizon"]), steps=int(setting["horizon"]))
    return cfg


def setting_tag(st):
    tag = f"{st['basis']}_k{st['k']}" + ("" if st.get("sided", True) else "_twosided")
    if "grazing" in st:
        tag += f"_gz{st['grazing']:g}"
    if "horizon" in st:
        tag += f"_h{st['horizon']}"
    return tag


def smoothed(ls, cfg, sigma):
    """The iterate without its sub-``sigma`` features, at its own area."""
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    grid, _ = OPT.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    area = float(L.body_volume(ls, lo, hi, 4))
    coef = jnp.asarray(gaussian_filter(np.asarray(ls.coef), sigma, mode="nearest"))
    out, _ = OPT.project_area(L.reinitialise(dataclasses.replace(ls, coef=coef)), area, lo, hi)
    return out


def gradient(cfg, ls, key, setting, virtual, dirs):
    ensemble_backward.configure("virtual", *virtual, "staged", setting["basis"])
    cfg = setting_cfg(cfg, setting)
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
    parts = {n: getattr(t, n) for n in TERMS if getattr(t, n, None) is not None}
    if s["volume"] is not None:
        parts["volume"] = s["volume"]
    terms = {n: {d: float(jnp.sum(v * dv)) for d, dv in dirs.items()} for n, v in parts.items()}
    return np.asarray(g), info, time.time() - t0, terms, parts


def direction(name, cfg, ls):
    """Coefficient direction ``d`` (``f64`` of the coefficient shape)."""
    if name == "shift":
        return jnp.ones_like(ls.coef)
    if name == "mode2":
        centre = jnp.asarray(EM.F.CASES[cfg["case"]]["centre"], dtype=jnp.float64)
        mode = EM._mode_shape(L.knots(ls), centre)
        return L.interpolate(mode.reshape(ls.coef.shape))
    if name == "nose":
        x = L.knots(ls)
        phi = L.value(ls, x)
        inside = np.asarray(phi) < 0.0
        xs = np.asarray(x)
        tip = xs[inside][np.argmin(xs[inside, 0])]
        bump = jnp.exp(-jnp.sum((x - jnp.asarray(tip)) ** 2, axis=1) / (2 * NOSE_SIGMA**2))
        return L.interpolate(bump.reshape(ls.coef.shape))
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
    sigma = over.pop("smooth_shape", 0)
    maps = over.pop("maps", None)
    if maps:
        os.makedirs(maps, exist_ok=True)
    cfg = config(over)
    _grid, ls = OPT.initial_levelset(cfg)
    ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{run}/coef_{it}.npy")))
    if sigma:
        ls = smoothed(ls, cfg, sigma)
    dirs = {n: direction(n, cfg, ls) for n in names}
    for i in range(seed0, seed1):
        key = jax.random.key(cfg["seed_start"] + i)
        row = {"run": run, "iterate": it, "seed": i, "config": over, "smooth_shape": sigma,
               "runs": {}}
        for st in settings:
            g, info, secs, terms, parts = gradient(cfg, ls, key, st, virtual, dirs)
            tag = setting_tag(st)
            if maps:
                arrays = {n: np.asarray(v, np.float32) for n, v in parts.items()}
                np.savez_compressed(f"{maps}/it{it}_seed{i}_{tag.replace(':', '-')}.npz",
                                    gradient=g.astype(np.float32), **arrays)  # fmt: skip
            row["runs"][tag] = {
                "terms": terms,
                "gradient": [] if maps else g.tolist(),  # maps: in the npz
                "directional": {n: float(jnp.sum(g * d)) for n, d in dirs.items()},
                "objective": info["objective"], "seconds": secs,
                "seconds_backward": info["seconds_backward"],
                "lambda_rms": info["lambda_rms"],
                "lambda_absmax": info.get("lambda_absmax"),
                "lambda_p9999": info.get("lambda_p9999"),
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
