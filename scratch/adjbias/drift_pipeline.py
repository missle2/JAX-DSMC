"""How much of the free M 10 runs' centroid slide does the gradient's translation component
drive (2026-10-25)?  The adjm10 ``body_vec`` gradients (8 seeds, the free M 10 Adam body at
iterate 100) through the shipped step pipeline (optax_rules.ox_adam: smooth, per-coefficient
rule, smooth, area tangent, rescale to ``step``), once as measured and once with the rigid
x-translation component removed (``g - (g.t / t.t) t``, ``t`` the surface-fitted translation
of vec_analysis: the projection a translation-invariant, i.e. correct, gradient has ~0 of).
Rules: steepest descent (the per-coefficient rule the identity), Adam's first step (sign) and
a stationary Adam (``m``, ``v`` the seed mean of the smoothed gradient and its square).  Also,
from the ox_adam_fix run's final optax state, how far Adam's direction is from momentum
steepest descent.

    PYTHONPATH=. python3 scratch/adjbias/drift_pipeline.py
"""

import dataclasses
import json
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

sys.path.insert(0, "scratch/adjbias")
import vec_analysis as V  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402

NAME = "adjm10_body_vec"
RUN = "circle_long_kn0.5_M10_S1_ox_adam_fix_tan_sm8_free_g32_w500"


def fitted_translation(ls, lo, hi):
    """Coefficients whose surface normal velocity is ``n_x`` (least squares, damp 1e-3), and
    the surface quadrature."""
    xs, ws, nrm, gn = V.surface_quadrature(ls, lo, hi, ls.h / 8)
    b = V.basis_matrix(ls, xs)
    a = -(sp.diags(np.sqrt(ws) / gn) @ b)
    t = spl.lsqr(a, np.sqrt(ws) * nrm[:, 0], damp=1e-3, atol=1e-12, btol=1e-12, iter_lim=20000)
    return t[0].reshape(ls.coef.shape), (xs, ws, nrm, gn, b)


def finish(u, ls, g_area, cfg):
    """The pipeline after the per-coefficient rule: smooth, area tangent, rescale."""
    u = O.smoothed_direction(u, ls, cfg["smooth"], cfg["band"] + ls.h)
    u = O.tangent_component(u, g_area)[0]
    return -cfg["step"] * u / np.abs(u).max()


def smooth(g, ls, cfg):
    return O.smoothed_direction(g, ls, cfg["smooth"], cfg["band"] + ls.h)


def rigid_x(d, quad):
    """Least-squares rigid x-translation (m) of the step's surface motion."""
    xs, ws, nrm, gn, b = quad
    vn = -(b @ d.ravel()) / gn
    return float(np.sum(ws * vn * nrm[:, 0]) / np.sum(ws * nrm[:, 0] ** 2))


def cos(a, b):
    return float(np.sum(a * b) / np.linalg.norm(a) / np.linalg.norm(b))


def report(label, d, dcx, quad):
    print(
        f"  {label:34s} centroid dx {np.sum(dcx * d):+.4f} m   rigid-fit dx {rigid_x(d, quad):+.4f} m"
    )


def main():
    d = np.load(f"scratch/adjbias/cluster/{NAME}_vectors.npz")
    cfg = dict(
        EM.default_config(), **json.load(open(f"scratch/adjbias/cluster/{NAME}.json"))["config"]
    )
    run_cfg = json.load(open(f"{O.RESULTS}/{RUN}/history.json"))["config"]
    cfg.update(smooth=run_cfg["smooth"], band=run_cfg["band"], step=run_cfg["step"])
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    ls = dataclasses.replace(ls0, coef=jnp.asarray(d["coef"]))
    seeds = sorted(int(k[2:]) for k in d.files if k.startswith("J_"))
    _a, g_area = O.area_and_gradient(ls, lo, hi)
    _c, jac = O.centroid_and_gradient(ls, lo, hi)
    dcx = jac[0]
    coef = np.asarray(ls.coef)
    kx = np.asarray(O.L.knots(ls)).reshape(*coef.shape, 2)
    t, quad = fitted_translation(ls, kx[coef < 0].min(0) - 0.5, kx[coef < 0].max(0) + 0.5)
    print(f"fitted translation: centroid dx per unit {np.sum(dcx * t):.3f} (1 = rigid)")
    gs, gfree, js = [], [], []
    for i in seeds:
        g = sum(d[f"{s}_{i}"] for s in V.TERMS)
        g = O.tangent_component(g, g_area)[0]  # what the optimiser receives
        gs.append(g)
        gfree.append(g - np.sum(g * t) / np.sum(t * t) * t)
        js.append(float(d[f"J_{i}"]))
    bias = np.array([100 * np.sum(g * t) / j for g, j in zip(gs, js, strict=True)])
    print(f"gradient translation number (% J/m): {bias.mean():.2f} ± "
          f"{bias.std(ddof=1) / np.sqrt(len(bias)):.2f}; removed per seed in the 'free' rows")  # fmt: skip
    print(f"|translation part| / |g| per seed: "
          f"{np.round([np.linalg.norm(g - f) / np.linalg.norm(g) for g, f in zip(gs, gfree, strict=True)], 3)}")  # fmt: skip

    print("single-seed steps (mean over the 8 seeds; the run uses 1 seed per iterate):")
    for rule in ("steepest", "adam_first"):
        for lab, gl in (("as measured", gs), ("translation removed", gfree)):
            steps = []
            for g in gl:
                u = smooth(g, ls, cfg)
                u = u if rule == "steepest" else np.sign(u)
                steps.append(finish(u, ls, g_area, cfg))
            cx = np.array([np.sum(dcx * s) for s in steps])
            rx = np.array([rigid_x(s, quad) for s in steps])
            print(f"  {rule:10s} {lab:20s} centroid dx {cx.mean():+.4f}±{cx.std(ddof=1) / np.sqrt(8):.4f} m"
                  f"   rigid-fit dx {rx.mean():+.4f}±{rx.std(ddof=1) / np.sqrt(8):.4f} m")  # fmt: skip
    print("seed-averaged directions (a noise-averaged iterate):")
    for lab, gl in (("as measured", gs), ("translation removed", gfree)):
        sm = [smooth(g, ls, cfg) for g in gl]
        m = np.mean(sm, 0)
        v = np.mean([s * s for s in sm], 0)
        sd = finish(m, ls, g_area, cfg)
        ad = finish(m / np.sqrt(v + 1e-300), ls, g_area, cfg)
        report(f"steepest, {lab}", sd, dcx, quad)
        report(f"adam (m/sqrt v), {lab}", ad, dcx, quad)
        print(f"    cos(adam, steepest) {cos(ad, sd):.3f}")

    print(f"\n{RUN}: final optax state")
    h = json.load(open(f"{O.RESULTS}/{RUN}/history.json"))
    rows = h["rows"]
    cxs = np.array([r["centroid"][0] for r in rows])
    for k in (5, 20, 50, 100, 150, 200):
        print(f"  observed centroid dx/iterate near {k}: {(cxs[k + 5] - cxs[k - 5]) / 10:+.4f} m")
    z = np.load(f"{O.RESULTS}/{RUN}/optax_state.npz")
    mu, nu = z["leaf1"], z["leaf2"]
    last = len(rows) - 1
    lsk = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{O.RESULTS}/{RUN}/coef_{last}.npy")))
    _a, ga = O.area_and_gradient(lsk, lo, hi)
    band = np.asarray(O.L.coefficient_band(lsk, cfg["band"] + lsk.h))
    r = np.sqrt(nu[band])
    print(f"  sqrt(nu) over the band: percentiles 10/50/90/99 / median = "
          f"{np.round(np.percentile(r, [10, 50, 90, 99]) / np.median(r), 2)}")  # fmt: skip
    ad = finish(mu / np.sqrt(nu + 1e-300), lsk, ga, cfg)
    sd = finish(mu, lsk, ga, cfg)
    print(f"  cos(adam step, momentum steepest-descent step) at iterate {last}: {cos(ad, sd):.3f}")


if __name__ == "__main__":
    main()
