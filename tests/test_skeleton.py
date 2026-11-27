"""The spread-skeleton length-scale penalty (jax_dsmc/mesh/skeleton.py, Hermann & Noel 2026)."""

import os
import sys

import jax.numpy as jnp
import numpy as np

from jax_dsmc.mesh import levelset as L
from jax_dsmc.mesh import skeleton as K

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "mesh"))
import optimize as O

LO, HI, H, R_S = np.array([-2.0, -2.0]), np.array([2.0, 2.0]), 0.1, 0.2


def _slab(t):
    origin, shape = L.grid_for_box(LO, HI, H)
    template = L.LevelSet(coef=jnp.zeros(shape), lo=jnp.asarray(origin), h=H)
    sdf = jnp.abs(L.knots(template)[:, 1]) - 0.5 * t
    return L.from_values(sdf.reshape(shape), origin, H)


def test_slab_penalty_falls_to_zero_at_the_length_scale():
    """Surface mean of S_c (threshold 1): clearly positive at t = r_s, ~0 from t = 2 r_s (the
    paper's Fig. 4), 0 for a thick disc."""
    s = K.SkeletonSettings(r_s=R_S)
    p = [K.penalty_and_gradient(_slab(m * R_S), LO, HI, 1.0, s)[0] for m in (1.0, 2.0, 4.0)]
    assert p[0] > 0.3
    assert p[1] < 0.05
    assert p[2] < 1e-3
    origin, shape = L.grid_for_box(LO, HI, H)
    disc = L.from_analytic(np.zeros(2), 1.0, origin, H, shape)
    assert K.penalty_and_gradient(disc, LO, HI, 1.0, s)[0] < 1e-3


def test_gradient_matches_central_differences():
    """``dP/dcoef`` along a random smooth direction against central differences of P (the
    voxel block is fixed by the unperturbed body)."""
    s = K.SkeletonSettings(r_s=R_S)
    ls = _slab(1.5 * R_S)
    origin, counts, hv = K.voxel_block(ls, LO, HI, s)
    x = jnp.asarray(K._centres(origin, counts, hv))
    _p, grad = K.penalty_and_gradient(ls, LO, HI, 1.0, s)
    rng = np.random.default_rng(0)
    knots = np.asarray(L.knots(ls)).reshape(*ls.coef.shape, 2)
    direction = jnp.asarray(np.cos(2.1 * knots[..., 0] + rng.uniform(0, 6))
                            * np.exp(-knots[..., 1] ** 2))  # fmt: skip
    eps = 1e-4
    plus = K.penalty(ls.coef + eps * direction, ls, x, tuple(counts), hv, 1.0, s)
    minus = K.penalty(ls.coef - eps * direction, ls, x, tuple(counts), hv, 1.0, s)
    fd = float(plus - minus) / (2 * eps)
    ad = float(jnp.sum(grad * direction))
    assert abs(ad - fd) < 1e-3 * abs(fd), (ad, fd)


def test_ramp_is_the_papers_continuation():
    assert K.ramp(0, 5, 25) == 0.0
    assert K.ramp(15, 5, 25) == 0.5
    assert K.ramp(30, 5, 25) == 1.0


def test_optimizer_dispatches_on_thickness_method():
    """``optimize.thinness_and_gradient`` gives the skeleton penalty for "skeleton" (``r_s``
    half of thickness_min, threshold ramped) and the probe share for "probe"."""
    ls = _slab(1.0 * R_S)
    cfg = dict(O.default_config(), thickness_min=2 * R_S, thickness_method="skeleton")
    t_skel, g = O.thinness_and_gradient(cfg, ls, LO, HI, 0)
    ref = K.penalty_and_gradient(ls, LO, HI, 1.0, K.SkeletonSettings(r_s=R_S))[0]
    assert abs(t_skel - ref) < 1e-12 and g.shape == ls.coef.shape
    cfg.update(thickness_ramp_start=0, thickness_ramp_end=10)
    assert O.thinness_and_gradient(cfg, ls, LO, HI, 0)[0] < 0.1 * t_skel  # threshold 0
    cfg["thickness_method"] = "probe"
    t_probe = O.thinness_and_gradient(cfg, ls, LO, HI, 0)[0]
    # the band version (optimize's) against the full sub-grid integral: band truncation only
    assert abs(t_probe / float(L.body_thinness(ls, LO, HI, 2 * R_S, 4)) - 1.0) < 1e-5
