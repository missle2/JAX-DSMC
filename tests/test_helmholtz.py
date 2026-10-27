"""The Helmholtz filter of the filtered level-set formulation (jax_dsmc/mesh/helmholtz.py,
2026-10-26) and its use by the step rules (``optimize --filter helmholtz``)."""

import os
import sys

import numpy as np

from jax_dsmc.mesh import helmholtz as HZ

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "mesh"))


def _rough(a):
    """Summed second differences along x over the summed magnitude."""
    return np.abs(np.diff(a, 2, axis=0)).sum() / np.abs(a).sum()


def _small_filter(radius=0.4):
    h, lo, hi = 0.25, np.zeros(2), np.array([3.0, 2.0])
    origin = lo - 4 * h
    shape = (int(np.ceil(3 / h)) + 9, int(np.ceil(2 / h)) + 9)
    return HZ.HelmholtzFilter(shape, origin, h, lo, hi, radius), (shape, origin, h, lo, hi)


def test_filter_is_the_dense_helmholtz_solve_and_its_adjoint():
    """``F s`` is ``A^{-1} M s`` with the dense Kronecker Gram matrices (``A = (r / h)^2 K +
    M`` in knot units), ``F^T`` passes the dot-product test, constants pass unchanged."""
    f, (shape, origin, h, lo, hi) = _small_filter()
    grams = [HZ.gram_1d(m, (lo[a] - origin[a]) / h, (hi[a] - origin[a]) / h)
             for a, m in enumerate(shape)]  # fmt: skip
    (m1, k1), (m2, k2) = grams
    mass = np.kron(m1, m2)
    a = (0.4 / h) ** 2 * (np.kron(k1, m2) + np.kron(m1, k2)) + mass
    act = f.mask.ravel()
    rng = np.random.default_rng(0)
    s = rng.standard_normal(shape) * f.mask
    ref = np.zeros(act.size)
    ref[act] = np.linalg.solve(a[np.ix_(act, act)], mass[np.ix_(act, act)] @ s.ravel()[act])
    np.testing.assert_allclose(f.forward(s).ravel(), ref, atol=1e-10 * np.abs(ref).max())
    g = rng.standard_normal(shape) * f.mask
    lhs, rhs = np.sum(f.forward(s) * g), np.sum(s * f.adjoint(g))
    assert abs(lhs - rhs) < 1e-12 * abs(rhs)
    np.testing.assert_allclose(f.forward(f.mask * 1.0)[f.mask], 1.0, atol=1e-12)


def test_filter_kernel_has_the_helmholtz_second_moment():
    """In 1-D the filter of a unit coefficient spike sums to 1 with variance ``2 r^2`` (the
    kernel ``exp(-|x| / r) / 2r``), the equivalence ``optimize.filter_radius`` uses."""
    h, length, r = 0.05, 40.0, 0.5
    m = int(length / h) + 9
    f = HZ.HelmholtzFilter((m,), np.array([-4 * h]), h, np.zeros(1), np.array([length]), r)
    spike = np.zeros(m)
    spike[m // 2] = 1.0
    y = f.forward(spike)
    x = (np.arange(m) - m // 2) * h
    assert abs(y.sum() - 1.0) < 1e-9
    assert abs((y * x**2).sum() - 2 * r**2) < 1e-6


def test_helmholtz_rules_descend_and_keep_the_area(tmp_path):
    """``mma_s`` and ``ox_adam`` with ``filter helmholtz`` on the test problem of
    test_optimizers (the band coefficients pulled towards the circle moved 0.5 m): the
    objective falls, the area holds to the tangent projection's second order, and the step
    reaches beyond the band (the filter acts on the whole box)."""
    import dataclasses  # noqa: PLC0415

    import jax.numpy as jnp  # noqa: PLC0415
    import optax_rules as OX  # noqa: PLC0415
    import optimize as O  # noqa: PLC0415
    import optimizers as OZ  # noqa: PLC0415
    from test_optimizers import _circle_levelset  # noqa: PLC0415

    target = np.asarray(_circle_levelset((5.5, 5.0))[0].coef)
    for rule in ("mma_s", "ox_adam"):
        ls, lo, hi = _circle_levelset()
        cfg = {"optimizer": rule, "step": 0.05, "smooth": 2.0, "band": 1.0,
               "filter": "helmholtz", "filter_radius": 0.0}  # fmt: skip
        area0 = O.area_and_gradient(ls, lo, hi)[0]
        state, objectives, beyond = {"area0": area0}, [], []
        ctx = OX.Context()
        tx = OX.build("ox_adam", ctx, cfg["step"])
        ox_state = tx.init(jnp.asarray(ls.coef))
        for k in range(8):
            band = np.asarray(O.L.coefficient_band(ls, cfg["band"] + ls.h))
            diff = (np.asarray(ls.coef) - target) * band
            objectives.append(0.5 * float(np.sum(diff**2)))
            g_area = O.area_and_gradient(ls, lo, hi)[1]
            g, _ = O.tangent_component(diff, g_area)
            if rule == "mma_s":
                base, delta, _p, state, _x = O.rule_step(
                    cfg, ls, g, diff, g_area, np.array([objectives[-1]]), state, k, lo, hi
                )
                OZ.save_state(tmp_path / "s.npz", state)
                state = OZ.load_state(tmp_path / "s.npz")
            else:
                base = ls
                # as optimize.run: the constraint projected in the design space
                rows = O.constraint_rows(g_area, ls.coef.ndim)
                sm = O.smoothing(cfg, ls, lo, hi, rows)
                delta, ox_state = OX.step(tx, ox_state, ctx, g, ls, None, cfg, sm)
            beyond.append(np.abs(delta[~band]).max() / np.abs(delta).max())
            stepped = dataclasses.replace(base, coef=base.coef + jnp.asarray(delta))
            ls, _, _ = O.after_step(stepped, "tangent", area0, lo, hi, 0.0)
        assert objectives[-1] < 0.8 * objectives[0], (rule, objectives)
        assert abs(O.area_and_gradient(ls, lo, hi)[0] - area0) < 1e-6 * area0
        assert max(beyond) > 0.05, rule


def test_design_space_projection_keeps_the_constraints_with_smooth_corrections():
    """With constraint rows the Helmholtz smoothing projects in the design space: the step
    ``F P F^T g`` changes neither the area nor the centroid to first order, and the change the
    projection makes is smooth (the filtered rows, not the one-knot-wide raw ones)."""
    import optimize as O  # noqa: PLC0415
    from test_optimizers import _circle_levelset  # noqa: PLC0415

    ls, lo, hi = _circle_levelset()
    cfg = {"filter": "helmholtz", "filter_radius": 0.0, "smooth": 2.0, "band": 1.0,
           "fix_centroid": 1}  # fmt: skip
    g_area = O.area_and_gradient(ls, lo, hi)[1]
    rows = O.constraint_rows(O.constraint_gradients(ls, lo, hi, g_area, cfg), ls.coef.ndim)
    assert rows.shape[0] == 3
    rng = np.random.default_rng(1)
    band = np.asarray(O.L.coefficient_band(ls, 1.0 + ls.h))
    g = rng.normal(size=ls.coef.shape) * band + rows[1]  # with a translation part
    free = O.smoothing(cfg, ls, lo, hi)
    kept = O.smoothing(cfg, ls, lo, hi, rows)
    step = kept.out(kept.into(g))
    for r in rows:
        assert abs(np.sum(r * step)) < 1e-9 * np.linalg.norm(r) * np.linalg.norm(step)
    correction = step - free.out(free.into(g))

    # smoothness: the correction's second differences are small against the row's own
    assert _rough(correction) < 0.2 * _rough(rows[1])


def test_polygon_starts_are_sharp_bodies_of_the_requested_size():
    """``--initial diamond`` has the circle's area and its thickness ratio; ``plate`` its chord
    and thickness, turned leading edge up; both bodies are one piece."""
    import optimize as O  # noqa: PLC0415

    base = dict(O.default_config(), case="circle_long", levelset_h=0.0625)
    radius = 3.0
    for shape, extra in (("diamond", {"initial_thickness": 0.3}),
                         ("plate", {"initial_thickness": 0.125, "initial_aoa": 10.0})):  # fmt: skip
        cfg = dict(base, initial=shape, **extra)
        grid, ls = O.initial_levelset(cfg)
        lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
        area = float(O.L.body_volume(ls, lo, hi, 4))
        v = O.polygon_vertices(cfg)
        if shape == "diamond":
            assert abs(area - np.pi * radius**2) < 0.02 * np.pi * radius**2
            assert abs(np.ptp(v[:, 1]) / np.ptp(v[:, 0]) - 0.3) < 1e-9
        else:
            assert abs(area - 8.0 * 1.0) < 0.05 * 8.0
            lead, trail = v[[0, 3]].mean(axis=0), v[[1, 2]].mean(axis=0)
            assert lead[1] > trail[1]  # leading edge up
            assert abs(np.degrees(np.arctan2(lead[1] - trail[1], trail[0] - lead[0])) - 10) < 1e-9
