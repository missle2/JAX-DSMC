"""The optimiser's step rules (examples/mesh/optimizers.py) on small numpy problems: the trust
region's judgement (paired per-seed differences, pooled rho, the floor's re-evaluation on new
seeds), L-BFGS on a quadratic, and one MMA subproblem under the area constraint."""

import dataclasses
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "mesh"))
import optimizers as OZ


def _trial(state, change, noise, step=0.05):
    """Judge a trial whose four seeds change by ``change`` +- ``noise`` against the base."""
    base = np.asarray(state["base_objectives"])
    return OZ.judge(state, base + change + noise * np.array([1.0, -1.0, 1.0, -1.0]), step)


BASE = {"radius": 0.05, "base_objectives": np.ones(4), "predicted": -0.1, "at_edge": True}


def test_trust_region_grows_on_agreement_shrinks_and_rejects_on_a_rise():
    """The 2026-10-20 judge: paired per-seed differences, a symmetric x2 / x0.5 update."""
    v = _trial(BASE, -0.09, 1e-4)
    assert v["accepted"] and abs(v["rho"] - 0.9) < 1e-9 and v["radius"] == 0.05 * OZ.GROW
    assert v["window"] == [[], [], []]  # the radius changed: pool again
    v = _trial(BASE, -0.01, 1e-4)
    assert v["accepted"] and v["radius"] == 0.05 * OZ.SHRINK
    v = _trial(BASE, 0.02, 1e-4)
    assert not v["accepted"] and not v["rebase"] and v["radius"] == 0.05 * OZ.SHRINK
    # rho centred on 0.5 keeps the radius (the earlier x1.5 / x0.5 drifted it down)
    v = _trial(BASE, -0.05, 1e-4)
    assert v["accepted"] and v["radius"] == 0.05 and len(v["window"][0]) == 1


def test_trust_region_noise_neither_rejects_nor_resizes():
    """A rise within the seed noise is accepted; a noisy rho is inconclusive (radius kept)."""
    v = _trial(BASE, 0.01, 0.1)  # paired se 0.058 > the rise; rho -0.1 +- 0.58
    assert v["accepted"] and v["radius"] == 0.05
    v = _trial(BASE, -0.01, 0.2)  # rho 0.1 +- ~1: no evidence either way
    assert v["accepted"] and v["radius"] == 0.05


def test_trust_region_pools_rho_over_the_window():
    """Trials individually inconclusive (rho 0.1 +- 0.29) become conclusive pooled (4 trials:
    +- 0.14) and shrink the radius (the window then restarts); at most RHO_WINDOW are held."""
    state = dict(BASE)
    radii = []
    for _ in range(OZ.RHO_WINDOW):
        v = _trial(state, -0.01, 0.05)
        assert v["accepted"]
        radii.append(v["radius"])
        state.update(radius=v["radius"], **OZ.window_state(v["window"]))
    assert radii[:3] == [0.05] * 3 and radii[3:] == [0.05 * OZ.SHRINK] * 2
    state = dict(BASE)
    for _ in range(OZ.RHO_WINDOW + 3):
        v = _trial(state, -0.05, 1e-4)
        state.update(radius=v["radius"], **OZ.window_state(v["window"]))
    assert len(v["window"][0]) == OZ.RHO_WINDOW and v["radius"] == 0.05


def test_trust_region_one_seed_uses_the_window_scatter():
    """One seed per iterate: no decision before RHO_MIN_ONE_SEED trials, then the scatter of
    the window's changes about rho * predicted is the noise."""
    state = {"radius": 0.05, "base_objectives": np.ones(1), "predicted": -0.1, "at_edge": True}
    for n, change in enumerate((-0.09, -0.095, -0.085, -0.09), start=1):
        v = OZ.judge(state, np.ones(1) + change, 0.05)
        if n < OZ.RHO_MIN_ONE_SEED:
            assert v["accepted"] and np.isnan(v["rho_se"]) and v["radius"] == 0.05
            state.update(**OZ.window_state(v["window"]))
    assert v["radius"] == 0.05 * OZ.GROW  # rho 0.9 +- 0.03 by the third trial
    rho, se, se_now = OZ.pooled_rho([-0.09, -0.01, -0.05], [-0.1] * 3, [np.nan] * 3)
    assert abs(rho - 0.5) < 1e-12 and se > 0.0 and se_now > 0.0


def test_trust_region_rejection_at_the_floor_rebases_instead_of_repeating():
    """At the floor a rejection cannot shrink: the verdict asks for the base point on new
    seeds, and the re-evaluated base is its own base (no judgement)."""
    lo, _hi = OZ.radius_bounds(0.05)
    v = _trial(dict(BASE, radius=lo), 0.02, 1e-4)
    assert not v["accepted"] and v["rebase"] and v["radius"] == lo
    v = OZ.judge(dict(BASE, radius=lo, rebase=True), np.full(4, 3.0), 0.05)
    assert v["accepted"] and not v["rebase"] and np.isnan(v["rho"])
    # the radius stays within its bounds
    _lo, hi = OZ.radius_bounds(0.05)
    assert _trial(dict(BASE, radius=hi), -0.09, 1e-4)["radius"] == hi


def test_lbfgs_solves_a_quadratic_with_the_identity_preconditioner():
    rng = np.random.default_rng(0)
    n = 12
    a = rng.normal(size=(n, n))
    hess = a @ a.T + n * np.eye(n)
    x_star = rng.normal(size=n)

    def grad(x):
        return hess @ (x - x_star)

    x = np.zeros(n)
    pairs = []
    g = grad(x)
    for _ in range(40):
        if np.linalg.norm(g) < 1e-12:
            break
        d, _ = OZ.lbfgs_direction(g, pairs, lambda v: v)
        # exact line search on the quadratic
        t = -float(g @ d) / float(d @ hess @ d)
        x_new = x + t * d
        g_new = grad(x_new)
        pairs, _ = OZ.add_pair(pairs, x_new - x, g_new - g)
        x, g = x_new, g_new
    assert np.linalg.norm(x - x_star) < 1e-6 * np.linalg.norm(x_star)
    # negative curvature is not kept
    assert not OZ.add_pair([], np.ones(3), -np.ones(3))[1]


def test_mma_subproblem_decreases_a_linear_objective_at_the_area_bound():
    """Shrinking pays (the objective falls with the area, like drag): the step falls in the
    objective, keeps the linearised area at its bound, every move within the move limit."""
    n = 40
    x = np.zeros(n)
    g_area = np.full(n, -1.0)  # area = A0 - sum x (phi up -> body shrinks)
    g0 = np.linspace(0.5, 1.5, n)  # every variable pays to shrink, the right ones more
    area0 = 100.0
    low, upp = OZ.mma_asymptotes({}, x, 0.05, 0)
    xs, lam = OZ.mma_step(x, g0, area0, g_area, area0, low, upp, 0.05)
    dx = xs - x
    assert float(g0 @ dx) < 0.0
    assert float(g_area @ dx) >= -1e-6 * area0  # the area does not fall (to first order)
    assert np.abs(dx).max() <= 0.05 + 1e-12
    assert lam[0] > 0.0  # the constraint is active


def test_warm_state_round_trip(tmp_path):
    """The optimiser's per-seed warm start file restores the gas exactly."""
    import jax.numpy as jnp  # noqa: PLC0415
    import optimize as O  # noqa: PLC0415

    from jax_dsmc.state import CollideState, ParticleState  # noqa: PLC0415

    rng = np.random.default_rng(1)
    state = ParticleState(jnp.asarray(rng.normal(size=(7, 2))), jnp.asarray(rng.normal(size=(7, 3))),
                          jnp.zeros(7, jnp.int32), jnp.asarray(rng.uniform(size=7) < 0.5),
                          jnp.zeros(7))  # fmt: skip
    cstate = CollideState(jnp.asarray(rng.uniform(size=(4, 1, 1))), jnp.zeros((4, 1, 1)))
    path = tmp_path / "warm" / "seed0.npz"
    assert O.load_warm(path) is None
    O.save_warm(path, (state, cstate))
    s2, c2 = O.load_warm(path)
    for a, b in ((state.x, s2.x), (state.v, s2.v), (state.alive, s2.alive),
                 (cstate.vremax, c2.vremax), (cstate.remainder, c2.remainder)):  # fmt: skip
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_warm_start_carries_the_cartesian_cells_onto_a_mesh_with_more_sub_cells():
    """2026-10-21: the gas of a mesh with 2 sub-cells carried onto one with 6 (the body moved:
    more split cells).  The particles inside the new body are dropped, the Cartesian cells
    keep their majorant and remainder, the new mesh's sub-cells start from its fresh state."""
    from types import SimpleNamespace  # noqa: PLC0415

    import jax.numpy as jnp  # noqa: PLC0415

    from examples import ensemble_mesh as EM  # noqa: PLC0415
    from jax_dsmc.mesh.types import AnalyticBody  # noqa: PLC0415
    from jax_dsmc.state import CollideState, ParticleState  # noqa: PLC0415

    n_cart, old_sub, new_sub = 12, 2, 6
    x = jnp.asarray([[0.0, 0.0], [5.0, 5.0], [0.5, 0.0]])
    state = ParticleState(x, jnp.zeros((3, 3)), jnp.zeros(3, jnp.int32), jnp.ones(3, bool),
                          jnp.zeros(3))  # fmt: skip
    old = CollideState(jnp.full((n_cart + old_sub, 1, 1), 7.0),
                       jnp.full((n_cart + old_sub, 1, 1), 0.25))  # fmt: skip
    fresh_c = CollideState(jnp.full((n_cart + new_sub, 1, 1), 1.0),
                           jnp.zeros((n_cart + new_sub, 1, 1)))  # fmt: skip
    fresh = (state, fresh_c)
    mesh = SimpleNamespace(kind="analytic",
                           body=AnalyticBody(jnp.zeros(2), jnp.asarray(1.0)))  # fmt: skip
    grid = SimpleNamespace(n=(3, 4), n_sub=new_sub)
    s, c = EM._carried(state, old, mesh, fresh, grid)
    np.testing.assert_array_equal(np.asarray(s.alive), [False, True, False])
    assert c.vremax.shape == fresh_c.vremax.shape
    np.testing.assert_array_equal(np.asarray(c.vremax[:n_cart]), 7.0)
    np.testing.assert_array_equal(np.asarray(c.vremax[n_cart:]), 1.0)
    np.testing.assert_array_equal(np.asarray(c.remainder[:n_cart]), 0.25)
    np.testing.assert_array_equal(np.asarray(c.remainder[n_cart:]), 0.0)


def test_replay_sizes_follow_the_record():
    """2026-10-22: the shape VJP's traced replay runs the passes and lane the captured record
    needed (the deepest event pass + 2, twice the most pass-0 events of a step, powers of
    two), at least the minimum and the floor and never above the forward's sizes; a dense
    record (another tracker) keeps the forward's."""
    import dataclasses  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    import jax.numpy as jnp  # noqa: PLC0415

    from examples import ensemble_mesh as EM  # noqa: PLC0415
    from jax_dsmc.state import SimStatic  # noqa: PLC0415

    static = SimStatic(D=2, N=195_000, S=1, G=1, A_max=256, K_bounce=32, n_steps=10,
                       tracker="lane")  # fmt: skip
    face = -jnp.ones((3, 1, 195_000), jnp.int32)  # (steps, pass 0, slots)
    face = face.at[1, 0, :300].set(5)  # 300 element hits in the busiest step
    walls = SimpleNamespace(head=SimpleNamespace(face=face),
                            pass_=jnp.asarray([[1, 2, -1], [5, -1, -1], [-1, -1, -1]]))  # fmt: skip
    small = EM.replay_static(static, walls)
    assert (small.K_bounce, small.lane_size) == (8, 1024)  # pass 5 + 2 -> 8; 2 x 300 -> 1024
    quiet = EM.replay_static(static, SimpleNamespace(head=SimpleNamespace(face=-jnp.ones((2, 1, 9))),
                                                     pass_=-jnp.ones((2, 4), jnp.int32)))  # fmt: skip
    assert (quiet.K_bounce, quiet.lane_size) == (EM.REPLAY_PASSES_MIN, EM.REPLAY_LANE_MIN)
    grown = EM.replay_static(static, walls, floor=(16, 2048))
    assert (grown.K_bounce, grown.lane_size) == (16, 2048)
    deep = EM.replay_static(static, SimpleNamespace(head=walls.head, pass_=jnp.asarray([[40]])))
    assert deep.K_bounce == 32  # capped at the forward's
    stencil = dataclasses.replace(static, tracker="stencil")
    assert EM.replay_static(stencil, walls) is stencil


def _circle_levelset(centre=(5.0, 5.0)):
    import jax.numpy as jnp  # noqa: PLC0415

    from jax_dsmc.mesh import levelset  # noqa: PLC0415

    lo, hi = np.zeros(2), np.full(2, 10.0)
    k_lo, m = levelset.grid_for_box(jnp.asarray(lo), jnp.asarray(hi), 0.125)
    ls = levelset.from_analytic(jnp.asarray(centre), 2.0, k_lo, 0.125, m, n_sample=16)
    return ls, lo, hi


def test_centroid_constraint_projects_and_restores():
    """2026-10-23 (``fix_centroid``): a step corrected along rigid translations changes
    neither the area nor the centroid to first order, and the correction is a translation
    (smooth: no more high-frequency content than the step's own); the constrained gradient
    has no translation part; a body displaced by a quarter metre is moved back rigidly, its
    area kept."""
    import optimize as O  # noqa: PLC0415
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    ls, lo, hi = _circle_levelset()
    _area0, g_area = O.area_and_gradient(ls, lo, hi)
    cons = O.constraint_gradients(ls, lo, hi, g_area, {"fix_centroid": 1, "band": 1.0})
    assert cons.shape == (2, 3, *ls.coef.shape)
    rows, modes = cons
    rng = np.random.default_rng(0)
    v0 = gaussian_filter(rng.normal(size=ls.coef.shape), 4.0)  # a smooth step
    v, _nu = O.tangent_component(v0, cons)
    np.testing.assert_allclose(rows.reshape(3, -1) @ v.ravel(), 0.0, atol=1e-9)
    # the correction lies in the span of the area row and the translations
    basis = modes.reshape(3, -1).T
    fit = np.linalg.lstsq(basis, (v0 - v).ravel(), rcond=None)
    np.testing.assert_allclose(basis @ fit[0], (v0 - v).ravel(), atol=1e-9)
    g, _ = O.constrained_gradient(v0, cons)
    np.testing.assert_allclose(modes.reshape(3, -1) @ g.ravel(), 0.0, atol=1e-9)
    c0 = np.asarray(O._centroid_of_coef(ls.coef, ls, lo, hi))
    np.testing.assert_allclose(c0, [5.0, 5.0], atol=1e-6)
    for centre in ((5.03, 5.0), (5.25, 4.9)):
        moved, _, _ = _circle_levelset(centre)
        a_moved = O.area_and_gradient(moved, lo, hi)[0]
        back, err = O.project_centroid(moved, c0, lo, hi)
        assert err < 1e-6
        assert abs(O.area_and_gradient(back, lo, hi)[0] - a_moved) < 1e-4 * a_moved
        # a rigid move: the body back where the centred circle is
        phi = np.asarray(O.L.value(back, O.L.knots(back)))
        ref = np.asarray(O.L.value(ls, O.L.knots(ls)))
        near = np.abs(ref) < 0.5
        assert np.abs(phi - ref)[near].max() < 2e-3
    # a single stacked row is the area-only projection
    w1, _ = O.tangent_component(v, g_area)
    w2, _ = O.tangent_component(v, g_area[None])
    np.testing.assert_allclose(w1, w2, atol=1e-12)


def test_prodigy_cosine_cap_bounds_the_step(tmp_path):
    """2026-10-23 ``ox_prodigycos``: Prodigy's step never exceeds the cosine cap (``4 step`` at
    the start, ``step / 4`` from :data:`optax_rules.COSINE_ITERATES` on), and its state
    survives a save / load (a resumed run continues the schedule)."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import optax_rules as OX  # noqa: PLC0415
    import optimize as O  # noqa: PLC0415

    ls, lo, hi = _circle_levelset()
    g_area = O.area_and_gradient(ls, lo, hi)[1]
    cfg = {"smooth": 2.0, "band": 1.0, "step": 0.05}
    ctx = OX.Context()
    tx = OX.build("ox_prodigycos", ctx, cfg["step"])
    state = tx.init(jnp.asarray(ls.coef))
    rng = np.random.default_rng(2)
    signal = rng.normal(size=ls.coef.shape)
    for k in range(6):
        g = signal + 0.3 * rng.normal(size=ls.coef.shape)
        delta, state = OX.step(tx, state, ctx, g, ls, g_area, cfg)
        cap = float(OX._cap_cosine(k)) * cfg["step"]
        assert np.abs(delta).max() <= cap * (1 + 1e-12)
        assert np.all(np.isfinite(delta))
    assert abs(float(OX._cap_cosine(OX.COSINE_ITERATES)) - OX.CAP / 16.0) < 1e-12
    OX.save_state(tmp_path / "s.npz", state)
    back = OX.load_state(tmp_path / "s.npz", tx.init(jnp.asarray(ls.coef)))
    leaves = zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(back), strict=True)
    for a, b in leaves:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _rule_iterates(rule, n, noise, tmp_path):
    """``n`` iterates of ``optimize.rule_step`` for ``rule`` on the test circle, the gradient of
    ``0.5 |coef - target|^2`` on the band (target: the circle moved 0.5 m in x) plus ``noise``
    times white noise, through after_step and a state save / load each iterate.  Returns
    ``(rows, objectives, steps, near)`` (``near``: the knots within 0.5 m of the last surface,
    inside the band -- where a step's roughness shapes the body)."""
    import dataclasses  # noqa: PLC0415

    import jax.numpy as jnp  # noqa: PLC0415
    import optimize as O  # noqa: PLC0415
    import optimizers as OZ  # noqa: PLC0415

    ls, lo, hi = _circle_levelset()
    target = np.asarray(_circle_levelset((5.5, 5.0))[0].coef)
    cfg = {"optimizer": rule, "step": 0.05, "smooth": 2.0, "band": 1.0}
    area0 = O.area_and_gradient(ls, lo, hi)[0]
    state = {"area0": area0}
    rng = np.random.default_rng(0)
    rows, objectives, steps = [], [], []
    for k in range(n):
        band = np.asarray(O.L.coefficient_band(ls, cfg["band"] + ls.h))
        diff = (np.asarray(ls.coef) - target) * band
        objectives.append(0.5 * float(np.sum(diff**2)))
        g_full = diff + noise * rng.normal(size=diff.shape) * band
        g_area = O.area_and_gradient(ls, lo, hi)[1]
        g, _ = O.tangent_component(g_full, g_area)
        base, delta, _pred, state, extra = O.rule_step(
            cfg, ls, g, g_full, g_area, np.array([objectives[-1]]), state, k, lo, hi
        )
        OZ.save_state(tmp_path / "s.npz", state)
        state = OZ.load_state(tmp_path / "s.npz")
        rows.append(extra)
        steps.append(delta)
        stepped = dataclasses.replace(base, coef=base.coef + jnp.asarray(delta))
        ls, _, _ = O.after_step(stepped, "tangent", area0, lo, hi, 0.0)
    near = np.abs(np.asarray(O.L.value(ls, O.L.knots(ls)))).reshape(ls.coef.shape) < 0.5
    return rows, objectives, steps, near


def test_lbfgs_fixed_step_accepts_every_trial_at_the_fixed_size(tmp_path):
    """2026-10-24 ``lbfgs_fs``: the L-BFGS direction at the fixed largest displacement
    ``step`` -- no trust radius, no rejection -- with curvature pairs accumulating; on the
    quadratic the objective falls."""
    rows, objectives, _, _ = _rule_iterates("lbfgs_fs", 6, 0.0, tmp_path)
    for r in rows:
        assert r["accepted"] and not r["rebase"]
        np.testing.assert_allclose(r["step_max"], 0.05, rtol=1e-9)
    assert rows[-1]["lbfgs_pairs"] >= 2
    assert objectives[-1] < objectives[0]


def test_mma_averaged_smoothed_step_is_smooth_on_a_noisy_gradient(tmp_path):
    """2026-10-24 ``mma_s``: on a noisy gradient its step has a far smaller high-frequency
    share than plain ``mma``'s (whose separable step follows every coefficient's noise), and
    its averaged gradient survives the state round trip (``mma_t`` counts the iterates)."""

    def high_frequency(d, near):
        lap = 4 * d[1:-1, 1:-1] - d[2:, 1:-1] - d[:-2, 1:-1] - d[1:-1, 2:] - d[1:-1, :-2]
        inner = near[1:-1, 1:-1]
        return float(np.sqrt(np.sum(lap[inner] ** 2) / np.sum(d[1:-1, 1:-1][inner] ** 2)))

    _, _, plain, near_plain = _rule_iterates("mma", 3, 0.5, tmp_path)
    _, objectives, smoothed, near = _rule_iterates("mma_s", 3, 0.5, tmp_path)
    import optimizers as OZ  # noqa: PLC0415

    assert int(OZ.load_state(tmp_path / "s.npz")["mma_t"]) == 3
    assert all(np.abs(d).max() > 0 for d in smoothed)
    assert high_frequency(smoothed[-1], near) < 0.25 * high_frequency(plain[-1], near_plain)
    assert np.all(np.isfinite(objectives))


def test_recentre_moves_the_body_back_by_whole_knots_exactly():
    """2026-10-25 (``recentre``): a body displaced by three knots and a bit is moved back to
    within half a knot of its start by an integer shift of its coefficients -- the same
    spline, no resample -- and an optimiser state's coefficient-shaped leaves move with it."""
    import jax.numpy as jnp  # noqa: PLC0415
    import optimize as O  # noqa: PLC0415

    ls, lo, hi = _circle_levelset()
    c0 = np.asarray(O._centroid_of_coef(ls.coef, ls, lo, hi))
    coef = np.asarray(ls.coef)
    moved = dataclasses.replace(ls, coef=jnp.asarray(O.shift_knots(coef, (-3, 1))))
    back, n = O.recentre(moved, c0 + np.array([0.2 * ls.h, 0.0]), lo, hi)
    assert n.tolist() == [3, -1]
    np.testing.assert_array_equal(np.asarray(back.coef)[4:-4, 4:-4], coef[4:-4, 4:-4])
    same, n0 = O.recentre(back, c0, lo, hi)
    assert not n0.any() and same is back
    state = {"m": coef, "pairs": np.stack([coef, 2 * coef]), "count": np.int32(7), "x": 1.5}
    out = O.shift_state(state, np.array([2, 0]), coef.shape)
    np.testing.assert_array_equal(out["m"], O.shift_knots(coef, (2, 0)))
    np.testing.assert_array_equal(out["pairs"][1], O.shift_knots(2 * coef, (2, 0)))
    assert out["count"] == 7 and out["x"] == 1.5


def test_lift_drag_weights_are_the_ratio_gradient():
    """2026-10-25 (``objective lift_drag``): the per-hit momentum weights are ``dJ/dF`` of
    ``J = -L / D`` at the windowed forces, and the drag keeps its unweighted reward (a
    weight ``(1, 0)`` gives it bit for bit)."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from examples import ensemble_mesh as EM  # noqa: PLC0415

    cfg = {"steps": 4, "window": 3, "objective": "lift_drag"}
    rng = np.random.default_rng(1)
    mom = np.zeros((4, 6, 3))
    mom[:, 4:, 0] = rng.uniform(1.0, 2.0, (4, 2))
    mom[:, 4:, 1] = rng.normal(0.0, 0.3, (4, 2))
    tallies = types.SimpleNamespace(momentum=jnp.asarray(mom))
    f = EM.objective_forces(cfg, tallies, 2, 0.5)
    np.testing.assert_allclose(f, 0.5 * mom[1:, 4:, :2].sum(axis=(0, 1)))
    assert EM.objective_value(cfg, tallies, 2, 0.5) == -f[1] / f[0]
    w = np.asarray(EM.objective_weights(cfg, tallies, 2, 0.5))
    np.testing.assert_allclose(w, jax.grad(lambda x: -x[1] / x[0])(jnp.asarray(f)), rtol=1e-12)
    assert EM.objective_weights(dict(cfg, objective="drag"), tallies, 2, 0.5) is None
    events = types.SimpleNamespace(
        v_in=jnp.asarray(rng.normal(size=(5, 3))), v_out=jnp.asarray(rng.normal(size=(5, 3)))
    )
    plain = EM._hit_momentum(events, 2.0, 2, None)
    np.testing.assert_array_equal(EM._hit_momentum(events, 2.0, 2, jnp.array([1.0, 0.0])), plain)
