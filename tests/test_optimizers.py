"""The optimiser's step rules (examples/mesh/optimizers.py) on small numpy problems: the trust
region's judgement (paired per-seed differences, pooled rho, the floor's re-evaluation on new
seeds), L-BFGS on a quadratic, and one MMA subproblem under the area constraint."""

import os
import sys

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
    """2026-10-23 (``fix_centroid``): a step projected on the stack of the area's and the
    centroid's gradients changes neither to first order; a body displaced by a quarter metre
    is moved back to the target centroid with its area kept."""
    import optimize as O  # noqa: PLC0415

    ls, lo, hi = _circle_levelset()
    _area0, g_area = O.area_and_gradient(ls, lo, hi)
    stack = O.constraint_gradients(ls, lo, hi, g_area, {"fix_centroid": 1})
    assert stack.shape == (3, *ls.coef.shape)
    rng = np.random.default_rng(0)
    v, _nu = O.tangent_component(rng.normal(size=ls.coef.shape), stack)
    np.testing.assert_allclose(stack.reshape(3, -1) @ v.ravel(), 0.0, atol=1e-9)
    c0 = np.asarray(O._centroid_of_coef(ls.coef, ls, lo, hi))
    np.testing.assert_allclose(c0, [5.0, 5.0], atol=1e-6)
    # Newton: quadratic -- a step-sized offset in the default iterations, a large one in four
    for centre, iters in (((5.03, 5.0), 3), ((5.25, 4.9), 4)):
        moved, _, _ = _circle_levelset(centre)
        a_moved = O.area_and_gradient(moved, lo, hi)[0]
        back, err = O.project_centroid(moved, c0, lo, hi, iters=iters)
        assert err < 1e-8
        # first order only (0.3 % at 0.27 m); after_step's project_area restores it after
        assert abs(O.area_and_gradient(back, lo, hi)[0] - a_moved) < 5e-3 * a_moved
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
