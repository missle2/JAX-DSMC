"""NTC collision driver (spec section 8).

Two of these are regression tests for defects that were found by measurement and
would otherwise have been silent:

* **defect 1** (8.1) -- ``n = floor(f)`` froze ``dt``, ``fnum`` and ``cell_volume``
  out of the gradient entirely, measured as exactly ``[0.0, 0.0, 0.0]``;
* **defect 2** (8.4.1) -- an empty cell's attempt wrote stale velocities over a
  neighbouring cell's particles, violating momentum in 176/300 runs.

Neither produced an error, a ``NaN``, or a diagnostic.  They are the reason the
assertions below are on exact zeros and exact conservation rather than tolerances.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

jax.config.update("jax_enable_x64", True)

from jax_dsmc import sampling  # noqa: E402
from jax_dsmc.collide import make_prefactor  # noqa: E402
from jax_dsmc.collide.ntc import (  # noqa: E402
    F_UP,
    attempt_count,
    collide,
    pair_ntc,
    pair_perm,
    thermal_majorant,
    vremax_init,
)
from jax_dsmc.collide.vss import collision_rate  # noqa: E402
from jax_dsmc.grid import num_cells  # noqa: E402
from jax_dsmc.sort import build_table  # noqa: E402
from jax_dsmc.state import (  # noqa: E402
    CollideState,
    GridSpec,
    ParticleState,
    SimParams,
    SimStatic,
    SpeciesParams,
)

AR_MASS, AR_DIAM, AR_TREF = 6.63e-26, 4.11e-10, 273.15


def build(omega=1.0, alpha=1.0, n=(1,), N=2000, T=300.0, A_max=400, dt=1e-6, seed=0):
    """A single-species case; ``n`` sets the dimension and the cell count."""
    mass = jnp.array([AR_MASS])
    mr = jnp.array([[AR_MASS / 2]])
    diam = jnp.array([[AR_DIAM]])
    om = jnp.array([[omega]])
    tref = jnp.array([[AR_TREF]])
    al = jnp.array([[alpha]])
    sp = SpeciesParams(
        mass=mass,
        diam=diam,
        omega=om,
        tref=tref,
        alpha=al,
        mr=mr,
        prefactor=make_prefactor(diam, om, tref, mr),
    )
    g = GridSpec(lo=jnp.zeros(len(n)), hi=jnp.full(len(n), 1e-4), n=n)
    C = num_cells(g)
    static = SimStatic(N=N, D=len(n), S=1, G=1, A_max=A_max, K_bounce=3, n_steps=1)
    params = SimParams(dt=jnp.float64(dt), fnum=jnp.full((C,), 1e12), epsilon=jnp.float64(0.0))
    k = jax.random.PRNGKey(seed)
    x = sampling.sample_positions(jax.random.fold_in(k, 1), N, g)
    v = sampling.sample_maxwellian(
        jax.random.fold_in(k, 2), N, jnp.float64(T), mass[0], jnp.zeros(3)
    )
    state = ParticleState(
        x=x, v=v, species=jnp.zeros(N, jnp.int32), alive=jnp.ones(N, bool), logh_acc=jnp.zeros(N)
    )
    cs = CollideState(
        vremax=vremax_init(
            jnp.full((C,), T), om[0, 0], tref[0, 0], mr[0, 0], sp.prefactor[0, 0], C, 1
        )
    )
    return state, cs, g, sp, params, static, jnp.full((C,), T), k


# --------------------------- pair selection ---------------------------------


def test_pair_ntc_is_uniform_on_the_complement():
    """``j`` must be uniform on ``{0..npc-1} \\ {i}`` -- chi-square, not eyeballing."""
    npc, n = 8, 60_000
    keys = jax.random.split(jax.random.PRNGKey(21), n)
    ij = jax.vmap(lambda k: pair_ntc(k, jnp.array([npc])))(keys)
    i = np.asarray(ij[0]).ravel()
    j = np.asarray(ij[1]).ravel()
    assert not (i == j).any(), "pair_ntc produced a self-pair"
    assert stats.chisquare(np.bincount(i, minlength=npc)).pvalue > 1e-3
    for a in range(npc):
        comp = j[i == a]
        counts = np.bincount(comp, minlength=npc)
        assert counts[a] == 0
        assert stats.chisquare(np.delete(counts, a)).pvalue > 1e-4


@pytest.mark.parametrize("npc", [0, 1])
def test_pair_ntc_clamps_degenerate_cell_counts(npc):
    """DEFECT 2 regression: ``randint(k, shape, 0, 0)`` returns 0 SILENTLY.

    With an unclamped ``npc`` an empty or single-particle cell yields ``i == j == 0``
    and, combined with ``order[starts + 0]``, indexes into a *neighbouring* cell.
    The clamp is what makes the scratch row reachable instead.
    """
    for s in range(200):
        i, j = pair_ntc(jax.random.PRNGKey(s), jnp.array([npc]))
        assert int(i[0]) != int(j[0]), f"npc={npc} seed={s} gave a self-pair"


def test_pair_perm_is_a_perfect_matching():
    """``pair_perm`` must use every particle exactly once, with no self-pairs."""
    for npc in (2, 4, 16, 64):
        i, j = pair_perm(jax.random.PRNGKey(5), npc)
        used = np.concatenate([np.asarray(i), np.asarray(j)])
        assert sorted(used.tolist()) == list(range(npc))
        assert not (np.asarray(i) == np.asarray(j)).any()


# --------------------------- attempt count ----------------------------------


def test_stochastic_rounding_is_unbiased():
    """``E[n] == f`` -- the property the deterministic carry also had, kept here."""
    counts = jnp.full((1,), 50)
    vol = jnp.full((1,), 1e-12)
    # vmapped over the 4000 seeds rather than looped: the loop dispatched 4000
    # separate eager calls for the same statistic.  Identical keys, identical draws.
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(4000))
    n, _ = jax.vmap(attempt_count, in_axes=(None, None, None, None, None, 0))(
        counts,
        jnp.full((1,), 1e-16),
        vol,
        jnp.float64(1e-6),
        jnp.full((1,), 1e12),
        keys,
    )
    est = np.asarray(n[:, 0])
    npc = 50.0
    f = 0.5 * npc * (npc - 1) * 1e-16 * 1e-6 * 1e12 / 1e-12
    assert np.mean(est) == pytest.approx(f, rel=0.02)


def test_log_p_is_the_log_probability_of_the_realised_count():
    counts = jnp.full((1,), 50)
    vol = jnp.full((1,), 1e-12)
    for s in range(50):
        n, logP = attempt_count(
            counts,
            jnp.full((1,), 1.3e-16),
            vol,
            jnp.float64(1e-6),
            jnp.full((1,), 1e12),
            jax.random.PRNGKey(s),
        )
        npc = 50.0
        f = 0.5 * npc * (npc - 1) * 1.3e-16 * 1e-6 * 1e12 / 1e-12
        frac = f - np.floor(f)
        went_up = int(n[0]) == int(np.floor(f)) + 1
        assert float(logP[0]) == pytest.approx(np.log(frac if went_up else 1 - frac), rel=1e-12)


def test_defect1_regression_count_score_carries_dt_fnum_volume():
    """DEFECT 1 regression: these three gradients were measured as EXACTLY 0.0.

    ``dt``, ``fnum`` and ``cell_volume`` appear *only* in the attempt count ``f``.
    Under ``n = floor(f)`` the path dies at the floor and no score term rescues it,
    because the per-pair scores depend on ``v`` and ``prefactor`` alone.  ``logP``
    is the entire sensitivity.
    """
    counts = jnp.full((4,), 30)
    vol = jnp.full((4,), 1e-12)

    def total_logp(dt, fnum, volume):
        _, logP = attempt_count(
            counts, jnp.full((4,), 1e-16), volume, dt, fnum, jax.random.PRNGKey(2)
        )
        return jnp.sum(logP)

    g = jax.grad(total_logp, argnums=(0, 1, 2))(jnp.float64(1e-6), jnp.full((4,), 1e12), vol)
    assert float(jnp.abs(g[0])) > 0.0, "dt gradient is zero -- defect 1 has regressed"
    assert float(jnp.max(jnp.abs(g[1]))) > 0.0, "fnum gradient is zero -- defect 1 regressed"
    assert float(jnp.max(jnp.abs(g[2]))) > 0.0, "volume gradient is zero -- defect 1 regressed"


def test_attempt_count_itself_is_frozen():
    """``n`` is an integer decision and must carry no tangent (section 0.5)."""
    counts = jnp.full((3,), 20)
    g = jax.grad(
        lambda dt: jnp.sum(
            attempt_count(
                counts,
                jnp.full((3,), 1e-16),
                jnp.full((3,), 1e-12),
                dt,
                jnp.full((3,), 1e12),
                jax.random.PRNGKey(0),
            )[0].astype(jnp.float64)
        )
    )(jnp.float64(1e-6))
    assert float(g) == 0.0


# --------------------------- the majorant (U3) ------------------------------


def test_thermal_majorant_bounds_every_equilibrium_pair():
    """U3 (7.2.3): the a-priori bound must actually bound ``q``."""
    state, cs, g, sp, params, static, T, k = build(omega=0.81)
    sigma = thermal_majorant(T, sp.omega[0, 0], sp.tref[0, 0], sp.mr[0, 0], sp.prefactor[0, 0])
    z = jnp.zeros(state.v.shape[0], jnp.int32)
    perm = jax.random.permutation(jax.random.PRNGKey(1), state.v.shape[0])
    q = collision_rate(state.v, state.v[perm], z, z, sp)
    assert float(jnp.max(q)) < float(sigma[0])
    assert float(sigma[0] / jnp.max(q)) > 2.0


def test_vremax_init_is_the_thermal_rule_not_sparta_s_seed():
    """U3 (17.5): SPARTA's ``2*cxs*vscale`` fails the gate; we must not be using it."""
    _, cs, g, sp, _, _, T, _ = build(omega=0.81)
    sparta_seed = 2.0 * np.pi * AR_DIAM**2 * np.sqrt(2 * 1.380649e-23 * AR_TREF / AR_MASS)
    got = float(cs.vremax[0, 0, 0])
    # NB compare RATIOS: these are ~1e-15, and `pytest.approx`'s absolute floor of
    # 1e-12 would call any two of them equal regardless of `rel`.
    assert got / sparta_seed > 1.5, "vremax_init looks like SPARTA's seed, which fails U3"
    want = thermal_majorant(T, sp.omega[0, 0], sp.tref[0, 0], sp.mr[0, 0], sp.prefactor[0, 0])
    assert got / float(want[0]) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("omega", [0.75, 0.81, 1.0])
def test_collide_reports_no_majorant_violations(omega):
    """The U3 acceptance criterion, as a standing assertion (7.2.1)."""
    args = build(omega=omega, A_max=600)
    state, cs, g, sp, params, static, T, k = args
    table = build_table(state, g)
    _, _, diag = collide(state, table, cs, g, sp, params, static, T, 0, k)
    assert int(diag.majorant_violations) == 0
    assert int(diag.attempts_clipped) == 0
    assert float(diag.min_majorant_margin) >= 2.0


def test_maxwell_margin_is_exactly_f_up():
    """At ``omega == 1`` every ``q`` equals ``prefactor``, so ``Sigma/q == F_UP`` exactly.

    A sharp consistency check on the majorant construction: any other value means
    the seed and the rate disagree about the prefactor.
    """
    state, cs, g, sp, params, static, T, k = build(omega=1.0)
    table = build_table(state, g)
    _, _, diag = collide(state, table, cs, g, sp, params, static, T, 0, k)
    assert float(diag.min_majorant_margin) == pytest.approx(F_UP, rel=1e-12)


# --------------------------- the step ---------------------------------------


@pytest.mark.parametrize("n", [(1,), (4,), (2, 2), (2, 2, 2)])
def test_collide_conserves_momentum_and_energy(n):
    """Across all dimensions, and with empty cells present (DEFECT 2 regression)."""
    state, cs, g, sp, params, static, T, k = build(n=n, N=1500, dt=1e-5, A_max=300)
    table = build_table(state, g)
    new, _, diag = collide(state, table, cs, g, sp, params, static, T, 0, k)
    m = AR_MASS
    p0, p1 = jnp.sum(m * state.v, 0), jnp.sum(m * new.v, 0)
    e0, e1 = jnp.sum(0.5 * m * state.v**2), jnp.sum(0.5 * m * new.v**2)
    assert float(jnp.max(jnp.abs(p1 - p0)) / jnp.max(jnp.abs(p0))) < 1e-13
    assert float(jnp.abs(e1 - e0) / e0) < 1e-13
    assert int(diag.majorant_violations) == 0


def test_defect2_regression_empty_cells_do_not_clobber():
    """DEFECT 2: a grid with many empty cells must still conserve exactly.

    The mechanism is not a duplicate-index race -- it is deterministic. ``vi`` and
    ``vj`` are gathered from the SAME pre-scatter ``v``, so wherever ``pi == pj``
    the second ``.set`` writes stale data over the first.  The scratch row at index
    ``N`` is what makes that harmless.
    """
    # 64 cells, 40 particles: most cells are empty or hold a single particle
    state, cs, g, sp, params, static, T, k = build(n=(4, 4, 4), N=40, dt=1e-4, A_max=64)
    table = build_table(state, g)
    assert int(jnp.sum(table.counts == 0)) > 0, "test needs empty cells to be meaningful"
    m = AR_MASS
    for s in range(25):
        new, _, _ = collide(state, table, cs, g, sp, params, static, T, s, jax.random.PRNGKey(s))
        p0, p1 = jnp.sum(m * state.v, 0), jnp.sum(m * new.v, 0)
        e0, e1 = jnp.sum(0.5 * m * state.v**2), jnp.sum(0.5 * m * new.v**2)
        assert float(jnp.max(jnp.abs(p1 - p0)) / jnp.max(jnp.abs(p0))) < 1e-13, f"seed {s}"
        assert float(jnp.abs(e1 - e0) / e0) < 1e-13, f"seed {s}"


def test_dead_particles_never_collide():
    state, cs, g, sp, params, static, T, k = build(n=(2,), N=600, dt=1e-5)
    dead = jnp.arange(600) >= 300
    state = ParticleState(
        x=state.x, v=state.v, species=state.species, alive=~dead, logh_acc=state.logh_acc
    )
    table = build_table(state, g)
    new, _, _ = collide(state, table, cs, g, sp, params, static, T, 0, k)
    assert bool(jnp.all(new.v[dead] == state.v[dead]))
    assert float(jnp.max(jnp.abs(new.logh_acc[dead]))) == 0.0


def test_collide_is_jittable_and_does_not_recompile():
    """Performance contract: repeated same-shape calls must not grow the cache."""
    state, cs, g, sp, params, static, T, k = build(n=(2, 2), N=800)
    table = build_table(state, g)
    f = jax.jit(collide, static_argnames=("static",))
    f(state, table, cs, g, sp, params, static, T, 0, k)[0].v.block_until_ready()
    n_after_first = f._cache_size()
    for step in range(1, 5):
        f(state, table, cs, g, sp, params, static, T, step, k)[0].v.block_until_ready()
    assert f._cache_size() == n_after_first


def test_collide_is_vmappable_over_an_ensemble():
    """``vmap`` over independent simulations -- proof of purity, and M8 needs it.

    ``dt`` is deliberately large enough that collisions actually occur: with too
    few, every ensemble member returns the untouched input and the test passes
    vacuously whatever the RNG is doing.
    """
    state, cs, g, sp, params, static, T, k = build(n=(2,), N=400, dt=1e-4, A_max=400)
    table = build_table(state, g)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    vs = jax.vmap(lambda kk: collide(state, table, cs, g, sp, params, static, T, 0, kk)[0].v)(keys)
    assert vs.shape == (3, 400, 3)
    n_moved = int(jnp.sum(jnp.any(jnp.abs(vs[0] - state.v) > 0, axis=-1)))
    assert n_moved > 0, "no collisions occurred, so the assertion below would be vacuous"
    assert not bool(jnp.allclose(vs[0], vs[1]))  # different keys => different trajectories


def test_no_data_dependent_while_loop_in_the_step():
    """Static-shape audit (section 14): a data-dependent trip count breaks the adjoint."""
    state, cs, g, sp, params, static, T, k = build(n=(2,), N=400)
    table = build_table(state, g)
    txt = jax.make_jaxpr(lambda s: collide(s, table, cs, g, sp, params, static, T, 0, k)[0].v)(
        state
    )
    assert "while" not in str(txt), "a while_loop appeared in the collide jaxpr"


def test_scores_are_finite_and_reach_colliding_particles():
    state, cs, g, sp, params, static, T, k = build(n=(1,), N=1000, dt=1e-5, A_max=400)
    table = build_table(state, g)
    new, _, _ = collide(state, table, cs, g, sp, params, static, T, 0, k)
    assert bool(jnp.all(jnp.isfinite(new.logh_acc)))
    assert float(jnp.max(jnp.abs(new.logh_acc))) > 0.0
