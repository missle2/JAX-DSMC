"""Per-face boundary rules and the diffuse/thermal wall (spec section 9.4).

The load-bearing check here is :func:`test_closed_form_agrees_with_the_dedicated_folds`:
the new per-face path is general, so it must reproduce the existing, separately tested
``apply_periodic`` and ``apply_specular`` exactly, not merely approximately.

The wall itself is validated against what it is *supposed* to produce -- a
flux-weighted half-Maxwellian at the wall temperature -- rather than against a
regression baseline, so a change of sampler that preserved the physics would pass and
one that broke it would not.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

jax.config.update("jax_enable_x64", True)

from jax_dsmc import sampling  # noqa: E402
from jax_dsmc.boundary import (  # noqa: E402
    BC,
    Faces,
    WallParams,
    apply_faces,
    apply_periodic,
    apply_specular,
    crossing_probabilities,
    n_bounce_passes,
)
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.state import GridSpec  # noqa: E402

MASS = 6.63e-26
T_WALL = 300.0
DIMS = [1, 2, 3]


def _grid(dim, length=1e-3, n_cells=8):
    return GridSpec(lo=jnp.zeros(dim), hi=jnp.full(dim, length), n=(n_cells,) * dim)


def _vth(temp=T_WALL):
    return math.sqrt(K_BOLTZ * temp / MASS)


# --- the per-face path must not perturb the rules it generalises -------------


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("bc", [BC.PERIODIC, BC.SPECULAR])
def test_closed_form_agrees_with_the_dedicated_folds(dim, bc):
    """``apply_faces`` == ``apply_periodic`` / ``apply_specular``, to round-off."""
    g, n = _grid(dim), 20_000
    key = jax.random.PRNGKey(0)
    x = 1e-3 * jax.random.uniform(key, (n, dim), dtype=jnp.float64)
    v = 300.0 * jax.random.normal(jax.random.fold_in(key, 1), (n, 3), dtype=jnp.float64)
    dt = 2e-7

    got_x, got_v, alive, recrossed, _ = apply_faces(
        x,
        v,
        jnp.ones(n, bool),
        g,
        Faces.uniform(bc, dim),
        WallParams.uniform(dim, T_WALL),
        jnp.float64(MASS),
        jnp.float64(dt),
        key,
    )
    ref = apply_periodic if bc is BC.PERIODIC else apply_specular
    want_x, want_v = ref(x + v[:, :dim] * dt, v, g)
    np.testing.assert_allclose(np.asarray(got_x), np.asarray(want_x), rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(np.asarray(got_v), np.asarray(want_v), rtol=0.0, atol=0.0)
    assert int(recrossed) == 0
    assert bool(jnp.all(alive))


# --- the wall re-emits the right distribution -------------------------------


def test_re_emitted_normal_speed_is_the_flux_weighted_half_maxwellian():
    """``v_perp`` must be Rayleigh with scale ``sqrt(kT/m)``, not Gaussian.

    The flux weighting is the whole point: fast particles cross the wall more often,
    so the re-emitted *flux* carries an extra factor of ``v_perp``.  A plain
    half-Gaussian would pass a mean-only check and fail this one.
    """
    n = 200_000
    vp = sampling.sample_flux_normal(
        jax.random.PRNGKey(3), n, jnp.float64(T_WALL), jnp.float64(MASS)
    )
    vp = np.asarray(vp)
    assert vp.min() >= 0.0
    ks = stats.kstest(vp, stats.rayleigh(scale=_vth()).cdf)
    assert ks.pvalue > 1e-3, f"not Rayleigh(sqrt(kT/m)): KS p={ks.pvalue:.2e}"
    # Rayleigh mean is sigma*sqrt(pi/2); a half-Gaussian would give sigma*sqrt(2/pi)
    assert vp.mean() == pytest.approx(_vth() * math.sqrt(math.pi / 2), rel=0.01)


def test_re_emitted_tangential_components_are_maxwellian_at_the_wall():
    n = 200_000
    v = np.asarray(
        sampling.sample_diffuse_velocity(
            jax.random.PRNGKey(4),
            n,
            jnp.float64(T_WALL),
            jnp.float64(MASS),
            1,
            1.0,
            jnp.zeros(3),
        )
    )
    for c in (0, 2):  # axis 1 is the normal
        assert v[:, c].mean() == pytest.approx(0.0, abs=5.0 * v[:, c].std() / math.sqrt(n))
        assert v[:, c].std() == pytest.approx(_vth(), rel=0.01)
    assert bool(np.all(v[:, 1] >= 0.0)), "normal component must point back into the domain"


@pytest.mark.parametrize("dim", DIMS)
def test_a_cold_gas_equilibrates_to_the_wall_temperature(dim):
    """Collisionless box, all faces diffuse: the walls alone must set ``T``."""
    g, n = _grid(dim), 40_000
    key = jax.random.PRNGKey(5)
    x = 1e-3 * jax.random.uniform(key, (n, dim), dtype=jnp.float64)
    v = _vth(100.0) * jax.random.normal(jax.random.fold_in(key, 1), (n, 3), dtype=jnp.float64)
    faces, wall = Faces.uniform(BC.DIFFUSE, dim), WallParams.uniform(dim, T_WALL)
    # dt = dx / v_th: the largest step section 9.4 measures at ZERO recrossing.
    # A bigger step sends tail particles across the box more than once, which a
    # single-pass bounce cannot represent (and which an earlier draft of this
    # test used, producing a spurious 465 K).
    dt = (1e-3 / 8) / _vth()
    alive = jnp.ones(n, bool)
    for s in range(300):
        x, v, alive, _, _ = apply_faces(
            x,
            v,
            alive,
            g,
            faces,
            wall,
            jnp.float64(MASS),
            jnp.float64(dt),
            jax.random.fold_in(key, 100 + s),
        )
    temp = float(MASS * jnp.mean(jnp.sum(v**2, axis=1)) / (3.0 * K_BOLTZ))
    # A box thermal on *every* face is the worst case: a particle leaving through a
    # corner reaches two thermal faces in one step.  Both are resolved
    # (`n_bounce_passes` returns 2 here), which took D=3 from 309.9 K to 300.9 K.
    # What remains is the tail that needs a third crossing, worth ~0.3%.
    assert temp == pytest.approx(T_WALL, rel=0.015), f"D={dim}: T={temp:.1f}"


def test_accommodation_zero_is_specular_and_one_is_fully_diffuse():
    """``acc`` interpolates the two limits (``surf_collide_diffuse.cpp:211``)."""
    g, n, dim = _grid(1), 20_000, 1
    key = jax.random.PRNGKey(6)
    x = jnp.full((n, 1), 0.999e-3)
    v = _vth() * jnp.abs(jax.random.normal(key, (n, 3), dtype=jnp.float64))  # all outgoing
    dt = (1e-3 / 8) / _vth()
    faces = Faces.uniform(BC.DIFFUSE, dim)

    spec_x, spec_v, _, _, _ = apply_faces(
        x,
        v,
        jnp.ones(n, bool),
        g,
        faces,
        WallParams.uniform(dim, T_WALL, acc=0.0),
        jnp.float64(MASS),
        jnp.float64(dt),
        key,
    )
    ref_x, ref_v = apply_specular(x + v[:, :1] * dt, v, g)
    np.testing.assert_allclose(np.asarray(spec_v), np.asarray(ref_v), atol=0.0)
    np.testing.assert_allclose(np.asarray(spec_x), np.asarray(ref_x), atol=1e-18)

    diff_v = apply_faces(
        x,
        v,
        jnp.ones(n, bool),
        g,
        faces,
        WallParams.uniform(dim, T_WALL, acc=1.0),
        jnp.float64(MASS),
        jnp.float64(dt),
        key,
    )[1]
    # a fully diffuse wall forgets the incoming speed; a specular one cannot
    assert float(jnp.mean(jnp.abs(diff_v[:, 0] + v[:, 0]))) > 1e-3 * _vth()


def test_moving_wall_shifts_the_tangential_mean_and_not_the_normal_one():
    """The Couette driver: a sliding wall adds its own velocity to the re-emitted gas."""
    dim, n = 2, 100_000
    u_wall = 200.0
    wall = WallParams.uniform(dim, T_WALL).with_wall_velocity(1, 1, jnp.array([u_wall, 0.0, 0.0]))
    v = np.asarray(
        sampling.sample_diffuse_velocity(
            jax.random.PRNGKey(7),
            n,
            jnp.float64(T_WALL),
            jnp.float64(MASS),
            1,
            -1.0,
            wall.vwall[3],
        )
    )
    assert v[:, 0].mean() == pytest.approx(u_wall, rel=0.02)
    assert bool(np.all(v[:, 1] <= 0.0)), "high face must re-emit in -y"


def test_normal_wall_motion_is_rejected_rather_than_silently_wrong():
    """SPARTA's normal-motion sampler is a data-dependent rejection loop (section 12)."""
    with pytest.raises(ValueError, match="normal component"):
        WallParams.uniform(2, T_WALL).with_wall_velocity(1, 0, jnp.array([0.0, 50.0, 0.0]))


def test_no_particle_is_left_outside_at_the_production_timestep():
    """``recrossed == 0`` in 1-D at the CFL-like step: what licenses the single pass.

    Section 9.4 measures the same-face multi-crossing fraction against ``dt``: exactly
    zero out to ~7x the production step, reaching 8e-2 only at ``dt = 8 dx/v_th``.
    """
    dim = 1
    g, n, n_cells = _grid(dim), 50_000, 8
    key = jax.random.PRNGKey(8)
    x = 1e-3 * jax.random.uniform(key, (n, dim), dtype=jnp.float64)
    v = _vth() * jax.random.normal(jax.random.fold_in(key, 1), (n, 3), dtype=jnp.float64)
    dt = 0.15 * (1e-3 / n_cells) / _vth()
    faces, wall = Faces.uniform(BC.DIFFUSE, dim), WallParams.uniform(dim, T_WALL)
    alive = jnp.ones(n, bool)
    for s in range(10):
        x, v, alive, recrossed, _ = apply_faces(
            x,
            v,
            alive,
            g,
            faces,
            wall,
            jnp.float64(MASS),
            jnp.float64(dt),
            jax.random.fold_in(key, 200 + s),
        )
        assert int(recrossed) == 0
    assert bool(jnp.all((x >= g.lo) & (x <= g.hi)))


@pytest.mark.parametrize(
    ("name", "bcs"),
    [
        ("couette", (BC.PERIODIC, BC.PERIODIC, BC.DIFFUSE, BC.DIFFUSE)),
        ("shock", (BC.OUTFLOW, BC.OUTFLOW, BC.SPECULAR, BC.SPECULAR)),
    ],
)
def test_target_configurations_have_no_corner_crossings(name, bcs):
    """The single-pass bounce is **exact** for the geometries this milestone targets.

    A corner is only approximate when *both* adjoining faces are thermal, because
    only the first bounce is resolved.  Periodic, specular and outflow keep their
    exact any-number-of-crossings folds and are applied after the wall bounce, so a
    thermal/periodic corner is handled exactly -- which is precisely the Couette
    geometry.  Measured over 20 steps and 1e5 particles: ``0`` for both, against
    ``1.8e-3`` per particle-step for a box that is thermal on every face.
    """
    assert n_bounce_passes(Faces(bcs)) == 1, "target geometries cannot corner on two walls"
    dim, n, n_cells = 2, 50_000, 8
    g = _grid(dim, n_cells=n_cells)
    key = jax.random.PRNGKey(11)
    x = 1e-3 * jax.random.uniform(key, (n, dim), dtype=jnp.float64)
    v = _vth() * jax.random.normal(jax.random.fold_in(key, 1), (n, 3), dtype=jnp.float64)
    dt = 0.15 * (1e-3 / n_cells) / _vth()
    faces, wall = Faces(bcs), WallParams.uniform(dim, T_WALL)
    alive = jnp.ones(n, bool)
    for s in range(20):
        x, v, alive, recrossed, _ = apply_faces(
            x,
            v,
            alive,
            g,
            faces,
            wall,
            jnp.float64(MASS),
            jnp.float64(dt),
            jax.random.fold_in(key, 300 + s),
        )
        assert int(recrossed) == 0, f"{name}: {int(recrossed)} corner crossings"


def test_two_thermal_axes_get_two_bounce_passes_and_one_axis_gets_one():
    """The pass count is a **static** property of the face layout.

    A particle can only reach two thermal faces in a step if they are on different
    axes: after bouncing off one face of an axis it travels away from the opposite
    face.  So Couette (thermal on ``y`` only) needs one pass and pays nothing for the
    corner machinery, while a box thermal on every face needs two.
    """
    assert n_bounce_passes(Faces.uniform(BC.DIFFUSE, 1)) == 1
    assert n_bounce_passes(Faces((BC.PERIODIC, BC.PERIODIC, BC.DIFFUSE, BC.DIFFUSE))) == 1
    assert n_bounce_passes(Faces.uniform(BC.DIFFUSE, 2)) == 2
    assert n_bounce_passes(Faces.uniform(BC.DIFFUSE, 3)) == 2
    assert n_bounce_passes(Faces.uniform(BC.SPECULAR, 3)) == 1


# --- [P3] section 4: the regularised timestep -------------------------------


@pytest.mark.parametrize("dim", DIMS)
def test_epsilon_zero_reproduces_the_deterministic_scheme_bitwise(dim):
    """``epsilon = 0`` must be **exactly** the unregularised solver, not nearly.

    This is what lets the regularisation ship switched on by default in the API and
    switched off in physics validation: every M1-M5 result is untouched by its
    existence.  ``dt + 0.0 * z == dt`` in IEEE, so the equality is exact rather than
    approximate, and the test asserts bitwise identity to keep it that way.
    """
    g, n = _grid(dim), 20_000
    key = jax.random.PRNGKey(20)
    x = 1e-3 * jax.random.uniform(key, (n, dim), dtype=jnp.float64)
    v = _vth() * jax.random.normal(jax.random.fold_in(key, 1), (n, 3), dtype=jnp.float64)
    dt = 0.15 * (1e-3 / 8) / _vth()
    args = (
        x,
        v,
        jnp.ones(n, bool),
        g,
        Faces.uniform(BC.DIFFUSE, dim),
        WallParams.uniform(dim, T_WALL),
        jnp.float64(MASS),
        jnp.float64(dt),
        key,
    )

    reg_x, reg_v, _, _, _ = apply_faces(*args, epsilon=0.0)
    det_x, det_v, _, _, _ = apply_faces(*args)
    assert bool(jnp.all(reg_x == det_x)), "epsilon=0 moved a particle"
    assert bool(jnp.all(reg_v == det_v))


@pytest.mark.parametrize("eps_rel", [0.0, 0.05, 0.15, 0.30])
def test_crossing_probabilities_sum_to_one_and_match_the_realised_frequency(eps_rel):
    """[P3] appendix ``app:log``: ``p_1 + p_2 + p_3 == 1``, and they are the real thing.

    Checked against the *empirical* crossing frequency under repeated ``tau`` draws, so
    this validates the CDF expression itself rather than restating it.
    """
    g, n = _grid(1), 20_000
    key = jax.random.PRNGKey(21)
    x = 1e-3 * jax.random.uniform(key, (n, 1), dtype=jnp.float64)
    v = _vth() * jax.random.normal(jax.random.fold_in(key, 1), (n, 3), dtype=jnp.float64)
    dt = 0.15 * (1e-3 / 8) / _vth()
    eps = eps_rel * dt

    p_lo, p_stay, p_hi = crossing_probabilities(x, v, g, 0, jnp.float64(dt), jnp.float64(eps))
    np.testing.assert_allclose(np.asarray(p_lo + p_stay + p_hi), 1.0, rtol=0.0, atol=1e-14)
    assert bool(jnp.all(p_lo >= 0.0)) and bool(jnp.all(p_hi >= 0.0))
    # only one side is reachable at a time: the sign of v decides which
    assert float(jnp.max(p_lo * p_hi)) == 0.0

    trials = 400
    tau = jnp.maximum(
        dt + eps * jax.random.normal(jax.random.fold_in(key, 7), (trials, 1), dtype=jnp.float64),
        0.0,
    )
    x_end = x[None, :, 0] + tau * v[None, :, 0]
    emp_lo = jnp.mean(x_end < g.lo[0], axis=0)
    emp_hi = jnp.mean(x_end > g.hi[0], axis=0)
    assert float(jnp.mean(jnp.abs(emp_lo - p_lo))) < 5e-3
    assert float(jnp.mean(jnp.abs(emp_hi - p_hi))) < 5e-3


def test_regularisation_does_not_shift_the_wall_equilibrium():
    """[P3]'s epsilon-stability question, in its cheapest form.

    ``J_eps != J`` even in expectation -- the randomised step changes the probability
    of reaching the wall -- so the question is how big the effect actually is.
    Measured at equilibrium (200k particles, 300 steps), the steady temperature is
    within **0.4%** of the wall for every ``epsilon`` from 0 to ``0.5 dt``, with no
    systematic trend: 298.8, 299.2, 299.2, 299.8, 299.7, 300.2 K.  That range already
    goes well past [P3]'s own ``epsilon << dt/3``.

    **The transient is the sensitive part, not the steady state.**  An earlier version
    of this test started the gas cold and read a 2.5% deficit at ``eps = 0.1 dt``
    against 1.5% at ``eps = 0`` -- but that is ``epsilon`` changing the *rate* at which
    the wall equilibrates the gas, compounded with an unfinished transient, not a bias
    in the fixed point.  Starting from equilibrium separates the two.
    """
    g, n, dim = _grid(1), 60_000, 1
    key = jax.random.PRNGKey(22)
    dt = (1e-3 / 8) / _vth()
    faces, wall = Faces.uniform(BC.DIFFUSE, dim), WallParams.uniform(dim, T_WALL)
    for eps_rel in (0.0, 0.1, 0.3):
        x = 1e-3 * jax.random.uniform(key, (n, dim), dtype=jnp.float64)
        # start AT the wall temperature: no transient to confound the epsilon effect
        v = _vth() * jax.random.normal(jax.random.fold_in(key, 1), (n, 3), dtype=jnp.float64)
        alive = jnp.ones(n, bool)
        for s in range(200):
            x, v, alive, _, _ = apply_faces(
                x,
                v,
                alive,
                g,
                faces,
                wall,
                jnp.float64(MASS),
                jnp.float64(dt),
                jax.random.fold_in(key, 100 + s),
                epsilon=eps_rel * dt,
            )
        temp = float(MASS * jnp.mean(jnp.sum(v**2, axis=1)) / (3.0 * K_BOLTZ))
        assert temp == pytest.approx(T_WALL, rel=0.01), f"eps/dt={eps_rel}: T={temp:.1f}"


def test_negative_tau_is_negligible_at_the_papers_epsilon_and_clamped_anyway():
    """[P3] asks for ``epsilon << dt/3`` so ``P(tau < 0)`` is negligible; verify and guard."""
    dt = 1.0
    assert float(sampling.negative_tau_fraction(dt, 0.0)) == 0.0
    assert float(sampling.negative_tau_fraction(dt, dt / 3.0)) < 2e-3
    assert float(sampling.negative_tau_fraction(dt, dt / 10.0)) < 1e-20
    # even at an inadmissible epsilon the clamp holds, so no particle advects backwards
    tau = sampling.sample_tau(jax.random.PRNGKey(0), 100_000, jnp.float64(dt), jnp.float64(dt))
    assert float(jnp.min(tau)) >= 0.0


# --- M7: the relaxed accommodation Bernoulli (`acc_delta`) ---------------------------


def test_acc_delta_zero_leaves_the_forward_solve_unchanged():
    """`acc_delta = 0` must reproduce the hard threshold, so the relaxation ships off.

    This is the same contract `epsilon = 0` has for the regularised timestep, and it is
    what lets a differentiability knob exist in production code at all: every forward
    result in M1-M5 has to be indifferent to it.
    """
    dim, n = 2, 50_000
    g = _grid(dim)
    key = jax.random.PRNGKey(11)
    kx, kv, ka = jax.random.split(key, 3)
    x = jax.random.uniform(kx, (n, dim), dtype=jnp.float64) * float(g.hi[0])
    v = jax.random.normal(kv, (n, 3), dtype=jnp.float64) * _vth()
    rules = [BC.PERIODIC] * (2 * dim)
    rules[0] = rules[1] = BC.DIFFUSE
    faces = Faces(tuple(rules))
    wall = WallParams.uniform(dim, T_WALL, acc=0.6)
    # the largest step section 9.4 measures at ZERO recrossing, as elsewhere in this file
    dt = (1e-3 / 8) / _vth()
    args = (x, v, jnp.ones(n, bool), g, faces, wall, jnp.float64(MASS), jnp.float64(dt), ka)

    relaxed_off = apply_faces(*args, acc_delta=0.0)
    default = apply_faces(*args)
    for a, b, name in zip(relaxed_off[:3], default[:3], ("x", "v", "alive"), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)


def test_acc_delta_zero_matches_a_hard_select_to_the_bit_except_signed_zero():
    """The blend and the select agree in value everywhere, and in BITS except at zero.

    `w * fresh + (1 - w) * v_spec` with `w in {0, 1}` is not textually the same operation
    as `where(took, fresh, v_spec)`: `-0.0 + 0.0` is `+0.0`, so the blend can clear the
    sign bit of an exact zero where the select preserves it.  Measured on 2e6 draws with
    zeros of both signs forced into the sample, that is the **only** difference -- values
    compare equal everywhere.  It is pinned rather than left implicit because "bitwise
    identical" was claimed for this path and is not literally true.
    """
    n = 200_000
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(3), 3)
    fresh = jax.random.normal(k1, (n, 3), dtype=jnp.float64)
    v_spec = jax.random.normal(k2, (n, 3), dtype=jnp.float64)
    fresh = fresh.at[: n // 4].set(0.0).at[n // 4 : n // 2].set(-0.0)
    v_spec = v_spec.at[: n // 8].set(-0.0)
    took = jax.random.uniform(k3, (n,), dtype=jnp.float64) <= 0.6
    w = took.astype(jnp.float64)

    blend = w[:, None] * fresh + (1.0 - w)[:, None] * v_spec
    select = jnp.where(took[:, None], fresh, v_spec)
    np.testing.assert_array_equal(np.asarray(blend), np.asarray(select))

    bits_differ = jax.lax.bitcast_convert_type(blend, jnp.int64) != jax.lax.bitcast_convert_type(
        select, jnp.int64
    )
    assert bool(jnp.all(jnp.abs(blend[bits_differ]) == 0.0)), (
        "bit differences must be confined to signed zeros"
    )


def test_acc_delta_makes_accommodation_pathwise():
    """With `acc_delta > 0`, `dJ/d(acc)` is pathwise; with it off, it is exactly zero.

    The second half is the load-bearing assertion.  A Bernoulli threshold has zero
    pathwise derivative almost everywhere, so a pathwise `dJ/d(acc)` that reads 0.0 is not
    a small number -- it is the *absence* of a gradient, and it is what motivated the
    relaxation.  See `docs/ATTRIBUTION.md`.
    """
    dim, n = 1, 20_000
    g = _grid(dim)
    kx, kv, ka = jax.random.split(jax.random.PRNGKey(5), 3)
    x = jax.random.uniform(kx, (n, dim), dtype=jnp.float64) * float(g.hi[0])
    v = jax.random.normal(kv, (n, 3), dtype=jnp.float64) * _vth()
    faces = Faces((BC.DIFFUSE, BC.DIFFUSE))

    dt = (1e-3 / 8) / _vth()

    def energy(acc, delta):
        wall = WallParams.uniform(dim, T_WALL, acc=acc)
        out = apply_faces(
            x,
            v,
            jnp.ones(n, bool),
            g,
            faces,
            wall,
            jnp.float64(MASS),
            jnp.float64(dt),
            ka,
            acc_delta=delta,
        )
        return jnp.mean(jnp.sum(out[1] ** 2, axis=-1))

    hard = float(jax.grad(energy)(0.6, 0.0))
    soft = float(jax.grad(energy)(0.6, 0.1))
    assert hard == 0.0, f"a hard threshold must have no pathwise derivative, got {hard}"
    assert abs(soft) > 0.0, "the relaxed threshold must carry one"

    # and it must be the *correct* pathwise derivative: fixed-key FD is the strong test
    h = 1e-6
    fd = (energy(0.6 + h, 0.1) - energy(0.6 - h, 0.1)) / (2 * h)
    np.testing.assert_allclose(soft, float(fd), rtol=1e-6)
