"""Independent regression cases from the forward review."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_dsmc import inflow
from jax_dsmc.boundary import BC, Faces
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.state import GridSpec, ParticleState


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_face_area_uses_physical_transverse_measure(dim):
    grid = GridSpec(
        lo=jnp.zeros(dim),
        hi=jnp.arange(2, 2 + dim, dtype=float),
        n=(1,) * dim,
        transverse=jnp.float64(0.01 if dim < 3 else 1),
    )
    for axis in range(dim):
        expected = float(grid.transverse) * np.prod(np.delete(np.asarray(grid.hi), axis))
        assert float(inflow.face_area(grid, axis)) == pytest.approx(expected)


def test_two_faces_allocate_actual_births_with_capacity_larger_than_storage():
    n = 40
    grid = GridSpec(lo=jnp.zeros(1), hi=jnp.ones(1), n=(1,))
    state = ParticleState(
        x=jnp.zeros((n, 1)),
        v=jnp.zeros((n, 3)),
        species=jnp.zeros(n, jnp.int32),
        alive=jnp.zeros(n, bool),
        logh_acc=jnp.zeros(n),
    )
    # Unit normal thermal variance, mean flux 1/sqrt(2 pi); exactly ten births/face.
    res = inflow.InflowParams.uniform(1, 10 * np.sqrt(2 * np.pi), 1.0)
    faces = Faces((BC.INFLOW, BC.INFLOW))
    out, inserted, shortfall = inflow.inject(
        state, grid, faces, res, K_BOLTZ, 1.0, 1.0, jax.random.key(3), capacity=60
    )
    assert int(inserted) == 20
    assert int(out.alive.sum()) == 20
    assert float(shortfall) == 0


@pytest.mark.parametrize("s", [-10.0, -3.0, 0.0, 3.0, 10.0])
def test_flux_quantile_and_implicit_derivative_against_quadrature(s):
    from scipy.integrate import quad  # noqa: PLC0415 -- optional reference dependency
    from scipy.optimize import brentq  # noqa: PLC0415 -- optional reference dependency

    u = 0.37

    def cdf(y, drift):
        shift = min(drift, 0.0) ** 2

        def f(z):
            return z * np.exp(-((z - drift) ** 2) + shift)

        return quad(f, 0, y, epsabs=1e-13)[0] / quad(f, 0, max(drift, 0) + 12, epsabs=1e-13)[0]

    def root(drift):
        return brentq(lambda y: cdf(y, drift) - u, 0, max(drift, 0) + 12, xtol=1e-13)

    got = inflow.flux_quantile(jnp.float64(u), jnp.float64(s))
    derivative = jax.grad(lambda drift: inflow.flux_quantile(jnp.float64(u), drift))(jnp.float64(s))
    expected = (root(s + 1e-4) - root(s - 1e-4)) / 2e-4
    np.testing.assert_allclose(got, root(s), rtol=2e-10, atol=1e-11)
    np.testing.assert_allclose(derivative, expected, rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize("s", [-10.0, -3.0, 0.0, 3.0, 10.0])
def test_pathwise_flux_moment_and_gradient_against_independent_quadrature(s):
    """Averaged inlet observables need the correct tangent, not just one quantile."""
    from scipy.integrate import quad  # noqa: PLC0415 -- optional reference dependency

    nodes, weights = np.polynomial.legendre.leggauss(256)
    uniform = jnp.asarray((nodes + 1) / 2)
    weights = jnp.asarray(weights / 2)

    def pathwise_mean(drift):
        speed = inflow.flux_quantile(uniform, drift)
        return jnp.dot(weights, speed**2)

    def reference_mean(drift):
        shift = min(drift, 0.0) ** 2

        def weighted_speed(y, power):
            return y ** (power + 1) * np.exp(-((y - drift) ** 2) + shift)

        numerator = quad(lambda y: weighted_speed(y, 2), 0, np.inf, epsabs=1e-12)[0]
        denominator = quad(lambda y: weighted_speed(y, 0), 0, np.inf, epsabs=1e-12)[0]
        return numerator / denominator

    actual = float(pathwise_mean(jnp.float64(s)))
    tangent = float(jax.grad(pathwise_mean)(jnp.float64(s)))
    reference = reference_mean(s)
    derivative = (reference_mean(s + 1e-4) - reference_mean(s - 1e-4)) / 2e-4
    np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(tangent, derivative, rtol=3e-5, atol=3e-5)


def _bisect_step(_, bounds, u, s):
    lo, hi = bounds
    mid = (lo + hi) / 2
    left = inflow.flux_cdf(mid, s) < u
    return jnp.where(left, mid, lo), jnp.where(left, hi, mid)


@jax.jit
def _bisection_quantile(u, s):
    """The 64-step bisection ``flux_quantile`` used to be: the reference the Halley solve must
    reproduce, since both invert the same computed :func:`inflow.flux_cdf`."""
    u, s = jnp.broadcast_arrays(u, s)
    body = functools.partial(_bisect_step, u=u, s=s)
    lo, hi = jax.lax.fori_loop(0, 64, body, (jnp.zeros_like(u), jnp.maximum(s, 0.0) + 12.0))
    return (lo + hi) / 2


def test_flux_quantile_reproduces_the_bisection_reference():
    """Same root as the old bisection, to the round-off of ``flux_cdf`` itself.

    That CDF loses digits to cancellation at strongly outward drift (``s = -10`` costs ~3)
    and is quantised near ``u = 1``, so the pin is tight where it is well conditioned and
    residual-only in the tails, where neither solver's root is better than the CDF's noise.
    """
    eps = np.finfo(np.float64).eps
    u = np.concatenate(
        [
            [eps, 1e-15],
            np.logspace(-14, -1, 27),
            np.linspace(0.05, 0.95, 19),
            1 - np.logspace(-2, -15, 27),
            [1 - eps],
        ]
    )
    s = np.linspace(-10.0, 10.0, 81)
    U, S = (a.ravel() for a in np.meshgrid(u, s))
    got = np.asarray(jax.jit(inflow.flux_quantile)(U, S))
    ref = np.asarray(_bisection_quantile(U, S))
    assert np.isfinite(got).all()
    residual = np.abs(np.asarray(inflow.flux_cdf(got, S)) - U)
    assert residual.max() < 1e-10  # the sampler's acceptance test holds everywhere
    rel = np.abs(got / ref - 1)
    mid = (U >= 1e-3) & (U <= 1 - 1e-3)
    tails = (U >= 1e-6) & (U <= 1 - 1e-6)
    # measured: 1.3e-11, 2.0e-8, 2.0e-9 -- the last is s = -10's cancellation in flux_cdf
    assert rel[mid & (np.abs(S) <= 3)].max() < 1e-10
    assert rel[tails & (np.abs(S) <= 3)].max() < 1e-7
    assert rel[mid].max() < 1e-8


@pytest.mark.parametrize(
    ("s", "rtol"),
    [
        (-10.0, 3e-4),
        (-3.0, 1e-6),
        (-0.5, 1e-8),
        (0.0, 1e-8),
        (0.5, 1e-8),
        (3.0, 1e-8),
        (10.0, 1e-8),
    ],
)
def test_flux_quantile_tangent_is_the_implicit_derivative(s, rtol):
    """The custom JVP against central differences of the Halley primal itself.

    Outward drift loosens ``rtol`` only because the differences see ``flux_cdf``'s
    cancellation error (~1e-9 relative at ``s = -10``) divided by the step.
    """
    u = jnp.linspace(0.02, 0.98, 25)
    drift = jnp.full_like(u, s)
    quantile = jax.jit(inflow.flux_quantile)
    _, d_drift = jax.jvp(lambda d: inflow.flux_quantile(u, d), (drift,), (jnp.ones_like(u),))
    _, d_u = jax.jvp(lambda v: inflow.flux_quantile(v, drift), (u,), (jnp.ones_like(u),))
    h_s, h_u = 1e-5, 1e-6
    fd_drift = (quantile(u, drift + h_s) - quantile(u, drift - h_s)) / (2 * h_s)
    fd_u = (quantile(u + h_u, drift) - quantile(u - h_u, drift)) / (2 * h_u)
    np.testing.assert_allclose(d_drift, fd_drift, rtol=rtol)
    np.testing.assert_allclose(d_u, fd_u, rtol=rtol)


def test_inlet_candidate_streams_are_capacity_independent():
    res = inflow.InflowParams.uniform(1, 1.0, 1.0, [2.0, 0.0, 0.0])
    key = jax.random.key(4)
    small, ok = inflow.sample_face_velocities(key, 8, res, 0, 0, K_BOLTZ)
    large, ok2 = inflow.sample_face_velocities(key, 32, res, 0, 0, K_BOLTZ)
    np.testing.assert_array_equal(small, large[:8])
    np.testing.assert_array_equal(ok, ok2[:8])


def test_source_allocation_diagnostics_close_requested_birth_identity():
    n = 8
    grid = GridSpec(jnp.zeros(1), jnp.ones(1), (1,))
    state = ParticleState(
        jnp.zeros((n, 1)),
        jnp.zeros((n, 3)),
        jnp.zeros(n, jnp.int32),
        jnp.zeros(n, bool),
        jnp.zeros(n),
    )
    res = inflow.InflowParams.uniform(1, 10 * np.sqrt(2 * np.pi), 1.0)
    _, inserted, shortfall, diag = inflow.inject(
        state,
        grid,
        Faces.uniform(BC.INFLOW, 1),
        res,
        K_BOLTZ,
        1.0,
        1.0,
        jax.random.key(2),
        6,
        return_diagnostics=True,
    )
    assert int(diag.requested) == 20
    assert int(diag.inserted) == int(inserted) == 8
    assert int(diag.candidate_overflow) == 8
    assert int(diag.storage_overflow) == 4
    assert int(diag.requested) == int(
        diag.inserted
        + diag.escaped
        + diag.sampler_failed
        + diag.candidate_overflow
        + diag.storage_overflow
    )
    assert float(shortfall) == 12
