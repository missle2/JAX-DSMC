"""Initial condition assembly (spec sections 5 and 12).

This is the one module that owns the ``STAGE_INIT`` stream.  Everything
stochastic here is *reparameterised* (section 5): the random draw is
parameter-independent and the parameters enter through a differentiable map, so
``d/dtemp`` and ``d/dvstream`` are pathwise-exact rather than score-based.  [P3]
states the requirement directly -- initial particles are generated as
``(x_0i, v_0i) = Psi(xi_i; m)`` with the distribution of ``xi_i`` independent of
``m`` -- and it is what makes the ``alpha`` half of the coupled adjoint reduce to
a VJP through the sampler.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_dsmc.collide.ntc import vremax_init
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import cell_volume, num_cells
from jax_dsmc.rng import STAGE_INIT, key_for, key_for_slot
from jax_dsmc.sampling import sample_maxwellian, sample_positions
from jax_dsmc.state import (
    CollideState,
    FlowParams,
    GridSpec,
    ParticleState,
    SimStatic,
    SpeciesParams,
    check_fnum_uniform,
)

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


def target_particle_count(flow: FlowParams, grid: GridSpec, fnum) -> jnp.ndarray:
    """Physical particle count the domain implies, ``f64`` scalar.

    ``N_real = nrho * V_domain`` and ``N_sim = N_real / fnum``.  Returned as a
    float on purpose: it is a *diagnostic* for choosing the static capacity ``N``,
    not something the code may round and act on, since ``N`` is a compile-time
    shape (section 3.2).
    """
    return flow.nrho * jnp.sum(cell_volume(grid)) / fnum[0]


def initial_state(
    base_key: jax.Array,
    flow: FlowParams,
    grid: GridSpec,
    sp: SpeciesParams,
    static: SimStatic,
    n_alive: int | None = None,
    vstream_fn=None,
) -> ParticleState:
    """Uniform-in-space, Maxwellian-in-velocity initial particles.

    Args:
        base_key: the run's fixed ``PRNGKey``.
        flow: ``FlowParams``; ``temp`` and ``vstream`` TRACED.
        grid: ``GridSpec``.
        sp: ``SpeciesParams`` -- ``mass[0]`` sets the thermal speed.
        static: ``SimStatic``; ``N`` is the array capacity.
        n_alive: how many of the ``N`` slots start alive.  STATIC.  Defaults to
            all of them.  Fewer is the normal case once inflow exists (M5): the
            capacity is sized for the peak, not the start.
        vstream_fn: optional ``x -> f64[N, 3]`` giving a **position-dependent**
            bulk velocity, used instead of the uniform ``flow.vstream``.  This is
            what sets up a non-equilibrium initial condition such as the
            sinusoidal shear that steepens into a shock (M2, ``examples/burgers``).
            It is a STATIC Python callable, and it is applied to the sampled
            positions, so gradients flow through it exactly like ``flow.vstream``
            does -- the reparameterisation is untouched.

    Returns:
        ``ParticleState`` with ``x f64[N, D]``, ``v f64[N, 3]``.

    Velocities are **always 3-component** whatever ``D`` is (section 0.2), and
    ``logh_acc`` starts at zero -- the score accumulator is per-run, not per-step.

    Dead slots are given valid-but-inert data (position at ``lo``, zero velocity)
    rather than ``NaN``: they are still gathered by the collision scan before being
    masked, and a ``NaN`` there would poison the cotangent of live particles
    through the double-``where`` pattern even though the primal is masked out.
    """
    n_alive = static.N if n_alive is None else n_alive
    if not 0 <= n_alive <= static.N:
        raise ValueError(f"n_alive must be in [0, N={static.N}]; got {n_alive}")

    k_init = key_for(base_key, 0, STAGE_INIT)
    x = sample_positions(key_for_slot(k_init, 0), static.N, grid)
    vstream = flow.vstream if vstream_fn is None else vstream_fn(x)
    v = sample_maxwellian(key_for_slot(k_init, 1), static.N, flow.temp, sp.mass[0], vstream)

    alive = jnp.arange(static.N) < n_alive
    x = jnp.where(alive[:, None], x, grid.lo)
    v = jnp.where(alive[:, None], v, 0.0)

    return ParticleState(
        x=x,
        v=v,
        species=jnp.zeros((static.N,), dtype=jnp.int32),
        alive=alive,
        logh_acc=jnp.zeros((static.N,), dtype=jnp.float64),
    )


def initial_collide_state(
    flow: FlowParams, grid: GridSpec, sp: SpeciesParams, static: SimStatic
) -> CollideState:
    """Seed the per-cell majorant.

    Uses the a-priori thermal majorant of section 7.2.3, **not** SPARTA's
    ``vremax = 2 * cxs * vscale`` (``collide_vss.cpp:107-133``).  That seed fails
    the U3 gate: with the stale-max rule it permits a ``1.1e-3`` violation rate in
    equilibrium and ``1.0e-2`` under counter-streaming beams, and every violated
    attempt contributes a zero score and biases the gradient (section 7.2.1).
    """
    C = num_cells(grid)
    temp = jnp.full((C,), flow.temp)
    return CollideState(
        vremax=vremax_init(
            temp, sp.omega[0, 0], sp.tref[0, 0], sp.mr[0, 0], sp.prefactor[0, 0], C, static.G
        )
    )


def suggested_a_max(
    flow: FlowParams,
    grid: GridSpec,
    sp: SpeciesParams,
    static: SimStatic,
    dt,
    fnum,
    inflation: float = 2.5,
) -> int:
    """A starting value for the static ``A_max``, as a Python int.

    ``A_max`` is a **correctness** knob, not a performance one: if ``max(n)``
    exceeds it the surplus attempts are silently dropped and the gas under-collides
    with no error and no ``NaN`` (section 8.2).

    The ``inflation`` factor exists because ``vremax`` never decays (section 16,
    open item 2), so a thermal transient permanently raises it -- measured at
    ``2.29x`` on a ``T: 1 -> 4 -> 1`` pulse.  An ``A_max`` sized at cold start is
    then exceeded for the rest of the run.  Size from the inflated majorant, and
    keep ``attempts_clipped`` asserted zero (U4).

    This is advisory only -- it returns a number for the caller to pass to
    ``SimStatic``; it cannot size anything itself, because ``A_max`` is a
    compile-time trip count.
    """
    check_fnum_uniform(fnum)
    C = num_cells(grid)
    npc = float(target_particle_count(flow, grid, fnum)) / C
    sigma = float(
        vremax_init(
            jnp.full((1,), flow.temp),
            sp.omega[0, 0],
            sp.tref[0, 0],
            sp.mr[0, 0],
            sp.prefactor[0, 0],
            1,
            1,
        )[0, 0, 0]
    )
    vol = float(cell_volume(grid)[0])
    f = 0.5 * npc * (npc - 1.0) * sigma * float(dt) * float(fnum[0]) / vol
    return max(1, int(jnp.ceil(inflation * f)))


def thermal_speed_of(flow: FlowParams, sp: SpeciesParams):
    """``sqrt(k T / m)`` for the initial condition -- handy for timestep sizing."""
    return jnp.sqrt(K_BOLTZ * flow.temp / sp.mass[0])
