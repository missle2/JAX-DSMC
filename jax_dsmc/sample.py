"""Per-cell macroscopic moments (spec section 10).

Every reduction is a :func:`jax.ops.segment_sum` with a **static**
``num_segments = C``.  Static because ``C = prod(grid.n)`` is a compile-time
constant (section 3.2) and because a data-dependent segment count would make
every downstream array shape dynamic.

**Two passes, deliberately.**  The single-pass identity
``<|c|^2> = <|v|^2> - |<v>|^2`` is catastrophically ill-conditioned in a
near-equilibrium cell: there the bulk speed is small against the thermal speed,
so the two terms agree to many digits and their difference keeps only the
digits they disagree in.  The second pass subtracts the measured bulk velocity
*inside* the sum instead, which is unconditionally stable.  It costs one extra
gather of ``u[c]`` and one extra reduction.

Differentiability.  This module is where objective functionals ``r(x, v)`` are
built, so it is the entry point of the whole adjoint (section 11): the moments
are differentiable in ``v`` directly, in ``x`` through nothing (``cell_of``
stop-gradients its argument — a frozen decision, section 0.5), and in the mesh
geometry through ``cell_volume``.  Empty cells go through
:func:`jax_dsmc.safe.safe_recip_count`, which returns ``0`` rather than ``inf``
and has a zero — not ``NaN`` — derivative there.

Time averaging follows SPARTA's ``fix ave/grid``: accumulate *sums* over the
sampling window and divide once at the end.  See :class:`MomentAccumulator`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import cell_of, cell_volume, num_cells
from jax_dsmc.safe import safe_div, safe_recip_count
from jax_dsmc.state import CellTable, GridSpec, ParticleState, SimParams, SpeciesParams

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


@register_dataclass
@dataclass(frozen=True)
class CellMoments:
    """Macroscopic moments of one cell population.  All fields TRACED.

    Fields:
        nrho: ``f64[C]`` number density ``count * fnum / V`` [1/m^3].
        u: ``f64[C, 3]`` bulk velocity [m/s].  Three components whatever ``D``
            is (section 0.2).
        temp: ``f64[C]`` translational temperature [K].
        P: ``f64[C, 3, 3]`` pressure tensor ``rho <c_a c_b>`` [Pa].
        q: ``f64[C, 3]`` heat flux ``(rho/2) <c_a |c|^2>`` [W/m^2].
        count: ``f64[C]`` number of simulator particles (a float, so it can be
            averaged; it is a frozen integer quantity and carries no tangent).

    ``c`` is the peculiar velocity ``v - u``.  Empty cells are exactly zero in
    every field, never ``NaN``.
    """

    nrho: jax.Array
    u: jax.Array
    temp: jax.Array
    P: jax.Array
    q: jax.Array
    count: jax.Array


@register_dataclass
@dataclass(frozen=True)
class MomentAccumulator:
    """Running *sums* over a sampling window (SPARTA ``fix ave/grid``).

    Sums, not averages: averaging incrementally would either need a division
    per step or a running-mean update, and both lose the exactness of "divide
    once at the end".  ``n_samples`` counts the steps folded in.

    Fields mirror :class:`CellMoments` and hold the same units multiplied by
    ``n_samples``.
    """

    nrho: jax.Array
    u: jax.Array
    temp: jax.Array
    P: jax.Array
    q: jax.Array
    count: jax.Array
    n_samples: jax.Array


def moments(
    state: ParticleState,
    table: CellTable,
    grid: GridSpec,
    params: SimParams,
    species: SpeciesParams,
) -> CellMoments:
    """Per-cell moments of the live particles.  Spec section 10.

    Args:
        state: ``ParticleState``; ``x`` is ``f64[N, D]`` and ``v`` is
            ``f64[N, 3]``, both TRACED; ``alive`` is ``bool[N]``.
        table: ``CellTable``.  Only ``table.cell`` is read, and only as a
            cross-check that it agrees with ``cell_of`` is *not* performed —
            the cell index is recomputed here through the mesh interface so
            that this module holds no cell-index arithmetic of its own
            (section 4).  ``table`` is accepted for signature compatibility and
            because M5's cut cells will need the occupancy structure.
        grid: ``GridSpec``; ``lo``/``hi`` TRACED, ``n`` STATIC.
        params: ``SimParams``; ``fnum`` is ``f64[C]``, **per cell** — not a
            scalar (section 3.3).
        species: ``SpeciesParams``.  ``mass[0]`` is used; ``S == 1`` for now.

    Returns:
        :class:`CellMoments`, every field ``f64`` with leading dimension ``C``.

    Reference: SPARTA ``compute_grid.cpp`` (``nrho``, ``u``, ``temp``) and
    ``compute_thermal_grid.cpp`` / ``compute_sonine_grid.cpp`` (``P``, ``q``).
    """
    C = num_cells(grid)
    c = cell_of(state.x, grid)  # i32[N], frozen decision
    w = state.alive.astype(jnp.float64)  # f64[N]

    cnt = jax.ops.segment_sum(w, c, num_segments=C)  # f64[C]
    inv = safe_recip_count(cnt)  # 0 where empty
    vol = cell_volume(grid)  # f64[C], TRACED

    nrho = safe_div(cnt * params.fnum, vol)

    # --- pass 1: bulk velocity ------------------------------------------
    u = jax.ops.segment_sum(w[:, None] * state.v, c, num_segments=C) * inv[:, None]

    # --- pass 2: central moments about the measured u --------------------
    # Never <v^2> - <v>^2: see the module docstring.
    cpec = state.v - u[c]  # f64[N, 3] peculiar velocity
    c2 = jnp.sum(cpec * cpec, axis=-1)  # f64[N]

    e = jax.ops.segment_sum(w * c2, c, num_segments=C) * inv  # <|c|^2>
    mass = species.mass[0]
    temp = mass * e / (3.0 * K_BOLTZ)

    rho = mass * nrho  # mass density
    P = rho[:, None, None] * (
        jax.ops.segment_sum(
            w[:, None, None] * cpec[:, :, None] * cpec[:, None, :], c, num_segments=C
        )
        * inv[:, None, None]
    )
    q = (
        0.5
        * rho[:, None]
        * (jax.ops.segment_sum(w[:, None] * cpec * c2[:, None], c, num_segments=C) * inv[:, None])
    )

    return CellMoments(nrho=nrho, u=u, temp=temp, P=P, q=q, count=cnt)


def cell_temperature(
    state: ParticleState,
    table: CellTable,
    grid: GridSpec,
    species: SpeciesParams,
) -> jax.Array:
    """Per-cell translational temperature only, ``f64[C]``.

    :func:`moments` computes the pressure tensor and heat flux as well, and the
    pressure tensor materialises an ``(N, 3, 3)`` intermediate -- nine f64 per
    particle.  The time loop calls this every step purely to feed the a-priori
    majorant of section 7.2.3, which needs nothing but ``temp``, so paying for
    ``P`` and ``q`` on every step is pure waste.  Measured 2.9-3.5x faster than
    the full :func:`moments` at production sizes.

    It also **reuses** ``table.cell`` rather than recomputing ``cell_of``.
    ``sort.build_table`` has already computed exactly this array and returns it
    for precisely this reason (section 6); recomputing it measured a further
    1.6-3.0x.  That does not weaken the section-4 mesh rule -- no cell-index
    arithmetic appears here, the index is simply taken from the module that owns
    it instead of asking for it twice.

    Uses the same two-pass central-moment form as :func:`moments`: the peculiar
    velocity is taken about the *measured* bulk velocity, never as
    ``<v^2> - <v>^2``, which cancels catastrophically at high Mach number.
    """
    C = num_cells(grid)
    c = table.cell  # i32[N], already computed by build_table
    w = state.alive.astype(jnp.float64)
    cnt = jax.ops.segment_sum(w, c, num_segments=C)
    inv = safe_recip_count(cnt)
    u = jax.ops.segment_sum(w[:, None] * state.v, c, num_segments=C) * inv[:, None]
    cpec = state.v - u[c]
    e = jax.ops.segment_sum(w * jnp.sum(cpec * cpec, axis=-1), c, num_segments=C) * inv
    return species.mass[0] * e / (3.0 * K_BOLTZ)


def energy_spectrum_1d(u: jax.Array, component: int = 0) -> jax.Array:
    """One-sided kinetic-energy spectrum of a 1-D periodic velocity field.

    Args:
        u: ``f64[C]`` or ``f64[C, 3]`` per-cell bulk velocity on a uniform
            periodic 1-D grid of ``C`` cells.  TRACED.
        component: which velocity component to transform when ``u`` is 2-D.

    Returns:
        ``f64[C // 2]`` energy at wavenumbers ``k = 1 .. C//2`` (the ``k = 0``
        mean is dropped, since a uniform drift carries no fluctuation energy).

    Normalised so that ``sum(E) == mean((u - <u>)**2) / 2``, i.e. Parseval holds
    against the specific fluctuation kinetic energy.  The factor of two for the
    conjugate pair ``+k, -k`` is folded in.

    **What to expect, and what will mislead you.** A 1-D compressible flow steepens
    into shocks; a field with jump discontinuities has Fourier coefficients decaying
    as ``1/k``, so ``E(k) ~ k^-2``.  That is the Burgers result and it is what this
    measures.  But a DSMC velocity field also carries *statistical* noise of size
    ``v_th / sqrt(npc)`` per cell, and that noise is **white** -- flat in ``k``.  So
    the measured spectrum is a ``k^-2`` range that rolls over onto a flat noise
    floor, and the floor is a property of the sampling, not of the physics.  The
    crossover sits near ``k`` where ``E_signal(k)`` falls to ``v_th^2/(npc * C)``;
    pushing it out means more particles per cell or more seeds, not a finer grid.
    Reporting a spectral slope without showing where that floor is would be
    meaningless.
    """
    if u.ndim == 2:
        u = u[:, component]
    C = u.shape[0]
    uk = jnp.fft.rfft(u - jnp.mean(u)) / C
    e = jnp.abs(uk) ** 2  # |u_k|^2; the 1/2 of kinetic energy cancels the +k/-k pair
    # ... except at Nyquist, which is its own conjugate and must not be doubled.
    e = e.at[C // 2].set(0.5 * e[C // 2]) if C % 2 == 0 else e
    return e[1 : C // 2 + 1]


def spectrum_bins(n: tuple[int, ...]) -> tuple[jax.Array, int]:
    """Shell index of every Fourier mode of an ``n``-shaped grid, and the count.

    ``n`` is STATIC, so this is pure host-side setup: the shell each mode falls in
    never depends on a traced value, which is what lets the binning be a
    ``segment_sum`` rather than a data-dependent scatter.

    Shell ``j`` collects the modes with ``round(|k|) == j`` where ``k`` is measured
    in units of the box wavenumber ``2 pi / L`` -- i.e. integer mode numbers, the
    convention in which a periodic box's lowest mode is ``k = 1``.
    """
    D = len(n)
    ks = jnp.meshgrid(*[jnp.fft.fftfreq(nd) * nd for nd in n], indexing="ij")
    kmag = jnp.sqrt(sum(k**2 for k in ks))
    idx = jnp.round(kmag).astype(jnp.int32)
    nb = int(min(n) // 2) + 1
    # Modes beyond the smallest Nyquist are in the corners of the cube, where the
    # shell is only partly sampled; folding them into the last bin would put a
    # spurious upturn there, so they go to a discard bin instead.
    idx = jnp.where(idx < nb, idx, nb)
    del D
    return idx.reshape(-1), nb


def energy_spectrum(u: jax.Array, n: tuple[int, ...]) -> jax.Array:
    """Shell-averaged kinetic-energy spectrum ``E(k)`` for ``D`` in {1, 2, 3}.

    Args:
        u: ``f64[C, 3]`` per-cell bulk velocity on the uniform periodic grid,
            raveled row-major to match ``grid.cell_of``.  TRACED.
        n: the grid shape.  STATIC.

    Returns:
        ``f64[min(n)//2 + 1]`` indexed by integer wavenumber, with ``E[0]`` the
        (dropped) mean and ``sum(E) == 0.5 * mean(|u - <u>|^2)`` by Parseval.

    All three velocity components are included whatever ``D`` is, because a DSMC
    velocity field is genuinely three-component even in a 1-D box (section 0.2) and
    the transverse components carry real energy.

    The same statistical-noise floor described in :func:`energy_spectrum_1d`
    applies, and it is worse here: shell ``k`` in ``D`` dimensions averages
    ``~k^(D-1)`` modes, so the noise floor *falls* with ``k`` as the shell count
    grows, which can be mistaken for a physical decay.  Plot the floor.
    """
    C = u.shape[0]
    field = u.reshape(*n, 3) - jnp.mean(u, axis=0)
    axes = tuple(range(len(n)))
    uk = jnp.fft.fftn(field, axes=axes) / C
    e = 0.5 * jnp.sum(jnp.abs(uk) ** 2, axis=-1).reshape(-1)
    idx, nb = spectrum_bins(n)
    return jax.ops.segment_sum(e, idx, num_segments=nb + 1)[:nb]


def zero_accumulator(grid: GridSpec) -> MomentAccumulator:
    """An empty accumulator sized for ``grid``.  Host-side setup."""
    C = num_cells(grid)
    return MomentAccumulator(
        nrho=jnp.zeros((C,), dtype=jnp.float64),
        u=jnp.zeros((C, 3), dtype=jnp.float64),
        temp=jnp.zeros((C,), dtype=jnp.float64),
        P=jnp.zeros((C, 3, 3), dtype=jnp.float64),
        q=jnp.zeros((C, 3), dtype=jnp.float64),
        count=jnp.zeros((C,), dtype=jnp.float64),
        n_samples=jnp.asarray(0.0, dtype=jnp.float64),
    )


def accumulate(acc: MomentAccumulator, m: CellMoments) -> MomentAccumulator:
    """Fold one sample into the running sums.  Pure; safe inside ``lax.scan``."""
    return MomentAccumulator(
        nrho=acc.nrho + m.nrho,
        u=acc.u + m.u,
        temp=acc.temp + m.temp,
        P=acc.P + m.P,
        q=acc.q + m.q,
        count=acc.count + m.count,
        n_samples=acc.n_samples + 1.0,
    )


def finalize(acc: MomentAccumulator) -> CellMoments:
    """Divide the accumulated sums by the sample count, once.

    Returns zeros — not ``NaN`` — if nothing was ever accumulated.
    """
    inv = jnp.where(acc.n_samples > 0.0, 1.0 / jnp.maximum(acc.n_samples, 1.0), 0.0)
    return CellMoments(
        nrho=acc.nrho * inv,
        u=acc.u * inv,
        temp=acc.temp * inv,
        P=acc.P * inv,
        q=acc.q * inv,
        count=acc.count * inv,
    )


def pressure_scalar(m: CellMoments) -> jax.Array:
    """``trace(P)/3``, ``f64[C]`` [Pa].

    **Not a physics check.**  Spec section 10.1: ``temp`` is *defined* from the
    same ``<|c|^2>`` that forms ``trace(P)``, so ``trace(P)/3 == nrho k T`` is
    an identity by construction.  It is worth asserting as a typo/refactor
    regression guard and worth nothing as validation; the real test of ``P`` is
    against the analytic Chapman-Enskog stress at a known shear rate (M4).
    """
    return jnp.trace(m.P, axis1=-2, axis2=-1) / 3.0
