"""State and parameter containers.

Spec: PSEUDOCODE.md section 3.

Every container here is a **registered pytree** so that ``jax.grad`` traverses it
without help.  The load-bearing property of this module is the traced/static
split of section 0.4:

- **TRACED** fields are differentiable pytree *leaves*.  They may carry a
  tangent/cotangent and must never be branched on with a Python ``if``.
- **STATIC** fields determine array shapes or Python control flow.  They are
  declared with ``field(metadata={"static": True})``, are compared by value for
  the jit cache, and changing one forces a recompilation.

A value may not be both.  The two recurring mistakes are putting ``n`` (cells per
axis) in a traced field and putting ``dt`` in a static one.

Three shape facts that are easy to get wrong:

- **Velocities are always 3-component** regardless of the spatial dimension
  ``D`` (section 0.2).  ``x`` is ``f64[N, D]``; ``v`` is ``f64[N, 3]``.
- **``fnum`` is per-cell, ``f64[C]``, not a scalar** (section 3.3).  It is a
  traced leaf, so ``dJ/d(fnum_c)`` exists.  The loaded-gun invariant that comes
  with shipping the field without the clone/delete machinery is checked by
  :func:`check_fnum_uniform`.
- **``CollideState`` has no ``remain`` field.**  Section 8.1 rounds the attempt
  count stochastically (``n = floor(f + U)``, SPARTA's default path), which needs
  no carry and yields a genuine ``log P(n | f)`` score term.  A deterministic
  ``remain`` carry has a degenerate ``P(n | f, remain)``, no score, and therefore
  exactly zero gradient to ``dt``, ``fnum`` and the cell volume.

Symbols follow the papers: ``N`` particle capacity, ``D`` spatial dimension,
``S`` species, ``G`` collision groups, ``C = prod(n)`` cells, ``A_max`` attempt
slots, ``K_bounce`` wall bounces.

All arrays are ``f64``/``i32``/``bool`` exactly as annotated; the dtypes are a
contract, not a suggestion (section 0.3 — the VSS energy test asserts ~1e-14).
Constructors deliberately do **not** coerce dtypes: the containers are
unflattened with tracers, and with sentinel objects during some JAX transforms,
so any ``asarray`` in ``__post_init__`` would break tracing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout

__all__ = [
    "CellTable",
    "CollideState",
    "Diagnostics",
    "FlowParams",
    "GridSpec",
    "ParticleState",
    "SimParams",
    "SimStatic",
    "SpeciesParams",
    "check_fnum_uniform",
    "num_cells_of",
    "static_field",
    "static_fields",
    "traced_fields",
]


def static_field(**kwargs):
    """A dataclass field marked STATIC for :func:`jax.tree_util.register_dataclass`.

    Static fields become pytree *metadata*: they are hashed and compared by value
    to key the jit cache, never traced, and must be immutable (a Python int, str
    or tuple — never a ``jax.Array``).

    Spec: section 0.4.
    """
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["static"] = True
    return field(metadata=metadata, **kwargs)


# --- state ----------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True)
class ParticleState:
    """The particle ensemble.  All fields TRACED leaves.

    Fields:
        x: ``f64[N, D]`` positions.
        v: ``f64[N, 3]`` velocities — **always 3 components**, whatever ``D`` is
            (section 0.2).  Advection uses ``v[:, :D]``.
        species: ``i32[N]`` species index; all zeros while ``S == 1``.  An
            integer, so it carries no tangent (a frozen decision, section 0.5).
        alive: ``bool[N]`` occupancy mask.  Dead slots are reusable — that is how
            outflow works today and how M5 inflow will work.
        logh_acc: ``f64[N]`` score accumulator ``L_i`` of section 11.  Must be
            zeroed when a dead slot is refilled (gap U6).

    ``N`` and ``D`` are STATIC and live in :class:`SimStatic`; they are shapes
    here, not fields.

    Spec: section 3.1.  Reference: SPARTA ``particle.h`` ``Particle::OnePart``.
    """

    x: jax.Array
    v: jax.Array
    species: jax.Array
    alive: jax.Array
    logh_acc: jax.Array


@register_dataclass
@dataclass(frozen=True)
class CollideState:
    """NTC majorant state carried across steps.  TRACED leaf.

    Fields:
        vremax: ``f64[C, G, G]`` majorant ``Sigma`` per cell per group pair.

    ``vremax`` is **read-only within a step**: it is captured before the attempt
    loop and the new maximum is published only for the next step.  We follow
    [DF] ``noTimeCounter.C:231-246`` rather than SPARTA's live ratchet
    (``collide_vss.cpp:225-231``), because a loop-invariant ``Sigma`` is what
    [P2]'s virtual-collision unbiasedness argument assumes and what makes the
    ``log(q/Sigma)`` score term well defined (sections 7.2, 8.4, 18.1).

    ``remainder`` stores the fractional proposal carry in remainder mode. It is
    initialized to zeros for compatibility with old constructors and persists
    across chunks. Remainder expectation gradients are not implemented.

    Spec: section 3.1.
    """

    vremax: jax.Array
    remainder: jax.Array = None

    def __post_init__(self):
        if self.remainder is None and hasattr(self.vremax, "shape"):
            object.__setattr__(self, "remainder", jnp.zeros_like(self.vremax))


@register_dataclass
@dataclass(frozen=True)
class CellTable:
    """Cell occupancy table, rebuilt once per step by ``sort.build_table``.

    Fields:
        order: ``i32[N]`` particle indices sorted by cell.
        starts: ``i32[C]`` exclusive prefix sum of ``counts``.
        counts: ``i32[C]`` particles per cell.
        cell: ``i32[N]`` per-particle cell index.

    Particle ``j`` of cell ``k`` is ``order[starts[k] + j]``, valid for
    ``j < counts[k]``.

    All four are integer arrays, hence **frozen decisions** (section 0.5) that
    carry no tangent.  They are still pytree *data* fields, not static ones:
    their values are traced arrays computed inside jit, and only their shapes
    (fixed by ``N`` and ``C``) are compile-time constants.

    ``cell`` is carried rather than discarded because ``sort.build_table``
    computes it anyway and the **count score** of section 8.1 needs it — that
    score is a per-cell term applied to every particle of the cell, so it must
    map particle -> cell without recomputing ``cell_of`` inside ``collide``
    (section 6).

    Spec: sections 3.1 and 6.
    """

    order: jax.Array
    starts: jax.Array
    counts: jax.Array
    cell: jax.Array


@register_dataclass
@dataclass(frozen=True)
class Diagnostics:
    """Per-step silent-failure counters.  TRACED leaves, host-side consumers only.

    Fields:
        attempts_clipped: ``i32`` scalar, number of cells where ``n > A_max`` and
            attempts were silently dropped (section 8.2, gap U4).
        max_attempts: ``i32`` scalar, ``max(n)`` over cells — used to size
            ``A_max``.
        majorant_violations: ``i32`` scalar, number of candidates with
            ``q > Sigma``.  Each is a **zero score and a biased gradient**
            (section 7.2.1, gap U3).
        min_majorant_margin: ``f64`` scalar, ``min(Sigma / q)``; target ``>= 2``
            (section 7.2.2).
        decision_checksum: ``f64`` scalar fingerprint of every **frozen decision**
            taken this step -- cell membership, attempt count, pair indices and
            accept bits.  Equal across two runs **iff** they took the same
            discrete path.  This is what makes a finite-difference gradient check
            meaningful: at frozen randomness ``J`` is a step function of any
            parameter entering a frozen decision, so an FD estimate is valid only
            where no decision flipped, and a flip otherwise produces a plausible
            wrong number rather than an error (section 13.4.1).

    These exist because the two defects this project has already produced were
    both silent: no error, no ``NaN``, no diagnostic.  They are arrays so they
    can be computed inside jit; they are **never branched on** there.

    Spec: section 3.1.
    """

    attempts_clipped: jax.Array
    max_attempts: jax.Array
    majorant_violations: jax.Array
    min_majorant_margin: jax.Array
    decision_checksum: jax.Array
    wall_recrossed: jax.Array
    """``i32`` particles that crossed more than one boundary face in a step.

    The wall analogue of ``attempts_clipped``: only the resolved crossings are
    applied and the remainder is clamped, so a nonzero value means the timestep is
    too large for the geometry (section 9.4).
    """
    inflow_injected: jax.Array
    """``i32`` particles injected through ``INFLOW`` faces this step."""
    inflow_shortfall: jax.Array
    """``f64`` emissions the per-face ``capacity`` could not place.

    A silently dropped entrant is a mass sink that looks exactly like correct
    physics, which is why it is counted rather than clipped away (section 9.3).

    All three are **required, not defaulted**.  A default of ``None`` makes them
    absent leaves, so a ``Diagnostics`` built without them has a different pytree
    structure -- which ``lax.scan`` rejects, and which silently changed the leaf
    count under ``tree_map``.  Requiring them makes an inconsistent instance
    unconstructible.
    """

    inflow_requested: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    inflow_escaped: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    inflow_sampler_failed: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    inflow_candidate_overflow: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    inflow_storage_overflow: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    inflow_tracking_exhausted: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    count_support_boundary: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    """Positive integer count means: ordinary two-sided score derivative is unsupported."""
    configuration_invalid: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    """Invalid traced weights or certified-bound inputs; checked before reporting."""
    surface_hits: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    """``i32`` embedded-surface (element / analytic body) hits resolved this step."""
    stencil_overrun: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    """``i32`` particles whose remaining move exceeded the stencil tracker's dilation
    (``|v| * remaining > radius * h``): an element outside the candidate list could have
    been missed.  Must be zero; raise the stencil radius or use ``tracker="walk"``."""
    walk_exhausted: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    """``i32`` particles the cell-walk tracker could not finish within ``K_walk``
    iterations (cell crossings plus surface hits); the remainder of their move was
    dropped.  Must be zero; raise ``K_walk``."""
    inside_body: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    """``i32`` particles found inside the body after the move and killed (the catch-all
    of SPARTA's tracking: a missed hit).  Must be zero."""
    surface_stuck: jax.Array = dataclasses.field(default_factory=lambda: jnp.int32(0))
    """``i32`` particles the stencil tracker deleted because a hit was still unresolved
    after ``K_bounce`` passes -- SPARTA's stuck particle (``MAXSTUCK``, deleted and
    counted in ``nstuck``): repeated reflections in a concave pocket of the body.  A rate
    to report, not an error; the replay deletes the same particle, so the adjoint stays
    consistent (the particle leaves like an outflow)."""


# --- parameters -----------------------------------------------------------


@register_dataclass
@dataclass(frozen=True)
class GridSpec:
    """Uniform orthogonal mesh description.

    Fields:
        lo: ``f64[D]`` domain lower corner.  **TRACED** — a gradient reaches the
            mesh geometry through the cell volume in the NTC attempt count
            (section 8.1) and through moment normalisation (section 10).
        hi: ``f64[D]`` domain upper corner.  **TRACED**.
        n: ``tuple[int, ...]`` of length ``D``, cells per axis.  **STATIC** —
            ``C = prod(n)`` is the shape of every per-cell array.
        transverse: ``f64`` scalar, the product of the domain extents in the
            ``3 - D`` suppressed directions.  **TRACED**.  Defaults to ``1.0``,
            which is correct for ``D == 3``.  It is an *array* default, not a bare
            Python float, so the invariant "every traced leaf is an f64 array"
            holds even when the field is left unset.

    **Why ``transverse`` exists.** A ``D``-dimensional DSMC run is not a
    ``D``-dimensional gas: it is a *three*-dimensional gas that is uniform in the
    ``3 - D`` directions being suppressed, which is why velocities stay
    3-component whatever ``D`` is (section 0.2).  Cell *volume* must therefore
    stay a true volume.  With ``prod(dx)`` alone it is a length at ``D == 1`` and
    an area at ``D == 2``, and since the NTC attempt count carries ``fnum / V``
    (section 8.1) and ``nrho`` carries ``count * fnum / V`` (section 10), a wrong
    ``V`` corrupts the collision rate and the density together.  It does **not**
    cancel: rescaling ``fnum`` to fix the rate leaves ``nrho`` wrong by the square.

    So ``transverse`` is the cross-sectional extent of the slab: an area
    (``Ly * Lz``) at ``D == 1``, a length (``Lz``) at ``D == 2``, and
    dimensionless ``1.0`` at ``D == 3``.  SPARTA reaches the same place from the
    other direction -- its domain is always three-dimensional and a "1-D" run is
    one cell in ``y`` and ``z`` with real extents.

    Only ``jax_dsmc.grid`` may do cell-index arithmetic with these (section 4);
    M5 replaces the implementation behind that module's four function names.

    Spec: section 3.1.
    """

    lo: jax.Array
    hi: jax.Array
    n: tuple[int, ...] = static_field()
    transverse: jax.Array = dataclasses.field(default_factory=lambda: jnp.float64(1.0))
    volume: jax.Array = dataclasses.field(default_factory=lambda: jnp.zeros(0, jnp.float64))
    """``f64[C]`` per-cell **flow** volume, or the empty ``f64[0]`` for the uniform
    ``prod(dx) * transverse``.  Set by :func:`jax_dsmc.mesh.cut.with_volume` for
    cut-cell grids (``Mesh.volume``); :func:`jax_dsmc.grid.cell_volume` returns it when
    its (static) length is ``C``, so the NTC attempt count, the moment normalisation and
    the initial fill see the cut volumes without any other change.  An empty array
    rather than ``None`` keeps the invariant that every traced leaf is an f64 array."""


@register_dataclass
@dataclass(frozen=True)
class SpeciesParams:
    """VSS species and pair tables.  All fields TRACED — this is the
    model-calibration gradient (section 7.1).

    Fields:
        mass: ``f64[S]`` molecular mass.
        diam: ``f64[S, S]`` VSS reference diameter.
        omega: ``f64[S, S]`` viscosity exponent; ``omega == 1`` is Maxwell
            molecules, which is Oracle 1's configuration (section 13.1).
        tref: ``f64[S, S]`` reference temperature.
        alpha: ``f64[S, S]`` VSS angular parameter; ``alpha == 1`` reduces to VHS.
        mr: ``f64[S, S]`` reduced mass, derived.
        prefactor: ``f64[S, S]`` cross-section prefactor, derived (section 7.1):
            ``pi * diam**2 * (2 k tref / mr)**(omega - 0.5) / Gamma(2.5 - omega)``.

    The pair tables are ``[S, S]`` and symmetric; ``mass`` is per species.
    ``S`` is STATIC and lives in :class:`SimStatic`.

    Spec: section 3.1.  Reference: ``collide_vss.cpp:122-125``.
    """

    mass: jax.Array
    diam: jax.Array
    omega: jax.Array
    tref: jax.Array
    alpha: jax.Array
    mr: jax.Array
    prefactor: jax.Array


@register_dataclass
@dataclass(frozen=True)
class FlowParams:
    """Free-stream / initial condition.  All fields TRACED.

    Fields:
        nrho: ``f64`` scalar number density.
        temp: ``f64`` scalar translational temperature.
        vstream: ``f64[3]`` bulk velocity — 3 components whatever ``D`` is
            (section 0.2).

    ``d/dtemp`` and ``d/dvstream`` are pathwise-exact through the reparameterised
    Maxwellian of section 5.1.

    Spec: section 3.1.
    """

    nrho: jax.Array
    temp: jax.Array
    vstream: jax.Array


# --- configuration: the traced/static split of SimConfig ------------------


@register_dataclass
@dataclass(frozen=True)
class SimParams:
    """The **traced** half of section 3.1's ``SimConfig``.  A pytree of leaves.

    Fields:
        dt: ``f64`` scalar timestep.  Traced: it enters the attempt count ``f``
            of section 8.1, whose sensitivity travels via the stochastic-rounding
            score term.
        fnum: ``f64[C]`` real molecules per simulator particle, **per cell**, not
            a scalar (section 3.3).  Read by the attempt count (8.1), moment
            normalisation (10) and boundary-flux measurement.
        epsilon: ``f64`` scalar randomised-timestep width; ``0.0`` until M7.

    ``fnum`` is per-cell because retrofitting it later would mean a simultaneous
    diff across ``ntc.py``, ``sample.py`` and ``boundary.py``; shipping it now
    costs one ``(C,)`` broadcast.  It comes with a loaded gun — see
    :func:`check_fnum_uniform`.

    Spec: sections 3.1 and 3.3.
    """

    dt: jax.Array
    fnum: jax.Array
    epsilon: jax.Array


@dataclass(frozen=True)
class SimStatic:
    """The **static** half of section 3.1's ``SimConfig``.

    Frozen, hashable, and compared by value: pass it through ``jax.jit``'s
    ``static_argnums`` (or ``static_argnames``).  It is deliberately *not* a
    registered pytree — every field determines a shape or a loop trip count, so
    none of them may ever become a leaf.

    Fields (all STATIC Python values, section 3.2):
        N: particle array capacity — the shape of every particle array.
        D: spatial dimension ``1 | 2 | 3`` — the shape of ``x`` and the ``v[:, :D]``
            slice.  Velocities stay 3-component (section 0.2).
        S: number of species — the shape of the parameter tables.
        G: number of collision groups — the shape of ``vremax``, ``G <= S``.
        A_max: maximum collision attempts per cell per step — the trip count of
            the attempt-slot ``scan`` (section 8.2).
        K_bounce: maximum wall bounces per step — the trip count of the bounce
            ``scan``.
        n_steps: number of timesteps — the trip count of the time ``scan``.
        remat_policy: name of the ``jax.checkpoint_policies`` attribute used by
            the time-loop ``scan`` (section 11.6).

    Changing any field forces a recompilation, which is exactly the intent: each
    one is a shape or a Python control-flow decision.

    Spec: sections 3.1 and 3.2.
    """

    N: int
    D: int
    S: int
    G: int
    A_max: int
    K_bounce: int
    n_steps: int
    remat_policy: str = "nothing_saveable"
    checkpoint_block_size: int = 0
    count_scheme: str = "stochastic"
    majorant_mode: str = "thermal"
    tracker: str = "stencil"
    """Embedded-surface tracker: ``"stencil"`` (per-cell candidate lists, ``K_bounce``
    passes) or ``"walk"`` (SPARTA's cell walk, ``K_walk`` iterations).  Ignored without a
    mesh."""
    K_walk: int = 8
    """Trip count of the cell walk: cell crossings plus surface hits per particle per step."""

    def __post_init__(self) -> None:
        """Reject values that would produce a wrong shape rather than an error."""
        if not isinstance(self.checkpoint_block_size, int) or self.checkpoint_block_size < 0:
            raise ValueError("checkpoint_block_size must be a nonnegative integer")
        if self.tracker not in ("stencil", "walk"):
            raise ValueError("tracker must be stencil or walk")
        if not isinstance(self.K_walk, int) or isinstance(self.K_walk, bool) or self.K_walk < 1:
            raise ValueError(f"K_walk must be a positive Python int; got {self.K_walk!r}")
        if self.count_scheme not in ("stochastic", "remainder"):
            raise ValueError("count_scheme must be stochastic or remainder")
        if self.majorant_mode not in ("thermal", "certified"):
            raise ValueError("majorant_mode must be thermal or certified")
        if self.D not in (1, 2, 3):
            raise ValueError(f"D must be 1, 2 or 3 (section 0.2); got {self.D}")
        if self.G > self.S:
            raise ValueError(f"G <= S is required (section 0.1); got G={self.G}, S={self.S}")
        for name in ("N", "S", "G", "A_max", "K_bounce", "n_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive Python int; got {value!r}")


# --- derived quantities and invariants ------------------------------------


def num_cells_of(n: tuple[int, ...]) -> int:
    """``C = prod(n)``.  STATIC (a Python int), from the STATIC ``GridSpec.n``.

    Duplicated deliberately from ``grid.num_cells``: ``types.py`` must not import
    ``grid.py`` (``grid`` imports ``types``), and callers building a ``(C,)``
    ``fnum`` before a ``GridSpec`` exists need the arithmetic.  This is not
    cell-index arithmetic, so it does not breach the narrow mesh interface of
    section 4.
    """
    c = 1
    for nd in n:
        c *= int(nd)
    return c


def check_fnum_uniform(fnum, has_reweighting: bool = False) -> None:
    """Assert the loaded-gun invariant of section 3.3.  Host-side, setup only.

    ``fnum`` is shipped as a traced ``f64[C]`` field but the clone/delete
    machinery that makes a *non-uniform* weight field correct is **not** shipped
    (section 3.3, half 2).  If ``fnum`` varies between cells and particles cross
    cells, a crossing particle silently represents the wrong number of real
    molecules and mass is not conserved.  Nothing else in the code notices — this
    is the silent-failure class (gap U7).

    Non-uniform ``fnum`` is legal only in a single-cell (0-D) run, or once
    reweighting exists.

    Args:
        fnum: ``f64[C]``.  Concrete (host) values — this runs at setup, outside
            jit, and raises a Python exception.
        has_reweighting: STATIC ``bool``; ``True`` once clone/delete or a
            continuous weight-transfer scheme exists.

    Raises:
        ValueError: if ``fnum`` is spatially non-uniform in a multi-cell run
            without reweighting.
    """
    if has_reweighting:
        return
    values = jnp.asarray(fnum)
    if values.size <= 1:
        return
    if not bool(jnp.allclose(values, values.reshape(-1)[0])):
        raise ValueError(
            "non-uniform fnum requires either C == 1 or reweighting machinery "
            "(PSEUDOCODE.md section 3.3); mass is otherwise silently not conserved"
        )


def _field_names(cls, static: bool) -> tuple[str, ...]:
    """Names of ``cls``'s STATIC (``static=True``) or TRACED fields, in order."""
    return tuple(
        f.name for f in dataclasses.fields(cls) if bool(f.metadata.get("static", False)) is static
    )


def traced_fields(cls) -> tuple[str, ...]:
    """Names of the TRACED (pytree leaf) fields of a container in this module."""
    return _field_names(cls, static=False)


def static_fields(cls) -> tuple[str, ...]:
    """Names of the STATIC (pytree metadata) fields of a container in this module."""
    return _field_names(cls, static=True)
