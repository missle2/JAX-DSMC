"""Performance and parallelisability contract (spec sections 8.6 and 14).

These are **structural** assertions, not wall-clock ones.  A timing threshold in
CI is a flaky test on shared hardware -- the measured numbers live in section 8.6
and in ``scratch/perf_*.py``.  What is asserted here is the shape of the compiled
program, which is what actually determines whether the design's parallelism is
available:

* no ``while_loop`` anywhere (a data-dependent trip count is not
  reverse-differentiable and not statically shaped);
* no RNG *inside* the attempt-slot scan (section 1 rule 3 -- drawing per slot
  halved throughput and cost 7x in compile time, section 8.6);
* no recompilation across calls that differ only in traced values;
* ``vmap`` composability, which M8's ensemble adjoint needs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_dsmc import sampling  # noqa: E402
from jax_dsmc.collide import make_prefactor  # noqa: E402
from jax_dsmc.collide.ntc import collide, vremax_init  # noqa: E402
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


def case(N=2000, n=(2, 2, 2), A_max=32, T=300.0):
    mass = jnp.array([AR_MASS])
    mr = jnp.array([[AR_MASS / 2]])
    diam = jnp.array([[AR_DIAM]])
    om = jnp.array([[1.0]])
    tref = jnp.array([[AR_TREF]])
    al = jnp.array([[1.0]])
    sp = SpeciesParams(
        mass=mass,
        diam=diam,
        omega=om,
        tref=tref,
        alpha=al,
        mr=mr,
        prefactor=make_prefactor(diam, om, tref, mr),
    )
    g = GridSpec(lo=jnp.zeros(3), hi=jnp.full(3, 1e-4), n=n)
    C = num_cells(g)
    st = SimStatic(N=N, D=3, S=1, G=1, A_max=A_max, K_bounce=1, n_steps=1)
    pr = SimParams(dt=jnp.float64(1e-8), fnum=jnp.ones(C), epsilon=jnp.float64(0.0))
    k = jax.random.PRNGKey(0)
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
    return state, cs, g, sp, pr, st, jnp.full((C,), T), k


def collide_jaxpr(**kw):
    state, cs, g, sp, pr, st, T, k = case(**kw)
    tb = build_table(state, g)
    return jax.make_jaxpr(lambda s: collide(s, tb, cs, g, sp, pr, st, T, 0, k)[0].v)(state)


def scan_bodies(jaxpr):
    out = []
    for e in jaxpr.jaxpr.eqns:
        if e.primitive.name == "scan":
            sub = e.params["jaxpr"]
            out.append(getattr(sub, "jaxpr", sub))
    return out


RNG_HINTS = ("random", "threefry", "rng", "prng")


def test_no_rng_inside_the_attempt_slot_scan():
    """Section 1 rule 3, as a structural regression.

    Drawing per slot -- ``fold_in`` plus ``randint``/``uniform`` in the loop body,
    about nine tiny kernels per iteration -- halved throughput and multiplied
    compile time by 7 (section 8.6).  All randomness must be drawn once per stage,
    shaped ``(A_max, C)``, and consumed as the scan's ``xs``.
    """
    bodies = scan_bodies(collide_jaxpr())
    assert len(bodies) == 1, f"expected exactly one scan in collide, found {len(bodies)}"
    names = {e.primitive.name for e in bodies[0].eqns}
    offenders = sorted(n for n in names if any(h in n.lower() for h in RNG_HINTS))
    assert offenders == [], f"RNG primitives inside the attempt-slot scan: {offenders}"


def test_collide_contains_no_while_loop():
    """A data-dependent trip count breaks both the adjoint and static shapes."""
    txt = str(collide_jaxpr())
    assert "while" not in txt


@pytest.mark.parametrize("n", [(1, 1, 1), (2, 2, 2), (4, 4, 4)])
def test_jaxpr_is_identical_across_particle_counts_at_fixed_capacity(n):
    """Static-shape audit (section 14): the program must not depend on live count."""
    state, cs, g, sp, pr, st, T, k = case(n=n)
    tb = build_table(state, g)
    a = str(jax.make_jaxpr(lambda s: collide(s, tb, cs, g, sp, pr, st, T, 0, k)[0].v)(state))
    half = ParticleState(
        x=state.x,
        v=state.v,
        species=state.species,
        alive=jnp.arange(state.x.shape[0]) < state.x.shape[0] // 2,
        logh_acc=state.logh_acc,
    )
    tb2 = build_table(half, g)
    b = str(jax.make_jaxpr(lambda s: collide(s, tb2, cs, g, sp, pr, st, T, 0, k)[0].v)(half))
    assert a == b


def test_no_recompilation_across_traced_values():
    """Catches Python-level branching on a traced value."""
    state, cs, g, sp, pr, st, T, k = case()
    tb = build_table(state, g)
    f = jax.jit(collide, static_argnames=("static",))
    f(state, tb, cs, g, sp, pr, st, T, 0, k)[0].v.block_until_ready()
    n0 = f._cache_size()
    for step in range(1, 6):
        pr2 = SimParams(dt=pr.dt * (1.0 + 0.01 * step), fnum=pr.fnum, epsilon=pr.epsilon)
        f(state, tb, cs, g, sp, pr2, st, T, step, k)[0].v.block_until_ready()
    assert f._cache_size() == n0


def test_step_is_vmappable():
    """Proof of purity, and directly reusable for M8's ensemble adjoint."""
    state, cs, g, sp, pr, st, T, k = case()
    tb = build_table(state, g)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    vs = jax.vmap(lambda kk: collide(state, tb, cs, g, sp, pr, st, T, 0, kk)[0].v)(keys)
    assert vs.shape == (4, st.N, 3)


def scan_eqn(jaxpr):
    eqns = [e for e in jaxpr.jaxpr.eqns if e.primitive.name == "scan"]
    assert len(eqns) == 1
    return eqns[0]


@pytest.mark.parametrize("a_max", [16, 64])
@pytest.mark.parametrize("n", [(2, 2, 2), (4, 4, 4)])
def test_batched_draws_are_shaped_a_max_by_cells(a_max, n):
    """Every scanned input is an ``(A_max, C)`` draw sliced one row per slot.

    This pins the *shape* of the section-1-rule-3 batching: the randomness scales
    with the work actually done (``A_max * C`` attempts) and not with ``N``.  If a
    draw ever became per-particle, or moved back inside the loop, this fails.
    """
    C = num_cells(GridSpec(lo=jnp.zeros(3), hi=jnp.ones(3), n=n))
    eqn = scan_eqn(collide_jaxpr(n=n, A_max=a_max))
    n_const = eqn.params["num_consts"]
    n_carry = eqn.params["num_carry"]
    xs = eqn.invars[n_const + n_carry :]
    assert xs, "the scan takes no xs -- randomness is not being batched"
    for v in xs:
        shape = tuple(v.aval.shape)
        assert shape[0] == a_max, f"scanned input has leading dim {shape[0]}, expected A_max"
        assert shape[1:] in ((), (C,)), f"scanned input shape {shape} is not (A_max,) or (A_max, C)"
