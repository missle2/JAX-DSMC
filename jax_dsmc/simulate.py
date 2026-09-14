"""The time loop (spec section 12).

Intra-step ordering is **`move -> sort -> collide`**, matching both reference
codes (SPARTA ``update.cpp:337-364``; [DF]'s ``move -> buildCellOccupancy ->
collisions``).

> **This is NOT [P3]'s ordering, and the difference is real.** [P3] specifies
> `collide -> move`, which puts the transport term *inside* ``B_k``:
> ``beta_k = B(beta_{k+1} + dt P* alpha_{k+1})``.  Ours puts it outside:
> ``beta_k = B beta_{k+1} + dt P* alpha_{k+1})``.  Verified by taking real
> ``jax.vjp``s of both (exact match one way, `2.45e-02` mismatch the other), and
> the forms differ by `O(dt ||B - I||)`, which does **not** vanish at realistic
> timesteps.
>
> This costs nothing in production -- ``jax.grad`` differentiates the loop as
> written, so the computed gradient is correct either way.  It matters only to
> someone hand-checking the adjoint against the paper's formula.  Note that the 0D
> oracles of section 13.1/13.2 provably **cannot** detect it (``alpha == 0`` there,
> so both forms collapse), which is why section 13.3's 1D two-step oracle exists
> (gap U5).

Every loop here is a ``scan`` with a **static** trip count -- the time loop, the
attempt-slot loop, and the wall-bounce loop.  No ``while_loop`` anywhere: a
data-dependent trip count is neither reverse-differentiable nor statically shaped,
and the jaxpr audit in the test suite enforces it.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax

from jax_dsmc import boundary, inflow, sample, sort
from jax_dsmc.boundary import BC
from jax_dsmc.collide.ntc import collide, decision_weights
from jax_dsmc.move import move_with_decisions
from jax_dsmc.rng import STAGE_INFLOW, STAGE_MOVE, key_for
from jax_dsmc.state import CollideState, ParticleState

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


def _step(
    carry,
    k,
    grid,
    sp,
    params,
    static,
    base_key,
    bc,
    collide_only,
    emit_moments,
    reward,
    faces=None,
    wall=None,
    res=None,
    capacity=0,
    score_spread="particle",
    acc_delta=0.0,
    score_mode="paper",
):
    """One timestep.  Module-level and bound with ``partial`` (decision 14).

    ``collide_only`` short-circuits advection for the 0D milestone-1 case, where
    there is a single cell and transport is identically a no-op.  ``emit_moments``
    selects whether the per-step output is the full :class:`CellMoments` or just
    the per-cell temperature.  Both are STATIC Python bools, so they select a code
    path at trace time rather than branching on a traced value.

    ``reward`` is the third: ``None`` (the default) leaves the carry as
    ``(state, cstate)``; a callable ``(state, table) -> f64`` extends it to
    ``(state, cstate, J)`` and accumulates a **windowed** objective in the causal
    DiCE form of section 11.5.  It is called *after* ``collide``, so
    ``state.logh_acc`` is the running prefix ``L_{<=k}`` and each reward is scored
    only against the decisions that preceded it.  ``table`` is still valid there:
    ``collide`` rewrites velocities but never positions, so cell occupancy is
    unchanged.
    """
    if reward is None:
        state, cstate = carry
    else:
        state, cstate, j_acc = carry
    bc_chk = jnp.float64(0.0)
    recrossed = jnp.array(0, jnp.int32)
    injected = jnp.array(0, jnp.int32)
    shortfall = jnp.float64(0.0)
    if not collide_only:
        if faces is None:
            state, fold = move_with_decisions(state, grid, params.dt, bc)
            # the boundary fold index is a frozen decision too -- see boundary.fold_index
            w = decision_weights(fold.size).reshape(fold.shape)
            bc_chk = lax.stop_gradient(jnp.sum(w * fold))
        else:
            # per-face path (section 9.4): thermal walls, inflow, mixed geometries
            k_move = key_for(base_key, k, STAGE_MOVE)
            x, v, alive, recrossed, logh_bc = boundary.apply_faces(
                state.x,
                state.v,
                state.alive,
                grid,
                faces,
                wall,
                sp.mass[0],
                params.dt,
                k_move,
                epsilon=params.epsilon,
                score_spread=score_spread,
                acc_delta=acc_delta,
            )
            fold = boundary.fold_index(x, grid, BC.PERIODIC)
            w = decision_weights(fold.size).reshape(fold.shape)
            bc_chk = lax.stop_gradient(jnp.sum(w * fold))
            # M7: the boundary decisions' score joins the collision score in the same
            # accumulator, because `surrogate` weights the reward by `grad` of the WHOLE
            # prefix `L_{<=k}` -- a separate accumulator would drop the cross terms.
            state = ParticleState(
                x=x,
                v=v,
                species=state.species,
                alive=alive,
                logh_acc=state.logh_acc
                + (jnp.sum(logh_bc) if score_mode == "trajectory" else logh_bc),
            )
            if res is not None:
                state, injected, shortfall = inflow.inject(
                    state,
                    grid,
                    faces,
                    res,
                    sp.mass[0],
                    params.fnum[0],
                    params.dt,
                    key_for(base_key, k, STAGE_INFLOW),
                    capacity,
                )
    table = sort.build_table(state, grid)
    # the cell temperature feeds the a-priori majorant floor (section 7.2.3)
    m = sample.moments(state, table, grid, params, sp)
    state, cstate, diag = collide(
        state,
        table,
        cstate,
        grid,
        sp,
        params,
        static,
        m.temp,
        k,
        base_key,
        score_mode=score_mode,
    )
    diag = dataclasses.replace(
        diag,
        decision_checksum=diag.decision_checksum + bc_chk,
        wall_recrossed=recrossed,
        inflow_injected=injected,
        inflow_shortfall=shortfall,
    )
    new_carry = (state, cstate) if reward is None else (state, cstate, j_acc + reward(state, table))
    if emit_moments == "none":
        # Emit nothing per step.  `scan` stacks its ys over time, so a full
        # CellMoments costs `n_steps * C * 6 * 8` bytes -- 29 GB for 8 steps at
        # C = 1.5e8, which is the regime the multi-device runs live in.  The
        # diagnostics stay: they are per-step SCALARS, and `A_max` resizing needs
        # them.  Snapshots come from `chunk_moments` instead.
        return new_carry, (None, diag)
    return new_carry, (m, diag)


def run(
    state: ParticleState,
    cstate: CollideState,
    grid,
    sp,
    params,
    static,
    base_key,
    bc: BC = BC.PERIODIC,
    collide_only: bool = False,
    emit_moments: bool = True,
    step0: int = 0,
    reward=None,
    faces=None,
    wall=None,
    res=None,
    capacity: int = 0,
    score_spread: str = "particle",
    acc_delta: float = 0.0,
    score_mode: str = "paper",
):
    """Run ``static.n_steps`` timesteps.

    ``score_mode="paper"`` keeps terminal-label accumulators. ``"trajectory"``
    stores the same complete likelihood prefix in every slot, counting each event
    once. Initialize the accumulator to zero or continue with the same mode.
    Consume the latter with ``adjoint.trajectory_surrogate``. This likelihood
    does not differentiate hard cell memberships or changing rounding support.

    Returns:
        ``((state, cstate), (out, diagnostics))`` where the second element is
        stacked over time.  ``out`` is the full :class:`CellMoments` per step when
        ``emit_moments`` (the default), or just ``f64[n_steps, C]`` temperature
        when not -- which skips the ``(N,3,3)`` pressure-tensor intermediate that
        the majorant does not need (section 8.6).  Every ``diagnostics`` field is
        ``[n_steps]``.

    ``jax.checkpoint`` wraps the step body so a reverse pass rematerialises rather
    than storing every intermediate.  It is a no-op in the forward pass, so it
    costs nothing until a gradient is actually taken.

    The RNG contract (section 1) means the step index ``k`` is the *only* thing
    that varies the random stream: keys are recomputed from ``(base_key, k, stage)``
    and never carried, which is what makes the rematerialised backward pass replay
    the same draws the forward pass took.

    ``step0`` offsets that index.  It exists so a long run can be executed in
    chunks -- see :func:`run_adaptive` -- without the chunk boundaries changing a
    single random draw: chunking ``[0, 200)`` into two halves consumes exactly the
    stream that one 200-step call would have.
    """
    if score_mode not in ("paper", "trajectory"):
        raise ValueError("score_mode must be paper or trajectory")
    if score_mode == "trajectory" and res is not None:
        raise ValueError(
            "Use ensemble_inflow for P3 inlet gradients; recycled slots do not preserve trajectory scores"
        )
    if score_mode == "trajectory" and score_spread != "particle":
        raise ValueError("Trajectory scoring requires each boundary event exactly once")
    body = functools.partial(
        _step,
        grid=grid,
        sp=sp,
        params=params,
        static=static,
        base_key=base_key,
        bc=bc,
        collide_only=collide_only,
        emit_moments=emit_moments,
        reward=reward,
        score_spread=score_spread,
        acc_delta=acc_delta,
        score_mode=score_mode,
        faces=faces,
        wall=wall,
        res=res,
        capacity=capacity,
    )
    policy = getattr(jax.checkpoint_policies, static.remat_policy, None)
    body = jax.checkpoint(body, policy=policy, static_argnums=())
    init = (state, cstate) if reward is None else (state, cstate, jnp.float64(0.0))
    return lax.scan(body, init, step0 + jnp.arange(static.n_steps))


def run_adaptive(
    state: ParticleState,
    cstate: CollideState,
    grid,
    sp,
    params,
    static,
    base_key,
    bc: BC = BC.PERIODIC,
    collide_only: bool = False,
    emit_moments: bool | str = True,
    chunk: int = 32,
    growth: float = 1.4,
    max_resizes: int = 8,
    verbose: bool = False,
    sample_stride: int = 1,
    chunk_moments: bool = False,
):
    """Run ``static.n_steps`` steps, growing ``A_max`` whenever the flow outgrows it.

    **Why this exists.** ``A_max`` is the trip count of the attempt-slot ``scan``,
    so it is baked into the compiled program and cannot change while that program
    runs.  But the *right* value is a property of the flow: the attempt count goes
    as ``npc^2`` at fixed cell volume, so compression drives it, and for a shape
    being optimised nobody can know it in advance.  Guessing it is not a strategy;
    the failure mode is silent under-collision (section 8.2, gap U4).

    **What makes resizing cheap.** The simulation state is just arrays, so
    continuing after a recompile costs *nothing but the compile* -- there is no
    restart, no re-initialisation, no lost physics.  So the loop runs in chunks of
    ``chunk`` steps, reads ``attempts_clipped`` on the host, and if the capacity was
    exceeded it grows ``A_max`` and **re-runs that chunk from the state it started
    from**.  Only the offending chunk is redone, never the whole run.  Each distinct
    ``A_max`` compiles once and is then reused (and the on-disk compilation cache
    carries it across processes), so a run settles onto its working capacity after
    a couple of resizes and pays nothing thereafter.

    ``step0`` keeps the random stream identical to an unchunked run, so chunking
    changes results not at all -- ``tests/test_simulate.py`` asserts bit equality.

    **Keeping the output affordable.**  ``scan`` stacks its per-step output, so a
    long run at a large cell count cannot emit full moments every step.  Two knobs
    control that, and at scale both are needed:

    ``sample_stride``
        keep only every ``stride``-th step of each chunk.  Reduces what is
        *retained*, not what is computed, so it does not help the peak inside a
        chunk -- shrink ``chunk`` for that.
    ``chunk_moments``
        with ``emit_moments="none"``, compute one :class:`CellMoments` snapshot from
        the state at the end of each chunk.  Peak output memory then depends on the
        cell count alone, not on the step count, which is what makes a
        billion-particle run's field output tractable.  Set ``chunk`` to the frame
        interval.

    Returns:
        ``((state, cstate), (out, diag), history)`` where ``history`` is the list of
        ``(step, A_max, clipped, peak_attempts)`` resize events.

    Not for use inside a differentiated run: the host-side branch is Python control
    flow.  Calibrate with this, then take gradients through :func:`run` at the
    settled ``A_max`` -- which is the natural split anyway, since an optimiser's
    outer loop can resize between iterations.
    """
    total = static.n_steps
    a_max = static.A_max
    done = 0
    outs, diags, history = [], [], []
    resizes = 0

    while done < total:
        n = min(chunk, total - done)
        st = dataclasses.replace(static, A_max=a_max, n_steps=n)
        saved = (state, cstate)
        (state, cstate), (out, diag) = run(
            state,
            cstate,
            grid,
            sp,
            params,
            st,
            base_key,
            bc=bc,
            collide_only=collide_only,
            emit_moments=emit_moments,
            step0=done,
        )
        clipped = int(jnp.sum(diag.attempts_clipped))
        if clipped > 0:
            peak = int(jnp.max(diag.max_attempts))
            new_a = max(a_max + 1, math.ceil(growth * peak))
            history.append((done, a_max, clipped, peak))
            if verbose:
                print(
                    f"  step {done}: A_max={a_max:,} clipped {clipped:,} "
                    f"(peak wanted {peak:,}) -> {new_a:,}"
                )
            resizes += 1
            if resizes > max_resizes:
                raise RuntimeError(
                    f"A_max failed to converge after {max_resizes} resizes "
                    f"(now {new_a:,}); the flow is outgrowing capacity faster than "
                    "the growth factor can follow."
                )
            a_max = new_a
            state, cstate = saved  # redo THIS chunk only, from where it began
            continue
        if chunk_moments:
            table = sort.build_table(state, grid)
            out = jax.tree.map(
                lambda a: a[None, ...], sample.moments(state, table, grid, params, sp)
            )
        elif sample_stride > 1:
            out = jax.tree.map(lambda a: a[sample_stride - 1 :: sample_stride], out)
        if out is not None:
            outs.append(out)
        diags.append(diag)
        done += n

    if not outs:
        cat = None
    elif len(outs) > 1:
        cat = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *outs)
    else:
        cat = outs[0]
    dcat = (
        jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *diags)
        if len(diags) > 1
        else diags[0]
    )
    return (state, cstate), (cat, dcat), history
