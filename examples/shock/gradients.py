"""M8 gradient validation: shock response to inlet velocity against Rankine-Hugoniot.

    python3 -m examples.shock.gradients

The closed form is Rankine-Hugoniot as a function of the upstream Mach number
``M = u_1 / a_1``, with ``a_1 = sqrt(gamma k_B T_1 / m)`` fixed because ``T_1`` is held:

    rho_2/rho_1 = (gamma+1) M^2 / ((gamma-1) M^2 + 2)
    T_2/T_1     = (2 gamma M^2 - (gamma-1)) ((gamma-1) M^2 + 2) / ((gamma+1)^2 M^2)

so ``d/du_1 = (1/a_1) d/dM``.  The two derivatives move in *opposite* directions with Mach
-- density response falls, temperature response rises -- which makes the sweep a shape test
rather than a single number.

**Result: the temperature ratio validates; the density ratio comes out with the WRONG
SIGN and is not usable.**

``d(T_2/T_1)/du_1`` agrees across ``M = 1.6 ... 4.0`` at a 300-step horizon.  It works
because it is set by the prescribed inflow temperature and by local equilibration, both of
which are fast and both of which reach the objective **pathwise**.

``d(rho_2/rho_1)/du_1`` does not, and the failure is not subtle -- it is a sign flip.
Against an ensemble finite difference of the *same* 300-step functional (ground truth,
since FD differences two ensemble means whatever the coupling):

| ``M_1`` | ensemble FD | AD | Rankine-Hugoniot |
|---|---|---|---|
| 1.60 | **+1.572e-03** +- 2.2e-04 | -2.538e-03 | +4.034e-03 |
| 2.00 | **+1.763e-03** +- 1.8e-04 | -1.937e-03 | +3.182e-03 |
| 3.00 | **+2.567e-04** +- 1.6e-04 | -4.298e-04 | +1.624e-03 |

**This is an attribution failure, not a chaos failure.**  At 300 steps the per-seed spread
is 8.7e-03 against a 3.2e-03 signal -- a factor of three, not the nine orders of magnitude
that appear at 1000 steps.  The gradient is precise enough here; it is simply wrong.

The mechanism is the near-cancellation of two score-carried terms that are attributed with
different accuracy.  The density ratio reaches ``u_1`` only through discrete decisions --
the initial-condition ``alive`` Bernoulli (which *raises* the ratio with ``u_1``) and the
injection count (which *lowers* it during the transient) -- and the per-particle attribution
of `docs/ATTRIBUTION.md` under-counts each by a different factor because their causal cones
differ.  Under-counting two terms of opposite sign by different factors can flip the sum,
and here it does.

A second, independent obstacle sits behind that one: reaching the Rankine-Hugoniot
compression at all takes ~1000 steps, because it needs mass to redistribute across the
domain, and by then chaotic amplification has put the pathwise derivative nine orders of
magnitude above its own signal:

| steps | forward error in ``rho_2/rho_1`` | per-seed spread of its derivative |
|---|---|---|
| 300 | -38.7 % | 8.7e-03 |
| 1000 | **-2.6 %** | **4.0e+06** |
| 1500 | -- | 1.0e+12 |

So the two failure modes bracket the problem from both sides: fix the attribution and the
horizon still fails; shorten the horizon and the attribution still fails.  Obtaining this
derivative needs the unbiased estimator of `docs/ATTRIBUTION.md` *and* a method built for
chaotic sensitivities (shadowing / least-squares sensitivity).  Neither is built.

**What the temperature comparison does and does not test.**  The two-sided-inflow geometry
prescribes the downstream reservoir from Rankine-Hugoniot at the *same* ``u_1``, so the
shock is held stationary, and the initial condition is built from the traced ``u_1`` as
well.  The derivative therefore runs end to end through initialisation, inflow injection at
both faces, advection, collisions and outflow.  It is **not** a test that the solver
discovers Rankine-Hugoniot unaided; the forward moments in `examples/shock/README.md`
section 4 check that, to 0.9 %.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

from examples.shock.inflow_driven import (
    GAMMA,
    GAS,
    NRHO1,
    TEMP1,
    Case,
    build,
)
from jax_dsmc import adjoint, inflow, sampling
from jax_dsmc.collide import vremax_init
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import num_cells
from jax_dsmc.safe import safe_log
from jax_dsmc.simulate import run
from jax_dsmc.state import CollideState, ParticleState
from jax_dsmc.viz import (
    INK_SECONDARY,
    REFERENCE,
    SERIES_ANALYTIC,
    SERIES_MEASURED,
    make_figure,
    savefig,
)

RESULTS = "examples/shock/results"
FIGURES = "examples/shock/figures"

# The domain must be long compared with the shock THICKNESS or there are no plateaux to
# take a ratio of: a Mach-2 shock is ~6 mfp thick, so an 8 mfp box is entirely shock.  A
# first attempt at this study used one, and its "plateau" ratios sat 22-46 % below
# Rankine-Hugoniot -- the same failure `examples/shock/README.md` section 4 records.
CASE = Case(mach=2.0, n_cells=80, npc=40, gap_mfp=60.0, a_max=96, capacity=64)
HORIZON = 300
SEEDS = 128
MACHS = (1.6, 2.0, 2.5, 3.0, 4.0)
EDGE = 8  # cells at each end taken as the plateau


def a1():
    return math.sqrt(GAMMA * K_BOLTZ * TEMP1 / GAS.mass)


def rh_ratios(mach, gamma=GAMMA):
    """``(rho_2/rho_1, T_2/T_1)`` -- written in ``jnp`` so it can be differentiated."""
    m2 = jnp.asarray(mach) ** 2
    rho = (gamma + 1.0) * m2 / ((gamma - 1.0) * m2 + 2.0)
    temp = (
        (2.0 * gamma * m2 - (gamma - 1.0)) * ((gamma - 1.0) * m2 + 2.0) / ((gamma + 1.0) ** 2 * m2)
    )
    return rho, temp


def analytic_dratios_du1(mach):
    """``d(rho_2/rho_1)/du_1`` and ``d(T_2/T_1)/du_1``, by differentiating the closed form."""
    scale = 1.0 / a1()
    drho = jax.grad(lambda m: rh_ratios(m)[0])(float(mach)) * scale
    dtemp = jax.grad(lambda m: rh_ratios(m)[1])(float(mach)) * scale
    return float(drho), float(dtemp)


def _initial_state_of(u1, rho_r, temp_r, key, grid, sp, n_part, length):
    """`inflow_driven.initial_state`, but as a function of the traced inlet velocity."""
    x = sampling.sample_positions(jax.random.fold_in(key, 1), n_part, grid)
    left = x[:, 0] < 0.5 * length
    v_up = sampling.sample_maxwellian(
        jax.random.fold_in(key, 2),
        n_part,
        jnp.asarray(TEMP1) * 1.0,
        sp.mass[0],
        jnp.array([1.0, 0.0, 0.0]) * u1,
    )
    v_dn = sampling.sample_maxwellian(
        jax.random.fold_in(key, 3),
        n_part,
        TEMP1 * temp_r,
        sp.mass[0],
        jnp.array([1.0, 0.0, 0.0]) * (u1 / rho_r),
    )
    u = jax.random.uniform(jax.random.fold_in(key, 4), (n_part,), dtype=jnp.float64)
    frac_up = 1.0 / (1.5 * rho_r)
    p_alive = jnp.where(left, frac_up, frac_up * rho_r)
    alive = u < p_alive
    # The initial density ratio is set by a Bernoulli whose probability depends on u1, so
    # it is score-carried exactly like the injection count.  Without this term
    # `d(rho_2/rho_1)/du_1` is missing its dominant contribution -- measured, it came out
    # with the WRONG SIGN, which is what a missing leading term looks like rather than
    # what noise looks like.
    logh = jnp.where(alive, safe_log(p_alive), safe_log(1.0 - p_alive))
    return ParticleState(
        x=x,
        v=jnp.where(left[:, None], v_up, v_dn),
        species=jnp.zeros(n_part, jnp.int32),
        alive=alive,
        logh_acc=logh,
    )


def _measured_ratios(u1, key, built, case, n_steps):
    """``(rho_2/rho_1, T_2/T_1)`` from the plateaux, differentiable in ``u1``."""
    grid, sp, params, static, faces, _res0, _lam, length, _fn, _st = built
    static = dataclasses.replace(static, n_steps=n_steps)
    mach = u1 / a1()
    rho_r, temp_r = rh_ratios(mach)
    # BOTH reservoirs follow u1, so the prescribed jump stays self-consistent and the
    # shock stays put; see the module docstring for what that means for the test
    res = inflow.InflowParams(
        nrho=jnp.stack([jnp.asarray(NRHO1) * 1.0, NRHO1 * rho_r]),
        temp=jnp.stack([jnp.asarray(TEMP1) * 1.0, TEMP1 * temp_r]),
        vstream=jnp.stack(
            [
                jnp.array([1.0, 0.0, 0.0]) * u1,
                jnp.array([1.0, 0.0, 0.0]) * (u1 / rho_r),
            ]
        ),
    )
    # Build the initial condition from the TRACED u1 rather than reusing `built`'s, which
    # is frozen at the nominal Mach and so contributes exactly zero to the derivative.
    # The velocities are pathwise in u1; the alive mask that sets the density ratio is a
    # threshold, hence score-carried -- see `docs/ATTRIBUTION.md`.
    state = _initial_state_of(u1, rho_r, temp_r, key, grid, sp, static.N, length)
    c = num_cells(grid)
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((c,), TEMP1),
            sp.omega[0, 0],
            sp.tref[0, 0],
            sp.mr[0, 0],
            sp.prefactor[0, 0],
            c,
            1,
        )
    )
    (out, _), _ = run(
        state,
        cstate,
        grid,
        sp,
        params,
        static,
        key,
        faces=faces,
        res=res,
        capacity=case.capacity,
        emit_moments="none",
    )
    n_cells = case.n_cells
    dx = length / n_cells
    idx = jnp.clip((out.x[:, 0] / dx).astype(jnp.int32), 0, n_cells - 1)
    # DiCE weighting, per particle.  The density ratio is built from particle COUNTS,
    # which are discrete: its pathwise derivative is identically zero, and the whole
    # sensitivity of the count to `u_1` lives in the injection score that `inflow.inject`
    # accumulates.  Weighting every per-particle contribution by `magic(L_i)` -- value 1,
    # gradient `grad L_i` -- carries that score into any smooth functional of the sums,
    # a ratio included.  Without it `d(rho_2/rho_1)/du_1` reads exactly 0.0.
    live = out.alive.astype(jnp.float64) * adjoint.magic(out.logh_acc)
    cnt = jnp.zeros(n_cells).at[idx].add(live)
    sq = jnp.zeros(n_cells).at[idx].add(live * jnp.sum(out.v**2, axis=-1))
    mv = jnp.stack([jnp.zeros(n_cells).at[idx].add(live * out.v[:, k]) for k in range(3)])
    safe = jnp.where(cnt > 0, cnt, 1.0)
    temp = GAS.mass * (sq / safe - jnp.sum((mv / safe) ** 2, axis=0)) / (3.0 * K_BOLTZ)
    up = slice(0, EDGE)
    dn = slice(n_cells - EDGE, n_cells)
    return jnp.mean(cnt[dn]) / jnp.mean(cnt[up]), jnp.mean(temp[dn]) / jnp.mean(temp[up])


def measure(mach, case=CASE, seeds=SEEDS, n_steps=HORIZON):
    built = build(dataclasses.replace(case, mach=mach))
    u1 = mach * a1()
    g = jax.jit(
        jax.vmap(
            lambda k: jax.jacrev(lambda u: jnp.stack(_measured_ratios(u, k, built, case, n_steps)))(
                u1
            )
        )
    )
    keys = jax.random.split(jax.random.PRNGKey(17), seeds)
    # the reverse pass holds a tape per particle-key; chunk rather than OOM, and halve on
    # failure so a bigger case does not need the constant retuned by hand
    chunk = max(1, min(seeds, int(2.0e9 / (4290 * case.n_cells * case.npc))))
    while chunk >= 1:
        try:
            j = np.concatenate(
                [
                    np.asarray(jax.block_until_ready(g(keys[i : i + chunk])))
                    for i in range(0, seeds, chunk)
                ]
            )
            return j[:, 0], j[:, 1]
        except Exception as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc) and "out of memory" not in str(exc).lower():
                raise
            chunk //= 2
    raise RuntimeError("a single key does not fit on this device")


def run_sweep(machs=MACHS, case=CASE, seeds=SEEDS, n_steps=HORIZON, out_dir=RESULTS):
    os.makedirs(out_dir, exist_ok=True)
    lines = [
        f"# case n_cells={case.n_cells} npc={case.npc} gap_mfp={case.gap_mfp} "
        f"seeds={seeds} horizon={n_steps} edge={EDGE} a1={a1():.6f}",
        "# mach drho_du1 drho_se drho_analytic dtemp_du1 dtemp_se dtemp_analytic",
    ]
    for m in machs:
        dr, dt = measure(m, case, seeds, n_steps)
        ar, at = analytic_dratios_du1(m)
        lines.append(
            f"{m:.4f} {dr.mean():.8e} {dr.std(ddof=1) / math.sqrt(seeds):.8e} {ar:.8e} "
            f"{dt.mean():.8e} {dt.std(ddof=1) / math.sqrt(seeds):.8e} {at:.8e}"
        )
        print(f"  M={m:.2f} done", flush=True)
    path = os.path.join(out_dir, "gradients.log")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def parse(out_dir=RESULTS):
    path = os.path.join(out_dir, "gradients.log")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run `python3 -m examples.shock.gradients`")
    meta, rows = {}, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                for tok in line[1:].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta[k] = v
                continue
            rows.append([float(t) for t in line.split()])
    return meta, np.array(rows)


def figure(meta, rows, path):
    m = rows[:, 0]
    dr, dr_se, dr_an = rows[:, 1], rows[:, 2], rows[:, 3]
    dt, dt_se, dt_an = rows[:, 4], rows[:, 5], rows[:, 6]
    fine = np.linspace(m.min() * 0.95, m.max() * 1.05, 200)
    an_r = np.array([analytic_dratios_du1(x)[0] for x in fine])
    an_t = np.array([analytic_dratios_du1(x)[1] for x in fine])

    fig, ax = make_figure(1, 3, figsize=(16.5, 4.6))
    for i, (meas, se, an_curve, lab, sym) in enumerate(
        (
            (dr, dr_se, an_r, r"$\partial(\rho_2/\rho_1)/\partial u_1$", "density"),
            (dt, dt_se, an_t, r"$\partial(T_2/T_1)/\partial u_1$", "temperature"),
        ),
    ):
        ax[i].plot(
            fine, an_curve, "-", color=SERIES_ANALYTIC, linewidth=2.0, label="Rankine-Hugoniot"
        )
        ax[i].errorbar(
            m,
            meas,
            yerr=se,
            fmt="o",
            color=SERIES_MEASURED,
            markersize=8,
            capsize=3,
            linewidth=1.5,
            label="DSMC (AD)",
        )
        ax[i].set(xlabel=r"upstream Mach $M_1$", ylabel=lab + r"  [(m/s)$^{-1}$]")
        ax[i].set_title(f"{sym} ratio response to inlet velocity", fontsize=11)
        ax[i].legend(frameon=False, fontsize=8.5)

    z_r = np.abs(dr - dr_an) / np.maximum(dr_se, 1e-30)
    z_t = np.abs(dt - dt_an) / np.maximum(dt_se, 1e-30)
    w = 0.35
    xs = np.arange(len(m))
    ax[2].bar(xs - w / 2, z_r, width=w, color=SERIES_MEASURED, label="density ratio")
    ax[2].bar(xs + w / 2, z_t, width=w, color=SERIES_ANALYTIC, label="temperature ratio")
    ax[2].axhline(2.0, color=REFERENCE, linestyle="--", linewidth=1.5)
    ax[2].text(len(m) - 0.6, 2.15, r"$|z|=2$", fontsize=8.5, color=INK_SECONDARY, ha="right")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels([f"{x:.1f}" for x in m])
    ax[2].set(xlabel=r"upstream Mach $M_1$", ylabel=r"$|z|$ from the closed form")
    ax[2].set_title(
        f"agreement ({meta['seeds']} seeds, {meta['horizon']}-step horizon)", fontsize=11
    )
    ax[2].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    return savefig(fig, path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m examples.shock.gradients", description=__doc__)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--figures", default=FIGURES)
    args = ap.parse_args(argv)
    if not args.plot_only:
        print(run_sweep(seeds=args.seeds))
    meta, rows = parse()
    print("\n=== shock ratio gradients vs Rankine-Hugoniot ==========================")
    print(
        f"  {meta['n_cells']} cells, npc {meta['npc']}, {meta['seeds']} seeds, "
        f"{meta['horizon']}-step horizon\n"
    )
    print(
        f"  {'M1':>5} {'d(rho2/rho1)/du1':>18} {'+-SE':>10} {'analytic':>12} {'|z|':>6}"
        f" | {'d(T2/T1)/du1':>14} {'+-SE':>10} {'analytic':>12} {'|z|':>6}"
    )
    for r in rows:
        zr = abs(r[1] - r[3]) / max(r[2], 1e-30)
        zt = abs(r[4] - r[6]) / max(r[5], 1e-30)
        print(
            f"  {r[0]:>5.2f} {r[1]:>18.5e} {r[2]:>10.2e} {r[3]:>12.5e} {zr:>6.2f}"
            f" | {r[4]:>14.5e} {r[5]:>10.2e} {r[6]:>12.5e} {zt:>6.2f}"
        )
    os.makedirs(args.figures, exist_ok=True)
    print(f"\n  {figure(meta, rows, os.path.join(args.figures, 'gradients.png'))}")


if __name__ == "__main__":
    main()
