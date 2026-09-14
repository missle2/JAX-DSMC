"""M7 gradient validation: DSMC boundary derivatives against the Couette closed form.

    python3 -m examples.couette.gradients              # run, log and plot
    python3 -m examples.couette.gradients --plot-only  # re-plot from the stored log

Two analytic targets, both **profiles** rather than single numbers, so agreement has to
hold pointwise across the gap and cannot be tuned by one scalar:

1. ``d u_s(y) / d u_wall  =  (2y/L - 1) / (1 + 2 zeta lambda / L)``

   The continuum solution with first-order Maxwell slip, ``zeta = (2-acc)/acc x 1.1466``.
   Because the profile is linear in ``u_wall``, this derivative is a pure function of
   geometry -- it does not depend on ``u_wall`` itself, which makes it a sharp test.

2. ``d T(y) / d T_wall  =  1`` exactly, for every ``y``.

   ``T(y) = T_wall + mu (du/dy)^2 y(L-y) / (2k)``, and Eucken's relation
   ``k = (15/4)(k_B/m) mu`` makes ``mu/k = 4m/(15 k_B)`` independent of temperature.  The
   mean free path ``1/(sqrt2 pi d^2 n)`` carries no temperature dependence at fixed number
   density either, so the viscous rise -- 25.6 K on the centre line at ``u_wall = 200`` --
   does **not** move with ``T_wall``.  A profile of exact ones is a strong pointwise test
   precisely because it is so easy to violate: any spurious coupling between the wall
   temperature and the transport shows up immediately as a tilt.

Both parameters reach the objective **pathwise** (`sampling.sample_flux_normal` is
reparameterised in ``T_wall``; ``vwall`` is added after the draw), so these gradients carry
no score term and none of the attribution bias in `docs/ATTRIBUTION.md` applies to them.
That is why they can be validated pointwise at modest ensemble sizes.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

from examples.couette import (
    GAS,
    TEMP,
    Case,
    build,
    initial_state,
    mean_free_path,
    slip_coefficient,
)
from jax_dsmc.boundary import WallParams
from jax_dsmc.collide import vremax_init
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import num_cells
from jax_dsmc.simulate import run
from jax_dsmc.state import CollideState
from jax_dsmc.viz import (
    INK_SECONDARY,
    REFERENCE,
    SERIES_ANALYTIC,
    SERIES_ERROR,
    SERIES_MEASURED,
    make_figure,
    savefig,
)

RESULTS = "examples/couette/results"
FIGURES = "examples/couette/figures"

#: Cheapest case that still resolves a gap and bounces off both walls.
CASE = Case(dim=1, n_cells=12, npc=60, gap_mfp=5.0, u_wall=200.0, acc=1.0, a_max=64)
N_SLAB = 6  # coarse bins: the gradient profile needs samples per bin, not resolution
#: Trajectory length for the profile comparison.  Chosen from the horizon study below,
#: not by taste: long enough to be near steady state (0.69 of this case's 290-step
#: momentum-diffusion time, against the ~0.28 at which the shear rate settles) and short
#: enough that chaotic amplification has not yet destroyed the pathwise derivative.
HORIZON = 200
SEEDS = 512
HORIZONS = (10, 25, 50, 100, 200, 400, 800)


def analytic_du_dwall(y_over_l, case: Case = CASE):
    """``d u_s / d u_wall`` -- continuum Couette with first-order Maxwell slip."""
    lam = mean_free_path()
    length = case.gap_mfp * lam
    zeta = slip_coefficient(case.acc)
    return (2.0 * np.asarray(y_over_l) - 1.0) / (1.0 + 2.0 * zeta * lam / length)


def analytic_dT_dwall(y_over_l):
    """``d T / d T_wall`` -- exactly one everywhere.  See the module docstring."""
    return np.ones_like(np.asarray(y_over_l, dtype=float))


def viscous_rise(case: Case = CASE):
    """Centre-line temperature rise, for context on how large the effect being held
    constant actually is."""
    return (2.0 / 15.0) * (GAS.mass * case.u_wall**2 / K_BOLTZ)


def _profiles(theta, key, built, case, n_steps, n_slab):
    """Slab-binned ``(u_s, T)`` after warm-up, differentiable in ``theta``."""
    grid, sp, params, static, faces, _w, _lam, length, _fn = built
    static = dataclasses.replace(static, n_steps=n_steps)
    gap, stream = case.gap, case.stream
    n_face = 2 * case.dim
    vwall = jnp.zeros((n_face, 3))
    vwall = vwall.at[2 * gap, stream].set(-theta["u_wall"])
    vwall = vwall.at[2 * gap + 1, stream].set(theta["u_wall"])
    wall = WallParams(
        temp=jnp.full((n_face,), theta["T_wall"]),
        vwall=vwall,
        acc=jnp.full((n_face,), case.acc),
    )
    state = initial_state(key, grid, sp, static.N)
    c = num_cells(grid)
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((c,), TEMP),
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
        wall=wall,
        emit_moments="none",
    )
    # bin on the gap axis.  One snapshot at the end rather than a time average: the
    # gradient of a time average would need the whole trajectory differentiated, and the
    # ensemble over seeds supplies the averaging instead.
    dy = length / n_slab
    slab = jnp.clip((out.x[:, gap] / dy).astype(jnp.int32), 0, n_slab - 1)
    live = out.alive.astype(jnp.float64)
    count = jnp.zeros(n_slab).at[slab].add(live)
    safe = jnp.where(count > 0, count, 1.0)
    u_s = jnp.zeros(n_slab).at[slab].add(live * out.v[:, stream]) / safe
    sq = jnp.zeros(n_slab).at[slab].add(live * jnp.sum(out.v**2, axis=-1)) / safe
    mean_v = jnp.stack(
        [jnp.zeros(n_slab).at[slab].add(live * out.v[:, c_]) / safe for c_ in range(3)]
    )
    temp = GAS.mass * (sq - jnp.sum(mean_v**2, axis=0)) / (3.0 * K_BOLTZ)
    return u_s, temp


def measure(case=CASE, n_slab=N_SLAB, seeds=SEEDS, n_steps=None):
    """Jacobian of the binned profiles with respect to the two wall parameters."""
    built = build(case)
    n_steps = HORIZON if n_steps is None else n_steps
    theta = {"u_wall": float(case.u_wall), "T_wall": float(TEMP)}

    def jac(key):
        return jax.jacrev(
            lambda t: jnp.concatenate(_profiles(t, key, built, case, n_steps, n_slab))
        )(theta)

    keys = jax.random.split(jax.random.PRNGKey(0), seeds)
    out = jax.jit(jax.vmap(jac))(keys)
    got = {k: np.asarray(v) for k, v in out.items()}
    return {
        "du_dwall": got["u_wall"][:, :n_slab],
        "dT_dTwall": got["T_wall"][:, n_slab:],
        "du_dTwall": got["T_wall"][:, :n_slab],
        "y": (np.arange(n_slab) + 0.5) / n_slab,
    }


def write_log(res, case=CASE, path=None):
    os.makedirs(RESULTS, exist_ok=True)
    path = path or os.path.join(RESULTS, "gradients.log")
    y = res["y"]
    lines = [
        f"# case dim={case.dim} n_cells={case.n_cells} npc={case.npc} "
        f"gap_mfp={case.gap_mfp} acc={case.acc} u_wall={case.u_wall} "
        f"seeds={res['du_dwall'].shape[0]} horizon={HORIZON}",
        f"# viscous_rise_K={viscous_rise(case):.4f}",
        "# y_over_L du_dwall du_dwall_se du_dwall_analytic dT_dTwall dT_dTwall_se "
        "dT_dTwall_analytic du_dTwall du_dTwall_se",
    ]
    n = res["du_dwall"].shape[0]
    for i, yi in enumerate(y):
        a = res["du_dwall"][:, i]
        b = res["dT_dTwall"][:, i]
        c = res["du_dTwall"][:, i]
        lines.append(
            f"{yi:.6f} {a.mean():.8e} {a.std(ddof=1) / math.sqrt(n):.8e} "
            f"{analytic_du_dwall(yi, case):.8e} {b.mean():.8e} "
            f"{b.std(ddof=1) / math.sqrt(n):.8e} 1.0 "
            f"{c.mean():.8e} {c.std(ddof=1) / math.sqrt(n):.8e}"
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def parse(path=None):
    path = path or os.path.join(RESULTS, "gradients.log")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run `python3 -m examples.couette.gradients`")
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
    if not rows:
        raise ValueError(f"{path}: no rows")
    return meta, np.array(rows)


def horizon_study(case=CASE, n_slab=N_SLAB, seeds=128, horizons=HORIZONS):
    """Gradient mean and spread against trajectory length -- where chaos takes over.

    **This is the convergence result that matters, and it is not a `1/sqrt(n)` plot.**
    Both parameters are pathwise, so each seed's gradient is an exact derivative of that
    realisation; more seeds always reduce the error on the mean.  What more seeds cannot
    fix is that DSMC is chaotic: the derivative of a *single* trajectory grows like
    `exp(lambda t)` while the quantity it estimates stays O(1), so the per-seed spread
    explodes with the horizon and the estimator becomes unusable long before it becomes
    wrong.

    Measured on this case, against an analytic first-slab value of -0.571: the per-seed
    spread is 0.04 at 50 steps, 0.8 at 200, and **341 at 800**.
    """
    built = build(case)
    theta = {"u_wall": float(case.u_wall), "T_wall": float(TEMP)}
    keys = jax.random.split(jax.random.PRNGKey(0), seeds)
    rows = []
    for m in horizons:
        g = jax.jit(
            jax.vmap(
                lambda k, mm=m: jax.grad(lambda t: _profiles(t, k, built, case, mm, n_slab)[0][0])(
                    theta
                )["u_wall"]
            )
        )(keys)
        a = np.asarray(g)
        rows.append(
            (m, float(a.mean()), float(a.std(ddof=1) / math.sqrt(seeds)), float(a.std(ddof=1)))
        )
    return np.array(rows)


def seed_convergence(case=CASE, n_slab=N_SLAB, seeds=SEEDS, n_steps=None):
    """SE against ensemble size at the chosen horizon -- the ordinary `1/sqrt(n)` half."""
    res = measure(case, n_slab, seeds, n_steps)
    out = {}
    for name, col in (("du_dwall", 0), ("dT_dTwall", n_slab // 2)):
        a = res[name][:, col]
        ns, ses = [], []
        n = 8
        while n <= seeds:
            ns.append(n)
            ses.append(float(a[:n].std(ddof=1) / math.sqrt(n)))
            n *= 2
        out[name] = {"n": np.array(ns), "se": np.array(ses)}
    return res, out


def figure(meta, rows, conv, path):
    """Measured boundary gradients against the closed form, and their convergence."""
    y = rows[:, 0]
    du, du_se, du_an = rows[:, 1], rows[:, 2], rows[:, 3]
    dt, dt_se = rows[:, 4], rows[:, 5]
    dutw, dutw_se = rows[:, 7], rows[:, 8]

    fig, ax = make_figure(1, 3, figsize=(16.5, 4.6))

    grid = np.linspace(0.0, 1.0, 200)
    ax[0].plot(
        grid,
        analytic_du_dwall(grid),
        "-",
        color=SERIES_ANALYTIC,
        linewidth=2.0,
        label=r"continuum + Maxwell slip",
    )
    ax[0].errorbar(
        y,
        du,
        yerr=du_se,
        fmt="o",
        color=SERIES_MEASURED,
        markersize=8,
        capsize=3,
        linewidth=1.5,
        label="DSMC (AD, pathwise)",
    )
    ax[0].axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
    ax[0].set(xlabel="$y/L$", ylabel=r"$\partial u_s / \partial u_{\rm wall}$")
    worst = np.max(np.abs(du - du_an))
    ax[0].set_title(f"velocity response to wall speed\nworst deviation {worst:.4f}", fontsize=10.5)
    ax[0].legend(frameon=False, fontsize=8.5)

    ax[1].axhline(1.0, color=SERIES_ANALYTIC, linewidth=2.0, label="analytic: exactly 1")
    ax[1].errorbar(
        y,
        dt,
        yerr=dt_se,
        fmt="o",
        color=SERIES_MEASURED,
        markersize=8,
        capsize=3,
        linewidth=1.5,
        label="DSMC (AD, pathwise)",
    )
    ax[1].errorbar(
        y,
        dutw,
        yerr=dutw_se,
        fmt="s",
        color=SERIES_ERROR,
        markersize=6,
        capsize=3,
        linewidth=1.2,
        markerfacecolor="none",
        label=r"$\partial u_s/\partial T_{\rm wall}$ (must be $\approx 0$)",
    )
    ax[1].axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
    dev = np.max(np.abs(dt - 1.0))
    ax[1].set(xlabel="$y/L$", ylabel=r"$\partial (\cdot) / \partial T_{\rm wall}$")
    ax[1].set_title(
        f"temperature response to wall temperature\nworst deviation from 1: {dev:.2e}",
        fontsize=10.5,
    )
    ax[1].legend(frameon=False, fontsize=8)

    h = conv["horizon"]
    m, se, sd = h[:, 0], h[:, 2], h[:, 3]
    target = abs(analytic_du_dwall(0.5 / len(y), CASE))
    ax[2].loglog(
        m, sd, "o-", color=SERIES_ERROR, markersize=7, linewidth=1.9, label="per-seed spread"
    )
    ax[2].loglog(
        m,
        se,
        "s-",
        color=SERIES_MEASURED,
        markersize=6,
        linewidth=1.5,
        label=f"SE over {int(conv['seeds'])} seeds",
    )
    ax[2].axhline(
        target, color=SERIES_ANALYTIC, linestyle="--", linewidth=1.6, label=r"$|$analytic value$|$"
    )
    ax[2].axvspan(m.min() * 0.8, HORIZON, color=SERIES_ANALYTIC, alpha=0.10)
    ax[2].annotate(
        "usable",
        xy=(math.sqrt(m.min() * HORIZON), sd.max() * 0.2),
        fontsize=9,
        color=INK_SECONDARY,
        ha="center",
    )
    ax[2].set(
        xlabel="trajectory length (steps)", ylabel=r"spread of $\partial u_s/\partial u_{\rm wall}$"
    )
    ax[2].set_title(
        "chaos, not sampling, sets the horizon\n"
        f"spread grows {sd[-1] / sd[0]:.0f}$\\times$ over "
        f"{int(m[-1] / m[0])}$\\times$ the trajectory",
        fontsize=10.5,
    )
    ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return savefig(fig, path)


def _run_scaling(args):
    """The N-scaling sweep: forward noise against gradient noise, at two horizons."""
    long_r = scaling_study(seeds=args.seeds)
    short_r = scaling_study(seeds=args.seeds, n_steps=25)
    print(write_scaling_log(long_r, short_r))
    long_r, short_r = parse_scaling()
    os.makedirs(args.figures, exist_ok=True)
    n = long_r[:, 0]
    print(f"\n  {'quantity':<28} {'N^p (25 steps)':>16} {'N^p (200 steps)':>17}")
    for label, (num, den) in (
        ("forward u_s", (2, 1)),
        ("d u_s/d u_wall", (4, 3)),
        ("forward T", (6, 5)),
        ("d T/d T_wall", (8, 7)),
    ):
        ps = [
            np.polyfit(np.log(n), np.log(np.abs(r[:, num] / r[:, den])), 1)[0]
            for r in (short_r, long_r)
        ]
        print(f"  {label:<28} {ps[0]:>16.3f} {ps[1]:>17.3f}")
    print(
        f"\n  {figure_scaling(long_r, short_r, os.path.join(args.figures, 'gradient_scaling.png'))}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m examples.couette.gradients", description=__doc__)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--horizon-seeds", type=int, default=128)
    ap.add_argument(
        "--scaling",
        action="store_true",
        help="run the N-scaling sweep (forward vs gradient) and plot it",
    )
    ap.add_argument("--figures", default=FIGURES)
    args = ap.parse_args(argv)

    if args.scaling:
        _run_scaling(args)
        return

    if args.plot_only:
        meta, rows = parse()
    else:
        res = measure(seeds=args.seeds)
        print(write_log(res))
        meta, rows = parse()
    conv = {"horizon": horizon_study(seeds=args.horizon_seeds), "seeds": args.horizon_seeds}

    print("\n=== M7 boundary gradients vs the Couette closed form ====================")
    print(
        f"  {CASE.n_cells} cells, npc {CASE.npc}, gap {CASE.gap_mfp} mfp, "
        f"acc {CASE.acc}, {meta['seeds']} seeds"
    )
    print(
        f"  centre-line viscous rise held constant by the T_wall test: "
        f"{float(meta['viscous_rise_K']):.1f} K\n"
    )
    print(
        f"  {'y/L':>6} {'du/du_wall':>13} {'+-SE':>10} {'analytic':>11} {'|z|':>6}"
        f" | {'dT/dT_wall':>11} {'+-SE':>10} {'|z| vs 1':>9}"
    )
    for r in rows:
        z1 = abs(r[1] - r[3]) / max(r[2], 1e-30)
        z2 = abs(r[4] - 1.0) / max(r[5], 1e-30)
        print(
            f"  {r[0]:>6.3f} {r[1]:>13.5f} {r[2]:>10.2e} {r[3]:>11.5f} {z1:>6.2f}"
            f" | {r[4]:>11.6f} {r[5]:>10.2e} {z2:>9.2f}"
        )
    print(f"\n  worst |du/du_wall - analytic| = {np.max(np.abs(rows[:, 1] - rows[:, 3])):.5f}")
    print(f"  worst |dT/dT_wall - 1|        = {np.max(np.abs(rows[:, 4] - 1.0)):.3e}")
    print(f"  worst |du/dT_wall| (expect ~0) = {np.max(np.abs(rows[:, 7])):.3e}")

    print("\n  horizon study -- where chaos takes the pathwise gradient away:")
    print(f"    {'steps':>7} {'mean':>11} {'+- SE':>10} {'per-seed sd':>12} {'vs analytic':>12}")
    tgt = float(analytic_du_dwall(0.5 / N_SLAB, CASE))
    for r in conv["horizon"]:
        print(
            f"    {int(r[0]):>7d} {r[1]:>11.4f} {r[2]:>10.2e} {r[3]:>12.3e} "
            f"{abs(r[1] - tgt) / max(r[2], 1e-30):>11.2f}z"
        )
    print(f"    (analytic first-slab value {tgt:+.5f})")

    os.makedirs(args.figures, exist_ok=True)
    print(f"\n  {figure(meta, rows, conv, os.path.join(args.figures, 'gradients.png'))}")


# --- how does gradient variance scale, against the forward solve's? -------------------

SCALE_NPC = (15, 30, 60, 120, 240)


def scaling_study(npcs=SCALE_NPC, case=CASE, seeds=128, n_steps=HORIZON, n_slab=N_SLAB):
    """Per-seed spread of forward observables and of their gradients, against ``N``.

    The horizon is **held fixed**, so this isolates particle-count scaling from the
    trajectory-length effect `horizon_study` measures separately; confounding the two
    makes both exponents meaningless.

    Each parameter is paired with the observable it actually drives -- ``u_wall`` with the
    streamwise velocity, ``T_wall`` with the temperature.  Pairing ``T_wall`` with the
    velocity instead would divide by a mean that is *supposed* to be zero and report the
    ratio of two noises, which an earlier version of this study did.

    Returns columns ``(N, u_mean, u_sd, du_mean, du_sd, T_mean, T_sd, dT_mean, dT_sd)``.
    """
    rows = []
    for npc in npcs:
        c = dataclasses.replace(case, npc=npc)
        built = build(c)
        theta = {"u_wall": float(c.u_wall), "T_wall": float(TEMP)}
        keys = jax.random.split(jax.random.PRNGKey(0), seeds)
        n_part = c.npc * c.n_cells

        def both(k, c=c, built=built, theta=theta):
            u, t = _profiles(theta, k, built, c, n_steps, n_slab)
            return jnp.stack([u[0], t[0]])

        def grads(k, c=c, built=built, theta=theta):
            du = jax.grad(lambda th: _profiles(th, k, built, c, n_steps, n_slab)[0][0])(theta)[
                "u_wall"
            ]
            dt = jax.grad(lambda th: _profiles(th, k, built, c, n_steps, n_slab)[1][0])(theta)[
                "T_wall"
            ]
            return jnp.stack([du, dt])

        fwd, grad = jax.jit(jax.vmap(both)), jax.jit(jax.vmap(grads))
        # the reverse pass holds a tape per particle-key; chunk rather than OOM
        chunk = max(1, min(seeds, int(1.2e9 / (4290 * n_part))))
        j = np.concatenate([np.asarray(fwd(keys[i : i + chunk])) for i in range(0, seeds, chunk)])
        g = np.concatenate([np.asarray(grad(keys[i : i + chunk])) for i in range(0, seeds, chunk)])
        rows.append(
            (
                n_part,
                j[:, 0].mean(),
                j[:, 0].std(ddof=1),
                g[:, 0].mean(),
                g[:, 0].std(ddof=1),
                j[:, 1].mean(),
                j[:, 1].std(ddof=1),
                g[:, 1].mean(),
                g[:, 1].std(ddof=1),
            )
        )
        print(f"    N = {n_part:>6d} done", flush=True)
    return np.array(rows)


def figure_scaling(rows_long, rows_short, path, horizons=(HORIZON, 25)):
    """Why particle count stops helping the gradient, and when.

    A gradient's variance has two parts that scale differently:

    * a **sampling** part, which averages over particles exactly like the forward
      observable does and falls as ``N^-1/2``;
    * a **chaotic** part, set by the trajectory's Lyapunov instability, which does not fall
      with ``N`` at all -- more particles add degrees of freedom, not averaging, to a
      *derivative* that diverges coherently.

    The two panels are the same sweep at two horizons, and the contrast is the result:
    at 25 steps every curve is ``N^-1/2``; at 200 steps the forward curves still are and
    the gradient curves are **flat**.  The practical rule is that particles buy gradient
    accuracy only up to the horizon where chaos takes over -- past that, only ensemble size
    does, at ``M^-1/2``.

    Note the short-horizon gradient is *precise and not yet converged*: ``du_s/du_wall``
    reads -0.46 there against an analytic -0.571, because 25 steps is well short of this
    case's 290-step momentum-diffusion time.  Low variance is not the same as correct.
    """
    labels = (
        "forward $u_s$",
        r"$\partial u_s/\partial u_{\rm wall}$",
        "forward $T$",
        r"$\partial T/\partial T_{\rm wall}$",
    )
    cols = ((2, 1), (4, 3), (6, 5), (8, 7))
    colours = (INK_SECONDARY, SERIES_MEASURED, SERIES_ANALYTIC, SERIES_ERROR)
    marks = ("o-", "s-", "o--", "s--")

    fig, ax = make_figure(1, 2, figsize=(12.6, 4.8))
    for k, rows in enumerate((rows_short, rows_long)):
        n = rows[:, 0]
        for label, (num, den), colour, mk in zip(labels, cols, colours, marks, strict=True):
            y = np.abs(rows[:, num] / rows[:, den])
            p_fit = np.polyfit(np.log(n), np.log(y), 1)[0]
            ax[k].loglog(
                n,
                100 * y,
                mk,
                color=colour,
                markersize=7,
                linewidth=1.9,
                label=f"{label}   $p={p_fit:+.2f}$",
            )
        ref = np.abs(rows[0, 2] / rows[0, 1])
        ax[k].loglog(
            n,
            100 * ref * (n / n[0]) ** -0.5,
            ":",
            color=REFERENCE,
            linewidth=1.6,
            label=r"$N^{-1/2}$",
        )
        ax[k].set(xlabel="particles $N$", ylabel="relative per-seed spread  [%]")
        ax[k].legend(frameon=False, fontsize=7.5)
    ax[0].set_title(
        f"{horizons[1]} steps: sampling-dominated\neverything gets "
        r"$N^{-1/2}$",
        fontsize=11,
    )
    ax[1].set_title(
        f"{horizons[0]} steps: chaos-dominated\ngradients go flat, forward does not", fontsize=11
    )
    fig.tight_layout()
    return savefig(fig, path)


def write_scaling_log(rows_long, rows_short, horizons=(HORIZON, 25), path=None):
    """Both sweeps to one parseable log -- the figure is rebuilt from this, not from RAM."""
    os.makedirs(RESULTS, exist_ok=True)
    path = path or os.path.join(RESULTS, "gradient_scaling.log")
    lines = [
        f"# horizons long={horizons[0]} short={horizons[1]} npcs={','.join(map(str, SCALE_NPC))}",
        "# horizon N u_mean u_sd du_mean du_sd T_mean T_sd dT_mean dT_sd",
    ]
    for steps, rows in ((horizons[0], rows_long), (horizons[1], rows_short)):
        for r in rows:
            lines.append(f"{steps} " + " ".join(f"{v:.8e}" for v in r))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def parse_scaling(path=None, horizons=(HORIZON, 25)):
    path = path or os.path.join(RESULTS, "gradient_scaling.log")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run `python3 -m examples.couette.gradients --scaling`")
    by = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            v = [float(t) for t in line.split()]
            by.setdefault(int(v[0]), []).append(v[1:])
    return np.array(by[horizons[0]]), np.array(by[horizons[1]])


if __name__ == "__main__":
    main()
