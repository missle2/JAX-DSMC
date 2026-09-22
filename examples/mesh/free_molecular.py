"""FG5: free-molecular drag of the analytic circle / sphere vs the closed-form reference.

The Schaaf-Chambre / Bird (ch. 7) element fluxes of a drifting Maxwellian -- number flux,
incident normal momentum flux and shear -- are integrated numerically over the body for a
specular surface (``p = 2 p_i``, no shear) and a fully diffuse one at ``T_w`` (``p = p_i +
n_dot m sqrt(pi k T_w / 2m)``, ``tau = tau_i``).  The stream the inlet produces is the
reservoir Maxwellian restricted to ``v_x > 0`` (a collisionless half-space: density
``n P(v_x > 0)``, mean speed above ``u``), so the reference actually compared against is the
same surface integral under that half-range distribution (Monte Carlo quadrature,
:func:`reference_drag_halfrange`); the full-Maxwellian closed form (which reproduces Bird's
specular-sphere ``C_D`` exactly) is recorded alongside.  The box blocks part of the stream
(reflected particles leave through the inlet), so the body is run at three radii in the
same box and the drag coefficient is extrapolated linearly in ``2r / L`` to zero blockage.

    python3 examples/mesh/free_molecular.py --case circle --radii 1.5 0.75 0.375 --steps 4000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import erf

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from forward import BLOCK, CASES, DT, FIGURES, NRHO, RESULTS, TEMP, VSTREAM, build  # noqa: E402

from jax_dsmc import initialize, simulate, viz  # noqa: E402
from jax_dsmc import species_data as sd  # noqa: E402
from jax_dsmc.state import FlowParams, ParticleState  # noqa: E402

K_BOLTZ = 1.380649e-23


def element_fluxes(s_n, s_t, n, mass, temp):
    """Per unit area of a drifting Maxwellian on a surface element (Bird ch. 7).

    ``s_n``, ``s_t``: speed ratios of the stream along the inward normal and the tangent
    (``beta = 1 / sqrt(2 k T / m)``).  Returns ``(number flux, incident normal momentum
    flux, incident shear)``.
    """
    beta = 1.0 / math.sqrt(2.0 * K_BOLTZ * temp / mass)
    rho = n * mass
    chi = np.exp(-(s_n**2)) + math.sqrt(math.pi) * s_n * (1.0 + erf(s_n))
    ndot = n / (2.0 * math.sqrt(math.pi) * beta) * chi
    p_i = (
        rho
        / (2.0 * beta**2)
        * (s_n / math.sqrt(math.pi) * np.exp(-(s_n**2)) + (0.5 + s_n**2) * (1.0 + erf(s_n)))
    )
    tau_i = rho * s_t / (2.0 * math.sqrt(math.pi) * beta**2) * chi
    return ndot, p_i, tau_i


def reference_drag(dim, radius, n, mass, temp, u, t_wall, diffuse, m=4000):
    """Drag force on a circle (per unit depth) or sphere in free-molecular flow."""
    beta = 1.0 / math.sqrt(2.0 * K_BOLTZ * temp / mass)
    s = u * beta
    if dim == 2:
        theta = (np.arange(m) + 0.5) * 2 * math.pi / m
        nx, ny = np.cos(theta), np.sin(theta)  # outward normal
        w = np.full(m, 2 * math.pi * radius / m)
    else:
        theta = (np.arange(m) + 0.5) * math.pi / m
        nx, ny = np.cos(theta), np.sin(theta)
        w = 2 * math.pi * radius**2 * np.sin(theta) * math.pi / m
    s_n = -s * nx  # inward component of the stream
    s_t = s * np.abs(ny)  # tangential magnitude
    ndot, p_i, tau_i = element_fluxes(s_n, s_t, n, mass, temp)
    if diffuse:
        sigma_w = math.sqrt(K_BOLTZ * t_wall / mass)
        p = p_i + ndot * mass * sigma_w * math.sqrt(math.pi / 2.0)
        tau = tau_i
    else:
        p = 2.0 * p_i
        tau = np.zeros_like(p_i)
    # pressure pushes along -n, shear along the tangential stream direction (x - (x.n) n)
    tx = 1.0 - nx * nx
    t_norm = np.sqrt(np.maximum(tx**2 + (-nx * ny) ** 2, 1e-300))
    fx = (p * (-nx) + tau * tx / t_norm) * w
    return float(fx.sum())


def reference_drag_halfrange(
    dim, radius, n, mass, temp, u, t_wall, diffuse, samples=4_000_000, seed=0
):
    """Drag in the stream the inlet actually produces: the reservoir Maxwellian restricted to
    ``v_x > 0`` (collisionless half-space fed by ``emit/face xlo``: density ``n P(v_x > 0)``,
    mean speed above ``u``).  Monte Carlo quadrature over the velocity, exact element
    integrals over the body.  Same surface laws as :func:`reference_drag`.
    """
    rng = np.random.default_rng(seed)
    sigma = math.sqrt(K_BOLTZ * temp / mass)
    v = rng.normal(size=(samples, 3)) * sigma
    v[:, 0] += u
    keep = v[:, 0] > 0.0
    v = v[keep]
    weight = n / samples  # each kept sample carries n / N_total of the density
    m = 720
    if dim == 2:
        theta = (np.arange(m) + 0.5) * 2 * math.pi / m
        normals = np.stack([np.cos(theta), np.sin(theta), np.zeros(m)], axis=1)
        w = np.full(m, 2 * math.pi * radius / m)
    else:
        theta = (np.arange(m) + 0.5) * math.pi / m
        normals = np.stack([np.cos(theta), np.sin(theta), np.zeros(m)], axis=1)
        w = 2 * math.pi * radius**2 * np.sin(theta) * math.pi / m
    fx = 0.0
    sigma_w = math.sqrt(K_BOLTZ * t_wall / mass)
    for k in range(m):
        n_out = normals[k]
        vn = -(v @ n_out)  # inward normal speed
        inc = vn > 0.0
        flux = weight * vn[inc]  # number flux per area contributions
        p_i = mass * np.sum(flux * vn[inc])  # incident normal momentum flux
        if diffuse:
            ndot = np.sum(flux)
            p = p_i + ndot * mass * sigma_w * math.sqrt(math.pi / 2.0)
            tau_x = mass * np.sum(flux * (v[inc, 0] + vn[inc] * n_out[0]))  # tangential x-momentum
            fx += (p * (-n_out[0]) + tau_x) * w[k]
        else:
            fx += 2.0 * p_i * (-n_out[0]) * w[k]
    return float(fx)


def drag_coefficient(force, dim, radius, n, mass, u):
    rho = n * mass
    ref_area = 2 * radius if dim == 2 else math.pi * radius**2
    return force / (0.5 * rho * u**2 * ref_area)


def _halfrange_start(key, grid, sp, static, mesh):
    """The steady collisionless stream, ``n M(v) 1[v_x > 0]`` outside the body, at t = 0.

    Filling from empty is not an option here: the density of the slow part of the stream
    (``v_x ~ 10 m/s``, several per cent of the half-range density) builds up over seconds,
    and the hit rate on a small body was 7 % low at 0.4 s.  Starting from the steady
    distribution leaves only the body's shadow to settle.
    """
    D = static.D
    lo = np.asarray(grid.lo)
    hi = np.asarray(grid.hi)
    volume = float(np.prod(hi - lo))
    fnum = CASES["circle" if D == 2 else "sphere"]["fnum"]
    n_full = round(NRHO * volume / fnum)  # count the full Maxwellian would have
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    sigma = math.sqrt(K_BOLTZ * TEMP / float(sp.mass[0]))
    v = rng.normal(size=(n_full, 3)) * sigma
    v[:, 0] += VSTREAM
    v = v[v[:, 0] > 0.0]
    x = lo + rng.random((v.shape[0], D)) * (hi - lo)
    c = np.asarray(mesh.body.centre)
    outside = np.sum((x - c) ** 2, axis=1) > float(mesh.body.radius) ** 2
    x, v = x[outside], v[outside]
    n = x.shape[0]
    if n > static.N:
        raise ValueError(f"capacity {static.N} < steady count {n}")
    alive = np.zeros(static.N, bool)
    alive[:n] = True
    xs = np.broadcast_to(lo, (static.N, D)).copy()
    xs[:n] = x
    vs = np.zeros((static.N, 3))
    vs[:n] = v
    return ParticleState(
        x=jnp.asarray(xs),
        v=jnp.asarray(vs),
        species=jnp.zeros(static.N, jnp.int32),
        alive=jnp.asarray(alive),
        logh_acc=jnp.zeros(static.N),
    )


def run_case(case, radius, acc, steps, window, seed, tracker):
    grid, mesh, sp, params, static, faces, res, wall, _target, capacity = build(
        case, "analytic", tracker, "Ar", acc, 1, body_radius=radius
    )
    # the steady half-range stream holds ~66 k (circle) / ~44 k (sphere) particles; room for
    # the fluctuations and the reflected population
    static = dataclasses.replace(
        static,
        N=int(
            1.3
            * NRHO
            * float(np.prod(np.asarray(grid.hi) - np.asarray(grid.lo)))
            / CASES[case]["fnum"]
        ),
    )
    key = jax.random.PRNGKey(seed)
    flow = FlowParams(
        nrho=jnp.float64(NRHO), temp=jnp.float64(TEMP), vstream=jnp.array([VSTREAM, 0.0, 0.0])
    )
    state = _halfrange_start(key, grid, sp, static, mesh)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    step = jax.jit(
        lambda s, c, k0: simulate.run(
            s,
            c,
            grid,
            sp,
            params,
            static,
            key,
            faces=faces,
            wall=wall,
            res=res,
            capacity=capacity,
            emit_moments=False,
            step0=k0,
            mesh=mesh,
            emit_surface=True,
        )
    )
    D = static.D
    fx_blocks = []
    hits = 0
    bad = 0
    t0 = time.time()
    for k0 in range(0, steps, BLOCK):
        (state, cstate), (_, diag, tallies) = step(state, cstate, k0)
        bad += int(
            diag.stencil_overrun.sum()
            + diag.inside_body.sum()
            + diag.walk_exhausted.sum()
            + diag.wall_recrossed.sum()
        )
        if k0 >= steps - window:
            fx_blocks.append(float(np.asarray(tallies.momentum)[:, 2 * D, 0].sum()))
            hits += int(np.asarray(tallies.hits)[:, 2 * D].sum())
    fnum = CASES[case]["fnum"]
    blocks = np.asarray(fx_blocks) * fnum / (BLOCK * DT)  # force per block
    # blocks are correlated over a flight time; batch them into 10 batches for the SE
    batches = np.array_split(blocks, 10)
    means = np.asarray([b.mean() for b in batches])
    force = float(blocks.mean())
    se = float(means.std(ddof=1) / math.sqrt(len(means)))
    return {
        "case": case,
        "radius": radius,
        "acc": acc,
        "steps": steps,
        "window": window,
        "seed": seed,
        "force": force,
        "force_se": se,
        "hits_per_step": hits / window,
        "n_alive": int(jnp.sum(state.alive)),
        "diagnostics_nonzero": bad,
        "seconds": time.time() - t0,
        "blockage": radius / (float(grid.hi[1]) - float(grid.lo[1])) * 2,
    }


def main(argv=None) -> int:  # noqa: PLR0915 -- one sweep, kept whole
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", choices=("circle", "sphere"), default="circle")
    ap.add_argument("--radii", type=float, nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--window", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tracker", default="stencil")
    ap.add_argument("--summarize-only", action="store_true", help="re-summarise the saved rows")
    args = ap.parse_args(argv)
    dim = CASES[args.case]["dim"]
    radii = args.radii or ([1.5, 0.75, 0.375] if dim == 2 else [0.5, 0.25])
    gas = sd.BY_SYMBOL["Ar"]
    rows = []
    path = os.path.join(RESULTS, f"free_molecular_{args.case}.json")
    if args.summarize_only:
        with open(path) as fh:
            rows = json.load(fh)["rows"]
        for row in rows:
            ref = reference_drag_halfrange(
                dim, row["radius"], NRHO, gas.mass, TEMP, VSTREAM, 300.0, row["acc"] > 0
            )
            row["cd_reference"] = drag_coefficient(ref, dim, row["radius"], NRHO, gas.mass, VSTREAM)
    for acc in (0.0, 1.0) if not args.summarize_only else ():
        for r in radii:
            row = run_case(args.case, r, acc, args.steps, args.window, args.seed, args.tracker)
            row["cd"] = drag_coefficient(row["force"], dim, r, NRHO, gas.mass, VSTREAM)
            row["cd_se"] = drag_coefficient(row["force_se"], dim, r, NRHO, gas.mass, VSTREAM)
            ref = reference_drag_halfrange(dim, r, NRHO, gas.mass, TEMP, VSTREAM, 300.0, acc > 0)
            row["cd_reference"] = drag_coefficient(ref, dim, r, NRHO, gas.mass, VSTREAM)
            full = reference_drag(dim, r, NRHO, gas.mass, TEMP, VSTREAM, 300.0, acc > 0)
            row["cd_reference_full_maxwellian"] = drag_coefficient(
                full, dim, r, NRHO, gas.mass, VSTREAM
            )
            rows.append(row)
            print(
                f"{args.case} r={r} acc={acc}: CD {row['cd']:.4f} ± {row['cd_se']:.4f} (ref {row['cd_reference']:.4f}), "
                f"hits/step {row['hits_per_step']:.1f}, bad {row['diagnostics_nonzero']}, {row['seconds']:.0f} s",
                flush=True,
            )
    summary = {}
    for acc in (0.0, 1.0):
        sel = [r for r in rows if r["acc"] == acc]
        b = np.asarray([r["blockage"] for r in sel])
        cd = np.asarray([r["cd"] for r in sel])
        se = np.asarray([r["cd_se"] for r in sel])
        # weighted linear extrapolation in the blockage ratio to zero
        wts = 1.0 / se**2
        A = np.stack([np.ones_like(b), b], axis=1)
        cov = np.linalg.inv(A.T @ (wts[:, None] * A))
        coef = cov @ (A.T @ (wts * cd))
        cd0, cd0_se = float(coef[0]), float(math.sqrt(cov[0, 0]))
        ref = sel[0]["cd_reference"]
        summary[f"acc{acc:g}"] = {
            "cd_extrapolated": cd0,
            "cd_se": cd0_se,
            "cd_reference": ref,
            "z": (cd0 - ref) / cd0_se,
            "slope": float(coef[1]),
        }
        print(
            f"acc={acc}: CD(0 blockage) = {cd0:.4f} ± {cd0_se:.4f}, reference {ref:.4f}, z = {(cd0 - ref) / cd0_se:.2f}"
        )
    out = {"case": args.case, "rows": rows, "summary": summary, "backend": jax.default_backend()}
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    figure(out, os.path.join(FIGURES, f"free_molecular_drag_{args.case}.png"))
    print("wrote", path)
    return 0


def figure(out, path):
    fig, (ax,) = viz.make_figure(1, 1)
    for acc, color, label in (
        (0.0, viz.SERIES_ANALYTIC, "specular"),
        (1.0, viz.SERIES_MEASURED, "diffuse, T_w = 300 K"),
    ):
        sel = [r for r in out["rows"] if r["acc"] == acc]
        b = [r["blockage"] for r in sel]
        ax.errorbar(
            b,
            [r["cd"] for r in sel],
            yerr=[r["cd_se"] for r in sel],
            fmt="o",
            color=color,
            label=label,
        )
        s = out["summary"][f"acc{acc:g}"]
        bb = np.linspace(0, max(b), 20)
        ax.plot(bb, s["cd_extrapolated"] + s["slope"] * bb, color=color, lw=1)
        ax.axhline(
            s["cd_reference"],
            color=color,
            ls="--",
            lw=1,
            label=f"free-molecular reference ({label})",
        )
    ax.set_xlabel("blockage 2r / L")
    ax.set_ylabel("C_D")
    ax.legend(fontsize=8)
    ax.set_title(
        f"{out['case']}: drag vs blockage, extrapolated to an unbounded stream", fontsize=10
    )
    viz.savefig(fig, path)


if __name__ == "__main__":
    sys.exit(main())
