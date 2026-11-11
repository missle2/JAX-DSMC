"""Fine-cell reference data for training coarse-cell collision operators (plan step 6).

A fine NTC run (``dx = 0.25 lambda``, many particles per cell: the converged truth) of a
periodic wave from :mod:`.shear_wave`, batched over seeds, records at every sample time
(``BLOCK`` steps, one mean collision time):

* the seed mean and seed variance of the coarse-cell moments (:mod:`.moments`, unbiased
  small-sample factors) on each coarse grid ``dx_c`` -- the target statistics a coarse run
  is trained to reproduce (fluxes ``pxy``, ``qx`` above all);
* full particle snapshots ``(x, v)`` of the first ``--snap-seeds`` seeds every
  ``--snap-every`` samples -- initial states of the coarse training windows, which thin
  them (Poisson thinning is unbiased in ``f``) to the coarse particle count.

Units: ``n`` relative to the mean count per coarse cell, velocities in ``v_th``, ``p`` in
``v_th^2``, ``q`` in ``v_th^3``; ``t`` in ``lambda / v_th``.

    PYTHONPATH=. python3 -m examples.learned_collision.reference --wave shear \
        [--length-mfp 64] [--seeds 64] [--coarse 1 2 4] [--snap-seeds 16] [--snap-every 10]

Writes ``examples/learned_collision/results/reference_<wave>_L<L>_<tag>.npz``.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples.learned_collision import moments
from examples.learned_collision.shear_wave import (
    BLOCK,
    LENGTH_MFP,
    RATE_CE,
    RESULTS,
    _stack,
    advance,
    operator,
    setup,
)


def _coarse(states, lo, hi, n_coarse, scale):
    """``f64[B, C, 13]`` nondimensional coarse moments of a batch of states."""
    m = jax.vmap(functools.partial(moments.coarse_moments, lo=lo, hi=hi, n_coarse=n_coarse))(states)
    return m * scale


def _scale(n_mean, vth):
    """Per-field divisors -> multipliers: count / mean count, v / v_th, p / v_th^2,
    q / v_th^3."""
    return jnp.array([1.0 / n_mean] + [1.0 / vth] * 3 + [1.0 / vth**2] * 6 + [1.0 / vth**3] * 3)


def _batch(ids, case, acc, blocks, snap_seeds, snap_every):
    """Run the seeds ``ids`` as one vmapped batch, adding their coarse moments into ``acc``
    (``{dx: [sum, sum of squares]}``); returns ``(snap_x, snap_v, clipped)``, the
    snapshots of the batch's first ``snap_seeds`` seeds (``None`` if there are none)."""
    wave, length_mfp, dx_fine, ppc, ny, grid, sp, params, static, jit_coarse = case
    collide_fn, theta, _ = operator("ntc")
    cases = [setup(dx_fine, ppc, s, 0, wave, length_mfp, ny) for s in ids]
    states = jax.tree.map(_stack, *[c[0] for c in cases])
    cstates = jax.tree.map(_stack, *[c[1] for c in cases])
    keys = jnp.stack([c[6] for c in cases])
    n_snap = max(0, min(snap_seeds - ids[0], len(ids)))
    sx, sv, clipped = [], [], 0
    for b in range(blocks + 1):
        if b:
            states, cstates, clip = advance(states, cstates, keys, grid, sp, params,
                                            (b - 1) * BLOCK, theta, static=static,
                                            collide_fn=collide_fn)  # fmt: skip
            clipped += int(clip)
        for dx, f in jit_coarse.items():
            m = np.asarray(f(states))
            acc[dx][0, b] += m.sum(axis=0)
            acc[dx][1, b] += (m**2).sum(axis=0)
        if n_snap and b % snap_every == 0:
            sx.append(np.asarray(states.x[:n_snap]))
            sv.append(np.asarray(states.v[:n_snap]))
    if not n_snap:
        return None, None, clipped
    return np.stack(sx, axis=1), np.stack(sv, axis=1), clipped


def run(wave, length_mfp, seeds, coarse, dx_fine=0.25, ppc=50, decays=2.0, snap_seeds=16,
        snap_every=10, batch_particles=1e5, ny=1):  # fmt: skip
    """The reference statistics and snapshots (a dict of arrays, see the module docstring).
    ``ny > 1``: a 2-D slab ``ny`` fine cells high (moments binned along ``x`` only)."""
    t0 = time.time()
    _, _, grid, sp, params, static, _, c = setup(dx_fine, ppc, 0, 0, wave, length_mfp, ny)
    lo, hi = float(grid.lo[0]), float(grid.hi[0])
    n_coarse = {dx: round(length_mfp / dx) for dx in coarse}
    jit_coarse = {}
    for dx, n in n_coarse.items():
        scale = _scale(c["n_part"] / n, c["vth"])
        jit_coarse[dx] = jax.jit(functools.partial(_coarse, lo=lo, hi=hi, n_coarse=n,
                                                   scale=scale))  # fmt: skip
    case = (wave, length_mfp, dx_fine, ppc, ny, grid, sp, params, static, jit_coarse)
    t_end = decays / (RATE_CE[wave] * c["mu_ce"] / c["rho"] * c["k"] ** 2)
    blocks = math.ceil(t_end / (c["dt"] * BLOCK))
    acc = {dx: np.zeros((2, blocks + 1, n, len(moments.FIELDS))) for dx, n in n_coarse.items()}
    snaps_x, snaps_v, clipped = [], [], 0
    batch = max(1, min(seeds, int(batch_particles // c["n_part"])))
    for first in range(0, seeds, batch):
        ids = list(range(first, min(first + batch, seeds)))
        sx, sv, clip = _batch(ids, case, acc, blocks, snap_seeds, snap_every)
        clipped += clip
        if sx is not None:
            snaps_x.append(sx)
            snaps_v.append(sv)
        print(f"  seeds {ids[0]}-{ids[-1]} done, {time.time() - t0:.0f} s", flush=True)
    t = c["dt"] * BLOCK * np.arange(blocks + 1) * c["vth"] / c["lambda"]
    out = {"t": t, "snap_t": t[::snap_every]}
    for dx, (s1, s2) in acc.items():
        mean = s1 / seeds
        out[f"mean_dx{dx:g}"] = mean
        out[f"var_dx{dx:g}"] = (s2 - seeds * mean**2) / max(seeds - 1, 1)
    if snaps_x:
        out["snap_x"] = np.concatenate(snaps_x) / c["lambda"]
        out["snap_v"] = np.concatenate(snaps_v) / c["vth"]
    meta = {"wave": wave, "length_mfp": length_mfp, "ny": ny, "height_mfp": ny * dx_fine,
            "seeds": seeds, "coarse": list(coarse),
            "dx_fine": dx_fine, "ppc": ppc, "decays": decays, "fields": moments.FIELDS,
            "attempts_clipped": clipped, "n_part": c["n_part"], "lambda_m": c["lambda"],
            "vth_ms": c["vth"], "dt_mfp": c["dt"] * c["vth"] / c["lambda"],
            "seconds": time.time() - t0}  # fmt: skip
    out["meta"] = np.array(json.dumps(meta))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wave", default="shear", choices=tuple(RATE_CE))
    ap.add_argument("--length-mfp", type=float, default=LENGTH_MFP)
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--coarse", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--dx-fine", type=float, default=0.25)
    ap.add_argument("--ppc", type=int, default=50)
    ap.add_argument("--decays", type=float, default=2.0)
    ap.add_argument("--ny", type=int, default=1, help="2-D slab this many fine cells high")
    ap.add_argument("--snap-seeds", type=int, default=16)
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = run(args.wave, args.length_mfp, args.seeds, args.coarse, args.dx_fine, args.ppc,
              args.decays, args.snap_seeds, args.snap_every, ny=args.ny)  # fmt: skip
    slab = f"_ny{args.ny}" if args.ny > 1 else ""
    path = RESULTS / f"reference_{args.wave}_L{args.length_mfp:g}{slab}_{args.tag}.npz"
    np.savez_compressed(path, **out)
    meta = json.loads(str(out["meta"]))
    print(f"wrote {path} ({meta['seconds']:.0f} s, clipped {meta['attempts_clipped']})")


if __name__ == "__main__":
    main()
