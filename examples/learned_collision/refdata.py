"""Fine 2-D reference data for the round-1 flows (:mod:`.flows`): one fine NTC run per
reference, batched over seeds, at ``dx = DX_FINE lambda`` and ``PPC_FINE`` particles per cell.

**Shear waves** record, at every sample block, the seed mean and variance of the coarse
moments (:mod:`.moments`) on each coarse grid the cases read -- the window targets are this
time series.

**Shocks** are steady: after a warm-up of ``--warm`` flow-through times they record, every
block, the per-seed shock position (:func:`.flows.shock_position`) and the fine-bin raw sums
along ``x`` *aligned on it* (shifted by whole fine bins to the reference position
``SHOCK_X``) -- the shock-frame profile.  A window's target is that profile placed at its
snapshot's shock position and summed into the coarse cells (:mod:`.windows`).

Both store **particle pools** instead of full snapshots: every ``--snap-every`` blocks, each
of the first ``--snap-seeds`` seeds' particles is assigned to one of ``K = 2`` disjoint pools
with probability ``p_pool`` each (the largest coarse thinning of the reference).  Window
replica ``r`` thins pool ``r`` by ``p / p_pool``: the same marginal as thinning the full
snapshot by ``p``, two replicas with no shared particle, at ``2 p_pool`` of the storage.

Units: positions in ``lambda``, velocities in ``v_th``, time in ``lambda / v_th`` (upstream
for a shock).

    PYTHONPATH=. python3 -m examples.learned_collision.refdata --ref shock_M3 [--seeds 4]

Writes ``examples/learned_collision/results/ref2d_<ref>_<tag>.npz``.
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

from examples.learned_collision import flows, moments
from examples.learned_collision.reference import _scale
from examples.learned_collision.shear_wave import BLOCK, RATE_CE, RESULTS

POOLS = 2


def pool(x, v, alive, key, p_pool, cap):
    """``(x[K, cap, D], v[K, cap, 3], alive[K, cap], overflow)``: the alive particles split
    into ``K = POOLS`` disjoint pools, each particle in pool ``r`` with probability
    ``p_pool`` (``K p_pool <= 1``), the rest dropped."""
    group = jnp.floor(jax.random.uniform(key, alive.shape) / p_pool).astype(jnp.int32)
    xs, vs, al, over = [], [], [], 0
    for r in range(POOLS):
        keep = alive & (group == r)
        count = jnp.sum(keep)
        idx = jnp.nonzero(keep, size=cap, fill_value=0)[0]
        xs.append(x[idx])
        vs.append(v[idx])
        al.append(jnp.arange(cap) < count)
        over = over + jnp.maximum(count - cap, 0)
    return jnp.stack(xs), jnp.stack(vs), jnp.stack(al), over


def _fine_sums(state, length, nx):
    """``f64[nx, 16]``: per fine ``x``-bin ``(N, S1, S2 (9), S3)`` summed over ``y``."""
    count, s1, s2, s3 = moments.raw_sums(state.x[:, 0], state.v, state.alive, 0.0, length, nx)
    return jnp.concatenate([count[:, None], s1, s2.reshape(-1, 9), s3], axis=1)


def _coarse(state, length, n, scale):
    return moments.coarse_moments(state, 0.0, length, n) * scale


class _Recorder:
    """The jitted reductions of one reference run and the accumulated statistics."""

    def __init__(self, ref, c, p_pool, pool_cap, dx, blocks):
        length = c["cells"] * dx * c["lambda"]
        self.ref, self.c, self.dx, self.nx = ref, c, dx, c["cells"]
        self.fine = jax.jit(jax.vmap(functools.partial(_fine_sums, length=length,
                                                       nx=self.nx)))  # fmt: skip
        self.pool = jax.jit(jax.vmap(functools.partial(pool, p_pool=p_pool, cap=pool_cap)))
        self.coarse, self.acc = {}, {}
        if ref["kind"] == "shear":
            for d in flows.coarse_resolutions(ref["name"]):
                n = round(ref["length"] / d)
                scale = _scale(c["n_part"] / n, c["vth"])
                self.coarse[d] = jax.jit(jax.vmap(functools.partial(
                    _coarse, length=length, n=n, scale=scale)))  # fmt: skip
                self.acc[d] = np.zeros((2, blocks + 1, n, 13))
        self.x_centres = (np.arange(self.nx) + 0.5) * dx
        self.prof = np.zeros((self.nx, 16))
        self.prof_n = np.zeros(self.nx)
        self.unaligned = np.zeros((self.nx, 16))
        self.shock_xs, self.pools, self.overflow = [], [], 0

    def record(self, k, states):
        """Add sample block ``k``; returns the per-seed shock positions (zeros for a wave)."""
        sums = np.asarray(self.fine(states))  # [B, nx, 16]
        for d, f in self.coarse.items():
            m = np.asarray(f(states))
            self.acc[d][0, k] += m.sum(axis=0)
            self.acc[d][1, k] += (m**2).sum(axis=0)
        if self.ref["kind"] != "shock":
            return np.zeros(sums.shape[0])
        xs = flows.shock_position(sums[:, :, 0], self.x_centres, self.c["ratio"],
                                  self.c["ppc_up"])  # fmt: skip
        self.shock_xs.append(xs)
        self.unaligned += sums.sum(axis=0)
        src = np.arange(self.nx)
        for i, x_s in enumerate(xs):
            dst = src - round((x_s - flows.SHOCK_X) / self.dx)
            ok = (dst >= 0) & (dst < self.nx)
            self.prof[dst[ok]] += sums[i, src[ok]]
            self.prof_n[dst[ok]] += 1.0
        return xs

    def snapshot(self, states, n_snap, key, xs):
        """The pools of the first ``n_snap`` seeds (host, float32) and their shock positions."""
        c = self.c
        x, v, al, over = self.pool(states.x[:n_snap], states.v[:n_snap],
                                   states.alive[:n_snap],
                                   jax.random.split(key, n_snap))  # fmt: skip
        self.overflow += int(jnp.sum(over))
        return (np.asarray(x / c["lambda"], np.float32), np.asarray(v / c["vth"], np.float32),
                np.asarray(al), np.asarray(xs[:n_snap]))  # fmt: skip

    def arrays(self, seeds, samples):
        """The output arrays (without ``t`` and ``meta``)."""
        cat = [np.concatenate([np.stack([s[i] for s in batch], axis=1) for batch in self.pools])
               for i in range(4)]  # fmt: skip
        out = {"pool_x": cat[0], "pool_v": cat[1], "pool_alive": cat[2], "snap_xs": cat[3]}
        for d, (s1, s2) in self.acc.items():
            mean = s1 / seeds
            out[f"mean_dx{d:g}"] = mean
            out[f"var_dx{d:g}"] = (s2 - seeds * mean**2) / max(seeds - 1, 1)
        if self.ref["kind"] == "shock":
            out["prof_raw"] = self.prof / np.maximum(self.prof_n, 1.0)[:, None]
            out["prof_samples"] = self.prof_n
            out["unaligned_raw"] = self.unaligned / samples
            out["shock_xs"] = np.stack(self.shock_xs)
        return out


def _stack_setups(ref, ids, dx, ppc):
    cases = [flows.setup(ref, dx, ppc, s) for s in ids]
    states = jax.tree.map(flows.stack, *[c[0] for c in cases])
    cstates = jax.tree.map(flows.stack, *[c[1] for c in cases])
    keys = jnp.stack([c[6] for c in cases])
    return states, cstates, keys


def _blocks(ref, c, warm, sample, decays):
    """``(warm_blocks, sample_blocks)``: a wave runs ``decays`` e-foldings from the start; a
    shock warms up ``warm`` and samples ``sample`` flow-through times (``L / u_2``)."""
    if ref["kind"] == "shear":
        rate = RATE_CE["shear"] * c["mu_ce"] / c["rho"] * c["k"] ** 2
        return 0, math.ceil(decays / rate / (c["dt"] * BLOCK))
    flow_through = c["length"] / c["u2"]
    return (math.ceil(warm * flow_through / (c["dt"] * BLOCK)),
            math.ceil(sample * flow_through / (c["dt"] * BLOCK)))  # fmt: skip


def run(name, seeds, snap_seeds=8, snap_every=10, warm=3.0, sample=4.0, decays=2.0,
        batch_particles=4e6, ppc=flows.PPC_FINE, dx=flows.DX_FINE):  # fmt: skip
    """The reference arrays (module docstring) of :data:`.flows.REFERENCES` ``name``."""
    t0 = time.time()
    ref = dict(flows.REFERENCES[name], name=name)
    _, _, grid, sp, params, static, _, c, bc = flows.setup(ref, dx, ppc, 0)
    faces, capacity, res, wall = flows.run_kwargs(bc)
    p_pool = min(1.0 / POOLS, max(flows.thinning(d, dx_f=dx, ppc_f=ppc)
                                  for d in flows.coarse_resolutions(name)))  # fmt: skip
    pool_cap = math.ceil(c["n_part"] * p_pool + 6.0 * math.sqrt(c["n_part"] * p_pool) + 32)
    warm_blocks, blocks = _blocks(ref, c, warm, sample, decays)
    rec = _Recorder(ref, c, p_pool, pool_cap, dx, blocks)
    clipped, short = 0, 0.0
    batch = max(1, min(seeds, int(batch_particles // c["n_part"])))
    for first in range(0, seeds, batch):
        ids = list(range(first, min(first + batch, seeds)))
        states, cstates, keys = _stack_setups(ref, ids, dx, ppc)
        n_snap = max(0, min(snap_seeds - first, len(ids)))
        snaps = []
        for b in range(warm_blocks + blocks + 1):
            if b:
                states, cstates, clip, sh = flows.advance(
                    states, cstates, keys, grid, sp, params, (b - 1) * BLOCK, None, res, wall,
                    static=static, collide_fn=None, faces=faces, capacity=capacity,
                )  # fmt: skip
                clipped += int(clip)
                short = max(short, float(sh))
            k = b - warm_blocks
            if k < 0:
                continue
            xs = rec.record(k, states)
            if n_snap and k % snap_every == 0:
                key = jax.random.fold_in(jax.random.PRNGKey(10_000 + first), k)
                snaps.append(rec.snapshot(states, n_snap, key, xs))
        if snaps:
            rec.pools.append(snaps)
        print(f"  {name} seeds {ids[0]}-{ids[-1]} done, {time.time() - t0:.0f} s, clipped "
              f"{clipped}, shortfall {short:g}, pool overflow {rec.overflow}",
              flush=True)  # fmt: skip
    out = rec.arrays(seeds, max(1, seeds * (blocks + 1)))
    dt_mfp = c["dt"] * c["vth"] / c["lambda"]
    out["t"] = dt_mfp * BLOCK * np.arange(blocks + 1)
    out["snap_t"] = out["t"][::snap_every]
    meta = {**ref, "seeds": seeds, "dx_fine": dx, "ppc": ppc, "p_pool": p_pool,
            "pools": POOLS, "dt_mfp": dt_mfp, "block": BLOCK, "warm_blocks": warm_blocks,
            "blocks": blocks, "cells": c["cells"], "ny": c["ny"], "ppc_up": c["ppc_up"],
            "n_part": c["n_part"], "lambda_m": c["lambda"], "vth_ms": c["vth"],
            "attempts_clipped": clipped, "inflow_shortfall": short,
            "pool_overflow": rec.overflow, "x_shock": flows.SHOCK_X,
            "ratio": c.get("ratio", 1.0), "seconds": time.time() - t0}  # fmt: skip
    out["meta"] = np.array(json.dumps(meta))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, choices=tuple(flows.REFERENCES))
    ap.add_argument("--seeds", type=int, default=None, help="default: 16 shear, 4 shock")
    ap.add_argument("--snap-seeds", type=int, default=8)
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--warm", type=float, default=3.0, help="shock warm-up, flow-throughs")
    ap.add_argument("--sample", type=float, default=4.0, help="shock sampling, flow-throughs")
    ap.add_argument("--decays", type=float, default=2.0, help="shear-wave run, e-foldings")
    ap.add_argument("--batch-particles", type=float, default=4e6)
    ap.add_argument("--ppc", type=int, default=flows.PPC_FINE)
    ap.add_argument("--dx", type=float, default=flows.DX_FINE)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)
    seeds = args.seeds or (16 if flows.REFERENCES[args.ref]["kind"] == "shear" else 4)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = run(args.ref, seeds, args.snap_seeds, args.snap_every, args.warm, args.sample,
              args.decays, args.batch_particles, args.ppc, args.dx)  # fmt: skip
    path = RESULTS / f"ref2d_{args.ref}_{args.tag}.npz"
    np.savez_compressed(path, **out)
    meta = json.loads(str(out["meta"]))
    print(
        f"wrote {path} ({meta['seconds']:.0f} s, clipped {meta['attempts_clipped']}, "
        f"shortfall {meta['inflow_shortfall']:g}, pool overflow {meta['pool_overflow']})"
    )


if __name__ == "__main__":
    main()
