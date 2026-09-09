"""Weak- and strong-scaling harness.

Run it on the node you care about::

    python3 -m examples.scaling --devices 1 2 4 --dim 3 --side 96

and to check it works without a multi-GPU node::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        python3 -m examples.scaling --devices 1 2 4 --dim 2 --side 32 --steps 4

Forced-host devices time-share one processor, so they verify that everything
*runs* and that particles are conserved -- they cannot measure speed, and the
harness says so rather than printing a meaningless efficiency.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from examples.turbulence.case import initial_particles, setup  # noqa: E402
from jax_dsmc import decomp, parallel  # noqa: E402
from jax_dsmc.boundary import BC  # noqa: E402
from jax_dsmc.simulate import run  # noqa: E402
from jax_dsmc.state import ParticleState  # noqa: E402


def check_idle(threshold_mib=512):
    """Warn if another process already holds device memory (spec section 8.6).

    A concurrent job on the same GPU once produced a phantom 36x performance cliff
    in this project.  Every number below is meaningless if this fires.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        ).stdout
        used = [int(v) for v in out.split()]
    except Exception:
        return
    busy = [i for i, u in enumerate(used) if u > threshold_mib]
    if busy:
        print(f"  ** WARNING: GPUs {busy} already busy; timings below are not valid **")


def timeit(fn, *args, repeat=3):
    """Median wall time, with compilation and async dispatch excluded.

    The ``block_until_ready`` is not optional: JAX dispatches asynchronously, so
    without it this times the *enqueue* and reports absurd speed.
    """
    jax.block_until_ready(fn(*args))  # warm-up pays the compile
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _pack_by_slab(st0, grid, p, per, dim):
    """Regroup a global state into ``p`` equal, padded per-device blocks."""
    slab = np.asarray(decomp.slab_of(st0.x, grid, p))
    xs, vs, sps, al, lg = [], [], [], [], []
    for r in range(p):
        idx = np.where(slab == r)[0]
        if idx.size > per:
            raise SystemExit(
                f"slab {r} holds {idx.size:,} particles but capacity is {per:,}; raise --headroom"
            )
        pad = per - idx.size
        xs.append(np.concatenate([np.asarray(st0.x)[idx], np.zeros((pad, dim))]))
        vs.append(np.concatenate([np.asarray(st0.v)[idx], np.zeros((pad, 3))]))
        sps.append(np.zeros(per, np.int32))
        al.append(np.concatenate([np.ones(idx.size, bool), np.zeros(pad, bool)]))
        lg.append(np.zeros(per))
    return ParticleState(
        x=jnp.asarray(np.concatenate(xs)),
        v=jnp.asarray(np.concatenate(vs)),
        species=jnp.asarray(np.concatenate(sps)),
        alive=jnp.asarray(np.concatenate(al)),
        logh_acc=jnp.asarray(np.concatenate(lg)),
    )


def one_point(n_dev, dim, side, npc, n_steps, headroom=1.6):
    """Build a Taylor-Green case, decompose it over ``n_dev`` devices, and time it."""
    sp, grid, static, params, cstate, info = setup(
        dim=dim, cells_per_side=side, npc=npc, n_steps=n_steps
    )
    static = dataclasses.replace(static, n_steps=n_steps)
    key = jax.random.PRNGKey(0)
    st0 = initial_particles(key, grid, static, info)

    if n_dev == 1:
        f = jax.jit(
            lambda s, c: run(
                s, c, grid, sp, params, static, key, bc=BC.PERIODIC, emit_moments="none"
            )
        )
        t = timeit(f, st0, cstate)
        mig = None
        alive = info["N"]
    else:
        if side % n_dev:
            raise SystemExit(f"--side {side} does not divide over {n_dev} devices")
        mesh = parallel.device_mesh(n_dev)
        per = decomp.suggest_capacity(info["N"], n_dev, headroom=headroom)
        m_max = decomp.suggest_migration_buffer(per, side // n_dev)
        st_d = _pack_by_slab(st0, grid, n_dev, per, dim)
        with mesh:
            st_d = parallel.shard_state(st_d, mesh)
            f = jax.jit(
                lambda s, c: decomp.run_decomposed(
                    s, c, grid, sp, params, static, key, mesh, BC.PERIODIC, n_steps, m_max
                )
            )
            t = timeit(f, st_d, cstate)
            (_s2, _c2), (_m2, _d2, mig) = f(st_d, cstate)
            alive = int(jnp.sum(_s2.alive))
    return {
        "devices": n_dev,
        "dim": dim,
        "side": side,
        "cells": info["C"],
        "N": info["N"],
        "alive": alive,
        "conserved": alive == info["N"],
        "steps": n_steps,
        "wall": t,
        "ns_per_particle_step": t / (info["N"] * n_steps) * 1e9,
        "sent_per_step": float(jnp.mean(mig["sent"])) if mig is not None else 0.0,
        "overflow": (
            int(jnp.sum(mig["overflow_send"]) + jnp.sum(mig["overflow_recv"]))
            if mig is not None
            else 0
        ),
    }


def _row(r, extra):
    flag = "" if r["conserved"] else "  ** PARTICLES LOST **"
    if r["overflow"]:
        flag += f"  ** OVERFLOW {r['overflow']} **"
    return (
        f"{r['devices']:>4} {r['side']:>6} {r['N']:>14,} {r['wall']:>10.3f}"
        f" {extra:>8} {r['ns_per_particle_step']:>12.1f} {r['sent_per_step']:>11.0f}{flag}"
    )


HEAD = f"{'dev':>4} {'side':>6} {'N':>14} {'wall (s)':>10} {'eff':>8} {'ns/p-step':>12} {'migrate/s':>11}"


def main():  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--mode", choices=["weak", "strong", "both"], default="both")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--side", type=int, default=48, help="cells per side on ONE device")
    ap.add_argument("--npc", type=int, default=50)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--headroom", type=float, default=1.6)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    have = len(jax.devices())
    devs = [d for d in a.devices if d <= have]
    if not devs:
        raise SystemExit(f"no requested device count fits: {have} device(s) present")
    if len(devs) < len(a.devices):
        print(f"  (only {have} device(s); skipping {sorted(set(a.devices) - set(devs))})")
    print(f"backend {jax.default_backend()}   {have} device(s)")
    if jax.default_backend() == "cpu":
        print("  ** forced-host devices share one processor: correctness only, NOT speed **")
    check_idle()
    results = {"weak": [], "strong": []}

    if a.mode in ("strong", "both"):
        print(f"\n=== STRONG scaling: fixed {a.side}^{a.dim} cells, npc {a.npc} ===")
        print(HEAD)
        base = None
        for p in devs:
            r = one_point(p, a.dim, a.side, a.npc, a.steps, a.headroom)
            base = base or r["wall"]
            r["speedup"] = base / r["wall"]
            r["efficiency"] = r["speedup"] / p
            results["strong"].append(r)
            print(_row(r, f"{r['efficiency']:.2f}"))

    if a.mode in ("weak", "both"):
        print(f"\n=== WEAK scaling: {a.side}^{a.dim} cells PER DEVICE, npc {a.npc} ===")
        print(HEAD)
        base = None
        for p in devs:
            # grow the box so the work per device is constant; the x axis carries
            # the cut, so it is the axis that must stay divisible by p
            side = round(a.side * p ** (1.0 / a.dim))
            side += (-side) % (2 * max(devs))
            r = one_point(p, a.dim, side, a.npc, a.steps, a.headroom)
            base = base or r["wall"]
            r["efficiency"] = base / r["wall"]
            results["weak"].append(r)
            print(_row(r, f"{r['efficiency']:.2f}"))

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {a.json}")
    return results


if __name__ == "__main__":
    main()
