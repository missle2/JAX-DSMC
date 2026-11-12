"""Warm pools of coarse shock states for training's persistent chains (``lcwarm``,
2026-11-12; :mod:`.train` ``--warm-pool``).

Per shock case: ``--size`` coarse runs started as the training windows start (thinned
reference snapshots, :func:`.evaluate._starts`: random seed, snapshot, pool, stripe and
thinning), relaxed ``--blocks`` blocks at the base operator (lctransient: the M3-M10 window
integrals settle on their plateau after ~80 blocks), saved together with
:func:`examples.mesh.optimize.save_warm` (the optimiser's warm-start format: every field of
the ``(ParticleState, CollideState)``, batched on the leading axis; an atomic write).  Every
training run starts from the base operator, so one pool serves all of them; a run draws its
``2 --batch`` chains from it by its seed.

    PYTHONPATH=. python3 -m examples.learned_collision.warm_pool [--cases K1 K4 K8]
        [--size 128] [--blocks 120]

Writes ``examples/learned_collision/results/warm_<case>_<tag>.npz`` and prints each pool's
final ``int tau_xx`` / ``int q_x`` ratios and shock positions (the plateau check).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples.learned_collision import evaluate as ev
from examples.learned_collision import flows, operators, refdata, transient
from examples.learned_collision import windows2d as w2
from examples.learned_collision.shear_wave import BLOCK
from examples.learned_collision.train import TRAIN_CASES, warm_path
from examples.mesh.optimize import save_warm


def relaxed(name, args, rng):
    """``(states, cstates)`` of ``--size`` relaxed coarse runs of shock case ``name``, in
    batches of ``--chunk`` (and their ``int tau_xx``, ``int q_x`` ratios, shock positions)."""
    ref, meta = ev.load_ref(flows.CASES[name][0], args.refs_tag)
    case = w2.make_case(name, meta, args.ppc)
    fn, theta, pairing = operators.operator("esfp_recon", "shock")
    static = dataclasses.replace(case.static, pairing=pairing)
    faces, capacity, res, wall = flows.run_kwargs(case.bc)
    sums = jax.jit(jax.vmap(functools.partial(refdata._fine_sums, length=case.hi,
                                              nx=case.n_coarse)))  # fmt: skip
    tgt = ev._fields(w2.placed_raw(ref["prof_raw"], meta["x_shock"], case, meta),
                     round(case.dx_c / meta["dx_fine"]), meta["ppc_up"], meta["vth_ms"])  # fmt: skip
    ref_int = (tgt[:, 2].sum(), tgt[:, 3].sum())
    ratio = flows.shock_states(flows.REFERENCES[flows.CASES[name][0]]["mach"])["ratio"]
    x_c = (np.arange(case.n_coarse) + 0.5) * case.dx_c
    out, txx, qx, xs = [], [], [], []
    for _ in range(0, args.size, args.chunk):
        states, cstates, _ = ev._starts(ref, case, args.chunk, rng)
        keys = jnp.stack([jax.random.PRNGKey(int(rng.integers(2**31 - 1)))
                          for _ in range(args.chunk)])  # fmt: skip
        for b in range(args.blocks):
            states, cstates, _, _ = flows.advance(
                states, cstates, keys, case.grid, case.sp, case.params, b * BLOCK, theta, res,
                wall, static=static, collide_fn=fn, faces=faces, capacity=capacity,
            )  # fmt: skip
        s = np.asarray(sums(states))
        t, q = transient._integrals(s, case, meta, ref_int)
        txx.append(t)
        qx.append(q)
        xs.append(flows.shock_position(s[:, :, 0], x_c, ratio, case.ppc_up))
        out.append((states, cstates))
    pool = jax.tree.map(_concat, *out)
    return pool, np.concatenate(txx), np.concatenate(qx), np.concatenate(xs)


def _concat(*leaves):
    return jnp.concatenate(leaves)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    shocks = [c for c in TRAIN_CASES if flows.CASES[c][0].startswith("shock")]
    ap.add_argument("--cases", nargs="+", default=shocks)
    ap.add_argument("--size", type=int, default=128, help="chains per pool")
    ap.add_argument("--chunk", type=int, default=32, help="runs relaxed side by side")
    ap.add_argument("--blocks", type=int, default=120)
    ap.add_argument("--ppc", type=int, default=flows.PPC_COARSE)
    ap.add_argument("--refs-tag", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)
    if args.size % args.chunk:
        raise SystemExit("--size must be a multiple of --chunk")
    rng = np.random.default_rng(args.seed)
    for name in args.cases:
        t0 = time.time()
        pool, txx, qx, xs = relaxed(name, args, rng)
        save_warm(warm_path(name, args.tag), pool)
        se = np.sqrt(len(txx))
        print(f"{name}: {len(txx)} chains after {args.blocks} blocks: int txx {txx.mean():.3f} "
              f"+- {txx.std(ddof=1) / se:.3f}, int qx {qx.mean():.3f} +- "
              f"{qx.std(ddof=1) / se:.3f}, shock at {np.median(xs):.1f} (spread "
              f"{np.percentile(xs, 5):.1f}-{np.percentile(xs, 95):.1f}) lambda_1  "
              f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
