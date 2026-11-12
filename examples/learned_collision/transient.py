"""How far are the training windows from the coarse operator's own steady shock?
(``lctransient``, 2026-11-12.)

Every loss variant of ``lcloss`` (per-cell, integrals, low modes, qx weighted x3 and x10)
trained the heat-flux rate *up* (int q_x 0.95 -> 0.89 at M10), although at x10 the steady
integrals' sensitivities (lcsteadyfd) ask for it to go down on every shock.  The windows
start from thinned fine snapshots and burn in only 2 blocks: the coarse run is still relaxing
from the fine profile, so the window bias the trainer sees may not be the steady error.  This
measures that transient: coarse runs from the same starts as training (thinned reference
snapshots), forward only at the base coefficients, and per block the shift-invariant
integrals ``int tau_xx dx`` and ``int q_x dx`` over the reference's.  The window (blocks
``burn`` to ``burn + blocks``) against the plateau (the last ``--plateau`` blocks) says
whether the window sees the steady error's sign, and the curve says how long a burn-in (or a
persistent chain's warm-up) must be.

    PYTHONPATH=. python3 -m examples.learned_collision.transient [--cases K1 K4 K8]
        [--blocks 160] [--starts 32]

Writes ``examples/learned_collision/results/transient_<tag>.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples.learned_collision import evaluate as ev
from examples.learned_collision import flows, operators, refdata
from examples.learned_collision import windows2d as w2
from examples.learned_collision.shear_wave import BLOCK, RESULTS


def _integrals(raw, case, meta, ref_int):
    """``(int tau_xx, int q_x)`` of each start's coarse sums ``raw[S, C, 16]`` over the
    reference's (one coarse cell per bin)."""
    f = np.stack([ev._fields(r, 1, case.ppc_up, meta["vth_ms"]) for r in raw])  # [S, C, 4]
    return f[:, :, 2].sum(axis=1) / ref_int[0], f[:, :, 3].sum(axis=1) / ref_int[1]


def study(name, op_name, args):
    ref, meta = ev.load_ref(flows.CASES[name][0], args.refs_tag)
    case = w2.make_case(name, meta, args.ppc)
    fn, theta, pairing = operators.operator(op_name, "shock")
    static = dataclasses.replace(case.static, pairing=pairing)
    faces, capacity, res, wall = flows.run_kwargs(case.bc)
    rng = np.random.default_rng(args.seed)
    states, cstates, _ = ev._starts(ref, case, args.starts, rng)
    keys = jnp.stack([jax.random.PRNGKey(int(rng.integers(2**31 - 1)))
                      for _ in range(args.starts)])  # fmt: skip
    sums = jax.jit(jax.vmap(functools.partial(refdata._fine_sums, length=case.hi,
                                              nx=case.n_coarse)))  # fmt: skip
    tgt = ev._fields(w2.placed_raw(ref["prof_raw"], meta["x_shock"], case, meta),
                     round(case.dx_c / meta["dx_fine"]), meta["ppc_up"], meta["vth_ms"])  # fmt: skip
    ref_int = (tgt[:, 2].sum(), tgt[:, 3].sum())
    txx, qx = [], []
    t0 = time.time()
    for b in range(args.blocks):
        # the integrals after b blocks (b = 0: the thinned snapshot itself)
        t, q = _integrals(np.asarray(sums(states)), case, meta, ref_int)
        txx.append(t)
        qx.append(q)
        states, cstates, _, _ = flows.advance(
            states, cstates, keys, case.grid, case.sp, case.params, b * BLOCK, theta, res, wall,
            static=static, collide_fn=fn, faces=faces, capacity=capacity,
        )  # fmt: skip
    txx, qx = np.array(txx), np.array(qx)  # [blocks, S]
    se = math.sqrt(args.starts)
    win = slice(args.burn, args.burn + args.window)
    plat = slice(args.blocks - args.plateau, args.blocks)
    out = {"case": name, "operator": op_name,
           "txx_mean": txx.mean(1).tolist(), "txx_se": (txx.std(1, ddof=1) / se).tolist(),
           "qx_mean": qx.mean(1).tolist(), "qx_se": (qx.std(1, ddof=1) / se).tolist(),
           "window_txx": float(txx[win].mean()), "window_qx": float(qx[win].mean()),
           "plateau_txx": float(txx[plat].mean()), "plateau_qx": float(qx[plat].mean()),
           "seconds": time.time() - t0}  # fmt: skip
    print(f"{name} {op_name}: window (blocks {args.burn}-{args.burn + args.window}) int txx "
          f"{out['window_txx']:.3f} qx {out['window_qx']:.3f} | plateau int txx "
          f"{out['plateau_txx']:.3f} qx {out['plateau_qx']:.3f}  {out['seconds']:.0f} s",
          flush=True)  # fmt: skip
    for b in (0, 1, 2, 4, 7, 10, 20, 40, 80, args.blocks - 1):
        if b < args.blocks:
            print(f"  block {b:3d}: int txx {txx[b].mean():.3f} +- {txx[b].std(ddof=1) / se:.3f}"
                  f"  qx {qx[b].mean():.3f} +- {qx[b].std(ddof=1) / se:.3f}", flush=True)  # fmt: skip
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", nargs="+", default=["K1", "K4", "K8"])
    ap.add_argument("--operators", nargs="+", default=["esfp_recon"])
    ap.add_argument("--blocks", type=int, default=160)
    ap.add_argument("--starts", type=int, default=32)
    ap.add_argument("--burn", type=int, default=2, help="the training windows' burn-in")
    ap.add_argument("--window", type=int, default=5, help="the training windows' length")
    ap.add_argument("--plateau", type=int, default=40, help="blocks averaged as the plateau")
    ap.add_argument("--ppc", type=int, default=flows.PPC_COARSE)
    ap.add_argument("--refs-tag", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)
    rows = []
    for name in args.cases:
        for op in args.operators:
            rows.append(study(name, op, args))
            (RESULTS / f"transient_{args.tag}.json").write_text(
                json.dumps({"args": vars(args), "rows": rows})
            )


if __name__ == "__main__":
    main()
