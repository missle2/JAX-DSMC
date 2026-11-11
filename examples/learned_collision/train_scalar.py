"""The first adjoint training test: ES-FP's two scalars ``(log_tau, log_pr)`` fitted through
the solver so a coarse run reproduces the fine references' fluxes.

Each iteration draws a minibatch of windows over the references (:mod:`.reference`): a
random snapshot of a random reference, thinned twice (two replicas) to the coarse particle
count, a forward-only burn-in of ``--burn`` sample blocks, then the two-replica loss
(:mod:`.windows`) over ``--blocks`` sample blocks against the reference means -- the shear
stress ``pxy`` for a shear reference, the heat flux ``qx`` for a thermal one.  Adam (optax).

    PYTHONPATH=. python3 -m examples.learned_collision.train_scalar \
        [--refs reference_shear_L64_v1.npz reference_thermal_L64_v1.npz] [--dx 4] [--ppc 10]
        [--iters 60] [--batch 4] [--blocks 5] [--burn 2] [--lr 0.05] [--operator esfp]

Writes ``examples/learned_collision/results/train_scalar_<tag>.json`` (theta and loss per
iteration).
"""

from __future__ import annotations

import argparse
import functools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.learned_collision import windows
from examples.learned_collision.shear_wave import RESULTS, _stack, operator

FIELD = {"shear": 7, "thermal": 10}
"""The flux each reference's loss compares."""


def load(name):
    """A reference's arrays and metadata."""
    d = np.load(RESULTS / name)
    meta = json.loads(str(d["meta"]))
    return {k: d[k] for k in d.files if k != "meta"}, meta


def _loss(theta, states, cstates, keys, ref, weights, case, collide_fn, blocks, step0, field):
    return windows.window_loss(theta, states, cstates, keys, ref, weights, case, collide_fn,
                               blocks, step0, fields=(field,),
                               basis=windows.wave_basis(case.n_coarse))  # fmt: skip


def compiled(case, collide_fn, args, field):
    """``(burn, value_and_grad)`` jitted for one coarse case (``case`` bound, not traced)."""
    burn = jax.jit(
        jax.vmap(
            functools.partial(windows.burn_in, case=case, collide_fn=collide_fn, blocks=args.burn),
            in_axes=(None, 0, 0, 0),
        )
    )
    grad = jax.jit(
        functools.partial(
            jax.value_and_grad(_loss),
            case=case,
            collide_fn=collide_fn,
            blocks=args.blocks,
            field=field,
        )
    )
    return burn, grad  # fmt: skip


def window(ref, meta, case, rng, fns, theta, args):
    """One window's ``(loss, grad)``: draw, thin, burn in, differentiate."""
    snaps = ref["snap_x"].shape[1]
    snap_every = int(np.round((ref["snap_t"][1] - ref["snap_t"][0]) / (ref["t"][1] - ref["t"][0])))
    last = (len(ref["t"]) - 1 - args.burn - args.blocks) // snap_every
    s = int(rng.integers(0, min(snaps, last + 1)))
    seed = int(rng.integers(0, ref["snap_x"].shape[0]))
    k = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    k_thin, k_run = jax.random.split(k)
    inits = [windows.initial_state(ref["snap_x"][seed, s], ref["snap_v"][seed, s], case,
                                   jax.random.fold_in(k_thin, r)) for r in range(2)]  # fmt: skip
    if any(int(i[2]) for i in inits):
        raise RuntimeError("thinning overflowed the coarse capacity")
    states = jax.tree.map(_stack, *[i[0] for i in inits])
    cstates = jax.tree.map(_stack, *[i[1] for i in inits])
    keys = jnp.stack([jax.random.fold_in(k_run, r) for r in range(2)])
    burn, grad_fn = fns
    if args.burn:
        states, cstates = burn(theta, states, cstates, keys)
    b0 = s * snap_every + args.burn
    target = jnp.asarray(ref[f"mean_dx{args.dx:g}"][b0 + 1 : b0 + 1 + args.blocks])
    field = FIELD[meta["wave"]]
    weights = jnp.asarray(
        windows.field_weights(target, (field,), windows.wave_basis(case.n_coarse))
    )
    return grad_fn(theta, states, cstates, keys, target, weights, step0=args.burn * windows.BLOCK)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--refs", nargs="+", default=["reference_shear_L64_v1.npz", "reference_thermal_L64_v1.npz"]
    )
    ap.add_argument("--dx", type=float, default=4.0)
    ap.add_argument("--ppc", type=int, default=10)
    ap.add_argument("--operator", default="esfp", choices=("esfp", "esfp_recon"))
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4, help="windows per reference per iteration")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--burn", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)  # fmt: skip
    refs = [load(name) for name in args.refs]
    collide_fn, theta, _ = operator(args.operator)
    cases, fns = [], []
    for ref, meta in refs:
        n_fine = ref["snap_x"].shape[2]
        cases.append(windows.coarse_case(meta["wave"], meta["length_mfp"], args.dx, args.ppc,
                                         n_fine, height_mfp=meta.get("height_mfp")
                                         if meta.get("ny", 1) > 1 else None))  # fmt: skip
        fns.append(compiled(cases[-1], collide_fn, args, FIELD[meta["wave"]]))
    opt = optax.adam(args.lr)
    state = opt.init(theta)
    rng = np.random.default_rng(args.seed)
    log, t0 = [], time.time()
    for it in range(args.iters):
        total, grads, per_ref = 0.0, jax.tree.map(jnp.zeros_like, theta), []
        for (ref, meta), case, fn in zip(refs, cases, fns, strict=True):
            losses = []
            for _ in range(args.batch):
                loss, g = window(ref, meta, case, rng, fn, theta, args)
                grads = jax.tree.map(jnp.add, grads, g)
                losses.append(float(loss))
            per_ref.append(float(np.mean(losses)))
            total += per_ref[-1]
        n = args.batch * len(refs)
        grads = jax.tree.map(functools.partial(jnp.multiply, 1.0 / n), grads)
        updates, state = opt.update(grads, state, theta)
        theta = optax.apply_updates(theta, updates)
        row = {
            "iter": it,
            "loss": total,
            "loss_per_ref": per_ref,
            "theta": {k: float(v) for k, v in theta.items()},
            "grad": {k: float(v) for k, v in grads.items()},
            "seconds": time.time() - t0,
        }
        log.append(row)  # fmt: skip
        print(f"it {it:3d}  loss {total: .4e}  per-ref {np.round(per_ref, 4)}  theta "
              f"{ {k: round(v, 4) for k, v in row['theta'].items()} }  {row['seconds']:.0f} s",
              flush=True)  # fmt: skip
        (RESULTS / f"train_scalar_{args.tag}.json").write_text(
            json.dumps({"args": vars(args), "log": log})
        )


if __name__ == "__main__":
    main()
