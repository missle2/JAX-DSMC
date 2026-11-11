"""Round-1 training (plan 2026-11-11): a correction to ES-FP + reconstruction fitted through
the solver on the training cases (:data:`.flows.CASES` role ``train``: S2, S3, K1, K8).

Each iteration draws ``--batch`` windows per case (:mod:`.windows2d`: a random snapshot
pool pair, stripe and thinning, a forward-only burn-in of ``--burn`` blocks, a
differentiable rollout of ``--blocks``), averages their gradients over all cases and takes
an Adam step.  ``--estimator bias`` (default) uses the running-bias gradient ``b grad m``,
``b`` a per-case table of the mean deviation ``E[m - ref]`` (per sample time and wave mode
for a wave, per window time and coarse cell for a shock) updated from every window -- a
running mean for its first ``1 / --bias-decay`` updates, then an exponential average;
``--bias-init`` forward-only windows per case fill it before the first step.  ``pair`` uses
the two-replica product.

Models: ``scalar`` (``log_tau``, ``log_pr``), ``linear`` (the 9-term theory basis, 18
coefficients), ``mlp`` (the same inputs, 2 x 64 tanh) -- :mod:`jax_dsmc.collide.relax`.

    PYTHONPATH=. python3 -m examples.learned_collision.train --model linear [--iters 200]

Writes ``examples/learned_collision/results/train_<model>_<tag>.json``: the arguments, the
final ``theta`` and per-iteration losses, bias norms and gradient norms.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.learned_collision import flows, operators
from examples.learned_collision import windows2d as w2
from examples.learned_collision.evaluate import load_ref
from examples.learned_collision.shear_wave import BLOCK, RESULTS

TRAIN_CASES = tuple(name for name, (_, _, role) in flows.CASES.items() if role == "train")


def _objective(theta, states, cstates, keys, target, weights, bias, case, collide_fn, blocks,
               step0, estimator):  # fmt: skip
    return w2.window_objective(theta, states, cstates, keys, target, weights, bias, case,
                               collide_fn, blocks, step0, estimator)  # fmt: skip


class CaseRunner:
    """One training case: its reference, compiled window functions and running-bias table."""

    def __init__(self, name, model, args):
        ref_name = flows.CASES[name][0]
        self.name = name
        self.ref, self.meta = load_ref(ref_name, args.refs_tag)
        self.case = w2.make_case(name, self.meta, args.ppc)
        fn, _, pairing = operators.operator(model, self.case.kind)
        self.case = self.case._replace(
            static=dataclasses.replace(self.case.static, pairing=pairing)
        )
        self.blocks, self.burn = args.blocks, args.burn
        self.burn_fn = jax.jit(jax.vmap(functools.partial(
            w2.burn_in, case=self.case, collide_fn=fn, blocks=args.burn),
            in_axes=(None, 0, 0, 0)))  # fmt: skip
        self.grad_fn = {
            est: jax.jit(jax.value_and_grad(functools.partial(
                _objective, case=self.case, collide_fn=fn, blocks=args.blocks, estimator=est),
                has_aux=True))
            for est in ("pair", "bias")
        }  # fmt: skip
        n_fields = len(w2.LOSS_FIELDS[self.case.kind])
        if self.case.kind == "shear":
            mean = self.ref[f"mean_dx{self.case.dx_c:g}"]
            self.weights = w2.field_weights(mean, self.case)
            self.bias = np.zeros((mean.shape[0], 2, n_fields))
        else:
            target = w2.shock_target(self.ref["prof_raw"], self.meta["x_shock"], self.case,
                                     self.meta, 1)  # fmt: skip
            self.weights = w2.field_weights(target, self.case)
            self.bias = np.zeros((args.blocks, self.case.n_coarse, n_fields))
        self.bias_n = np.zeros(self.bias.shape[0])
        self.decay = args.bias_decay

    def draw(self, rng):
        """A window: ``(states, cstates, keys, target, rows)``, ``rows`` the bias-table rows."""
        ref, meta = self.ref, self.meta
        seeds, snaps = ref["pool_x"].shape[:2]
        if self.case.kind == "shear":
            every = round((ref["snap_t"][1] - ref["snap_t"][0]) / (ref["t"][1] - ref["t"][0]))
            last = (len(ref["t"]) - 1 - self.burn - self.blocks) // every
            s = int(rng.integers(0, min(snaps, last + 1)))
            b0 = s * every + self.burn
            rows = np.arange(b0 + 1, b0 + 1 + self.blocks)
            target = ref[f"mean_dx{self.case.dx_c:g}"][rows]
        else:
            s = int(rng.integers(0, snaps))
            rows = np.arange(self.blocks)
        seed = int(rng.integers(0, seeds))
        if self.case.kind == "shock":
            target = w2.shock_target(ref["prof_raw"], float(ref["snap_xs"][seed, s]), self.case,
                                     meta, self.blocks)  # fmt: skip
        key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
        k_thin, k_run = jax.random.split(key)
        inits = [w2.pool_state(ref["pool_x"][seed, s, r], ref["pool_v"][seed, s, r],
                               ref["pool_alive"][seed, s, r], self.case,
                               jax.random.fold_in(k_thin, r)) for r in range(2)]  # fmt: skip
        if any(int(i[2]) for i in inits):
            raise RuntimeError(f"{self.name}: thinning overflowed the coarse capacity")
        states = jax.tree.map(flows.stack, *[i[0] for i in inits])
        cstates = jax.tree.map(flows.stack, *[i[1] for i in inits])
        keys = jnp.stack([jax.random.fold_in(k_run, r) for r in range(2)])
        return states, cstates, keys, jnp.asarray(target), rows

    def window(self, theta, rng, estimator, bias=None):
        """``(pair, d_mean, rows, grad)`` of one window (``bias`` overrides the table)."""
        states, cstates, keys, target, rows = self.draw(rng)
        if self.burn:
            states, cstates = self.burn_fn(theta, states, cstates, keys)
        b = self.bias[rows] if bias is None else bias[rows]
        (_, (pair, d_mean)), grad = self.grad_fn[estimator](
            theta, states, cstates, keys, target, jnp.asarray(self.weights), jnp.asarray(b),
            step0=self.burn * BLOCK,
        )  # fmt: skip
        return float(pair), np.asarray(d_mean), rows, grad

    def update_bias(self, rows, d_mean):
        """Fold one window's mean deviation into the running-bias table."""
        self.bias_n[rows] += 1.0
        beta = np.maximum(self.decay, 1.0 / self.bias_n[rows])[:, None, None]
        self.bias[rows] = (1.0 - beta) * self.bias[rows] + beta * d_mean

    def bias_norm(self):
        """``mean(w b^2)`` over the visited rows: the squared bias the model is fitting."""
        seen = self.bias_n > 0
        if not seen.any():
            return float("nan")
        return float(np.mean(np.sum(self.weights * self.bias[seen] ** 2, axis=-1)))


def _scaled(scale, g):
    return scale * g


def train(args):
    """The training loop (module docstring); returns the log dict."""
    _, theta, _ = operators.operator(args.model, "shear", jax.random.PRNGKey(args.seed))
    runners = [CaseRunner(name, args.model, args) for name in args.cases]
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    if args.estimator == "bias":
        for r in runners:
            for _ in range(args.bias_init):
                _, d_mean, rows, _ = r.window(theta, rng, "pair")
                r.update_bias(rows, d_mean)
        print(f"bias tables filled ({args.bias_init} windows per case), "
              f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip
    opt = optax.adam(args.lr)
    state = opt.init(theta)
    log = []
    out_path = RESULTS / f"train_{args.model}_{args.tag}.json"
    for it in range(args.iters):
        grads = jax.tree.map(jnp.zeros_like, theta)
        pairs = {}
        for r in runners:
            vals = []
            for _ in range(args.batch):
                pair, d_mean, rows, g = r.window(theta, rng, args.estimator)
                r.update_bias(rows, d_mean)
                grads = jax.tree.map(jnp.add, grads, g)
                vals.append(pair)
            pairs[r.name] = float(np.mean(vals))
        grads = jax.tree.map(functools.partial(_scaled, 1.0 / (args.batch * len(runners))),
                             grads)  # fmt: skip
        updates, state = opt.update(grads, state, theta)
        theta = optax.apply_updates(theta, updates)
        row = {"iter": it, "pair": pairs, "bias2": {r.name: r.bias_norm() for r in runners},
               "grad_norm": float(optax.global_norm(grads)),
               "log_tau": float(theta["log_tau"]), "log_pr": float(theta["log_pr"]),
               "seconds": time.time() - t0}  # fmt: skip
        log.append(row)
        print(f"it {it:4d}  bias2 {sum(row['bias2'].values()): .4e}  pair "
              f"{sum(pairs.values()): .4e}  |g| {row['grad_norm']:.3e}  log_tau "
              f"{row['log_tau']:+.4f} log_pr {row['log_pr']:+.4f}  {row['seconds']:.0f} s",
              flush=True)  # fmt: skip
        out = {"args": vars(args), "theta": {k: np.asarray(v).tolist() for k, v in theta.items()},
               "log": log}  # fmt: skip
        out_path.write_text(json.dumps(out))
    return out


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="linear", choices=operators.MODELS)
    ap.add_argument("--cases", nargs="+", default=list(TRAIN_CASES))
    ap.add_argument("--estimator", default="bias", choices=("bias", "pair"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4, help="windows per case per iteration")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--burn", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--bias-decay", type=float, default=0.05)
    ap.add_argument("--bias-init", type=int, default=32)
    ap.add_argument("--ppc", type=int, default=flows.PPC_COARSE)
    ap.add_argument("--refs-tag", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    return ap


def main(argv=None):
    train(parser().parse_args(argv))


if __name__ == "__main__":
    main()
