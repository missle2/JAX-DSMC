"""Round-1 training (plan 2026-11-11): a correction to ES-FP + reconstruction fitted through
the solver on the training cases (:data:`.flows.CASES` role ``train``: S3, S4, K1, K4, K8).

Each iteration draws ``--batch`` windows per case (:mod:`.windows2d`: a random snapshot
pool pair, stripe and thinning, a forward-only burn-in of ``--burn`` blocks, a
differentiable rollout of ``--blocks``), averages their gradients over all cases and takes
an Adam step.  ``--estimator bias`` (default) uses the running-bias gradient ``b grad m``,
``b`` a per-case table of the mean deviation ``E[m - ref]`` (per window time and wave mode,
relative to the reference flux, for a wave; per window time and coarse cell for a shock) updated from every window -- a
running mean for its first ``1 / --bias-decay`` updates, then an exponential average;
``--bias-init`` forward-only windows per case fill it before the first step.  ``pair`` uses
the two-replica product.

Models: ``scalar`` (``log_tau``, ``log_pr``), ``linear`` (the 9-term theory basis, 18
coefficients), ``mlp`` (the same inputs, ``--depth`` x ``--width`` tanh, zero at zero
features) -- :mod:`jax_dsmc.collide.relax`.  The linear and MLP corrections read features
standardised by their training spread (``--features``, :mod:`.features`) and, by default,
keep the base's global ``log_tau`` / ``log_pr`` frozen at the physical values
(``--train-intercept`` frees them): a correction then vanishes in resolved equilibrium cells.
**Steps** (2026-11-12): Adam with each weight matrix's step divided by its fan-in, so ``--lr``
is the per-iteration change of the correction's output whatever the width (plain Adam moves
every coordinate by ~``lr`` at once: on near-collinear features the linear ``dlog_tau`` moved
~``lr sum phi`` per step, and at lr 0.02 six iterations took the shear viscosity to 0.5x and
saturated the MLP's tanh bound, lctime 2026-11-11).  The final ``theta`` is the mean of the
last ``--average`` fraction of the iterates (the per-step gradient SNR is ~1).
Regularisation: ``--weight-decay`` (a decoupled per-iteration shrink of the MLP weights,
independent of ``--lr``), ``--l1`` (a proximal soft-threshold of the linear coefficients after
each step).

**Persistent shock chains** (``--warm-pool TAG``, 2026-11-12): a window from a thinned fine
snapshot sees the coarse run still relaxing from the fine profile -- at M10 its ``int q_x``
is 1.01 over blocks 2-7 against the steady 0.94 (lctransient), so the window objective never
saw the heat-flux deficit and every loss variant traded heat flux for stress (lcloss).  With
a warm pool (:mod:`.warm_pool`, ``lcwarm``: coarse runs from the snapshots relaxed ~120 blocks
at the base operator, saved by :func:`examples.mesh.optimize.save_warm`) each shock case
keeps ``2 --batch`` chains: every window continues two of them from where their last window
ended, and its end states go back -- the windows follow the model's own steady state at no
burn-in cost (the optimiser's restart, :func:`examples.ensemble_mesh._warm_up`, applied per
window).  The target is placed at the chains' measured shock position, and a chain whose
shock drifted more than ``--chain-drift`` [lambda_1] from the pool's median is replaced by
a copy of a random healthy chain (its own key decorrelates it).  The waves are unchanged.

Cross-validation: ``--cases`` the cases trained on, ``--pool-seeds`` the reference seeds whose
snapshots are drawn; ``--validate`` cases (and ``--validate-train``: the training cases too)
are evaluated forward at the end (:mod:`.evaluate`, ``--eval-seeds`` / ``--eval-starts``) and
the metrics written with the run.

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

from examples.learned_collision import evaluate as ev
from examples.learned_collision import flows, operators, refdata
from examples.learned_collision import windows2d as w2
from examples.learned_collision.evaluate import load_ref
from examples.learned_collision.shear_wave import BLOCK, RESULTS
from examples.mesh.optimize import load_warm
from jax_dsmc.collide import relax

TRAIN_CASES = tuple(name for name, (_, _, role) in flows.CASES.items() if role == "train")


def _objective(theta, states, cstates, keys, target, weights, bias, case, collide_fn, blocks,
               step0, estimator):  # fmt: skip
    return w2.window_objective(theta, states, cstates, keys, target, weights, bias, case,
                               collide_fn, blocks, step0, estimator, final=True)  # fmt: skip


def _put(chain, new, idx):
    return chain.at[idx].set(new)


def _pairs(x):
    """``[m, 2, ...]`` -> ``[2 m, ...]`` (a window's two replicas, consecutive chains)."""
    return x.reshape(-1, *x.shape[2:])


def _windows_of(x, idx):
    """Chains ``idx`` (``2 m``, consecutive pairs) as ``[m, 2, ...]`` windows."""
    return x[idx].reshape(-1, 2, *x.shape[1:])


def _shift(x, s):
    """``x[T, K, ...]`` moved ``s`` cells along ``K`` (``out[:, k] = x[:, k - s]``), zeros
    filled in: a chain's deviation between the absolute cells and the shock's frame."""
    out = np.zeros_like(x)
    k = x.shape[1]
    if s >= 0:
        out[:, s:] = x[:, : k - s]
    else:
        out[:, : k + s] = x[:, -s:]
    return out


def warm_path(name, tag):
    """A shock case's warm pool (:mod:`.warm_pool`)."""
    return RESULTS / f"warm_{name}_{tag}.npz"


class CaseRunner:
    """One training case: its reference, compiled window functions and running-bias table."""

    def __init__(self, name, model, args):
        ref_name = flows.CASES[name][0]
        self.name = name
        self.ref, self.meta = load_ref(ref_name, args.refs_tag)
        self.case = w2.make_case(name, self.meta, args.ppc)
        fn, _, pairing = operators.operator(model, self.case.kind)
        self.collide_fn = fn
        self.seeds = np.arange(self.ref["pool_x"].shape[0])
        if args.pool_seeds:
            self.seeds = np.array([k for k in args.pool_seeds if k < self.ref["pool_x"].shape[0]])
        self.case = self.case._replace(
            static=dataclasses.replace(self.case.static, pairing=pairing),
            modes=min(args.shock_modes, self.case.n_coarse),
        )
        # a long burn-in reaches a shock's steady state; a shear wave decays and has none (its
        # windows are matched at the reference's own times: S3's series is 51 blocks long)
        shock_burn = args.burn if args.shock_burn is None else args.shock_burn
        chains = self.case.kind == "shock" and bool(args.warm_pool)
        if chains and args.shock_burn is None:
            shock_burn = 0  # the chains are already relaxed
        self.blocks = args.blocks
        self.burn = shock_burn if self.case.kind == "shock" else args.burn
        self.window_batch = args.window_batch
        # every compiled function maps over a batch of windows (leading axis), each window
        # over its two replicas: one launch runs window_batch x 2 coarse runs side by side
        # (2026-11-11: one window at a time ran ~1e5 particle-steps/s, ~1e-3 of the card)
        replicas = jax.vmap(functools.partial(
            w2.burn_in, case=self.case, collide_fn=fn, blocks=self.burn),
            in_axes=(None, 0, 0, 0))  # fmt: skip
        self.burn_fn = jax.jit(jax.vmap(replicas, in_axes=(None, 0, 0, 0)))
        self.grad_fn = {
            est: jax.jit(jax.vmap(jax.value_and_grad(functools.partial(
                _objective, case=self.case, collide_fn=fn, blocks=args.blocks,
                step0=self.burn * BLOCK, estimator=est), has_aux=True),
                in_axes=(None, 0, 0, 0, 0, None, 0)))
            for est in ("pair", "bias")
        }  # fmt: skip
        self.thin = jax.jit(functools.partial(w2.pool_state, case=self.case))
        self.chains = None
        if chains:
            self._init_chains(args)
        n_fields = len(w2.LOSS_FIELDS[self.case.kind])
        if self.case.kind == "shear":
            mean = self.ref[f"mean_dx{self.case.dx_c:g}"]
            self.weights = w2.field_weights(mean, self.case)
            # the wave's bias in units of its own decaying flux, per block of the window, mode
            # and field: a viscosity-type error scales the flux by a factor growing from 1
            # through the window.  Keyed by absolute time instead, the 800 rows of a long wave
            # were visited < 1x each by the gate's 64 windows (a zero-mean gradient, lcsnr S2);
            # one entry for the whole window under-weighted S3 by ~1.5x (lcsnr v2, 2026-11-11)
            proj = np.asarray(w2.project(w2.flux(jnp.asarray(mean), "shear"), self.case))
            self.scale = np.sqrt(np.mean(proj**2, axis=1, keepdims=True))  # [T, 1, F]
            self.bias = np.zeros((args.blocks, 2, n_fields))
        else:
            target = w2.shock_target(self.ref["prof_raw"], self.meta["x_shock"], self.case,
                                     self.meta, 1)  # fmt: skip
            self.weights = w2.field_weights(target, self.case)
            self.weights[w2.LOSS_FIELDS["shock"].index("qx")] *= args.qx_weight
            self.scale = None
            self.bias = np.zeros((args.blocks, w2.loss_rows(self.case), n_fields))
        self.bias_n = np.zeros(self.bias.shape[0])
        self.decay = args.bias_decay

    def bias_at(self, rows):
        """``[T, K, F]``: the running bias at a window's rows."""
        if self.scale is None:
            return self.bias[rows]
        return self.bias * self.scale[rows]

    def draw(self, rng, with_seed=False):
        """A window: ``(states, cstates, keys, target, rows)`` (replicas on the leading axis),
        ``rows`` the bias-table rows (and the reference seed drawn from, ``with_seed``)."""
        ref, meta = self.ref, self.meta
        snaps = ref["pool_x"].shape[1]
        if self.case.kind == "shear":
            every = round((ref["snap_t"][1] - ref["snap_t"][0]) / (ref["t"][1] - ref["t"][0]))
            last = (len(ref["t"]) - 1 - self.burn - self.blocks) // every
            if last < 0:
                raise ValueError(f"{self.name}: burn {self.burn} + {self.blocks} blocks exceed "
                                 f"the reference's {len(ref['t']) - 1} blocks")  # fmt: skip
            s = int(rng.integers(0, min(snaps, last + 1)))
            b0 = s * every + self.burn
            rows = np.arange(b0 + 1, b0 + 1 + self.blocks)
            target = ref[f"mean_dx{self.case.dx_c:g}"][rows]
        else:
            s = int(rng.integers(0, snaps))
            rows = np.arange(self.blocks)
        seed = int(rng.choice(self.seeds))
        if self.case.kind == "shock":
            target = w2.shock_target(ref["prof_raw"], float(ref["snap_xs"][seed, s]), self.case,
                                     meta, self.blocks)  # fmt: skip
        key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
        k_thin, k_run = jax.random.split(key)
        inits = [self.thin(ref["pool_x"][seed, s, r], ref["pool_v"][seed, s, r],
                           ref["pool_alive"][seed, s, r], key=jax.random.fold_in(k_thin, r))
                 for r in range(2)]  # fmt: skip
        if any(int(i[2]) for i in inits):
            raise RuntimeError(f"{self.name}: thinning overflowed the coarse capacity")
        states = jax.tree.map(flows.stack, *[i[0] for i in inits])
        cstates = jax.tree.map(flows.stack, *[i[1] for i in inits])
        keys = jnp.stack([jax.random.fold_in(k_run, r) for r in range(2)])
        if with_seed:
            return states, cstates, keys, jnp.asarray(target), rows, seed
        return states, cstates, keys, jnp.asarray(target), rows

    def _init_chains(self, args):
        """``2 --batch`` chains drawn from the case's warm pool (``--warm-pool``)."""
        path = warm_path(self.name, args.warm_pool)
        pool = load_warm(path)
        if pool is None:
            raise SystemExit(f"no warm pool {path}: run examples.learned_collision.warm_pool")
        size = pool[0].x.shape[0]
        n = 2 * args.batch
        if size < n:
            raise SystemExit(f"{path}: {size} chains, the run needs {n} (2 x --batch)")
        pick = np.random.default_rng([args.seed, size]).choice(size, n, replace=False)
        self.chains = jax.tree.map(functools.partial(_index, i=jnp.asarray(pick)), pool)
        self.pool = pool  # the refill when most chains drifted together (_write_back)
        self.cursor = 0
        self.refreshed = 0
        self.refilled = 0
        sums = functools.partial(refdata._fine_sums, length=self.case.hi, nx=self.case.n_coarse)
        self.sums_fn = jax.jit(jax.vmap(sums))
        self.x_c = (np.arange(self.case.n_coarse) + 0.5) * self.case.dx_c
        self.ratio = flows.shock_states(flows.REFERENCES[flows.CASES[self.name][0]]["mach"])[
            "ratio"
        ]
        self.x_home = float(np.median(self.positions(self.chains[0])))
        self.drift = args.chain_drift

    def positions(self, states):
        """Each state's shock position [lambda_1] (the mass integral, as :mod:`.evaluate`)."""
        s = np.asarray(self.sums_fn(states))
        return np.atleast_1d(flows.shock_position(s[:, :, 0], self.x_c, self.ratio,
                                                  self.case.ppc_up))  # fmt: skip

    def _draw_chains(self, rng, n):
        """The next ``n`` windows of consecutive chain pairs (cycling), each replica's target
        placed at its own shock position (``[n, 2, T, C, 13]``); ``seeds`` hold the chain
        indices of the write-back and each replica's whole-cell offset from the pool's
        median (``[n, 2]``; 0 with ``--shock-modes``), which moves its deviation into the
        bias table's frame."""
        total = self.chains[0].x.shape[0] // 2
        win = (self.cursor + np.arange(n)) % total
        self.cursor = int((self.cursor + n) % total)
        idx = jnp.asarray(np.stack([2 * win, 2 * win + 1], axis=1).ravel())
        states, cstates = (jax.tree.map(functools.partial(_windows_of, idx=idx), c)
                           for c in self.chains)  # fmt: skip
        xs = self.positions(jax.tree.map(_pairs, states)).reshape(n, 2)
        targets = jnp.stack([jnp.stack([jnp.asarray(w2.shock_target(
            self.ref["prof_raw"], float(x), self.case, self.meta, self.blocks)) for x in pair])
            for pair in xs])  # fmt: skip
        shifts = np.rint((xs - self.x_home) / self.case.dx_c).astype(int)
        if self.case.modes > 0:  # the domain's modes: no cell frame to align
            shifts = np.zeros_like(shifts)
        keys = jnp.stack([jnp.stack([jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
                                     for _ in range(2)]) for _ in range(n)])  # fmt: skip
        rows = [np.arange(self.blocks)] * n
        return states, cstates, keys, targets, rows, (idx, shifts)

    def _write_back(self, idx, final, rng):
        """The windows' end states ``final = (states, cstates)`` (``[m, 2, ...]``) back into
        chains ``idx``; then every chain whose shock drifted past ``--chain-drift`` from the
        pool's median replaced by a copy of a random healthy one -- or, once fewer than half
        are healthy (the inflow-held shock's position random-walks, and K1's chains crept
        upstream together, ~1.2 lambda_1 in 68 iterations of lcchain_ctl_long), by random
        members of the warm pool (relaxed at the base operator, not the current model's
        state: the rarer, rougher refill)."""
        self.chains = tuple(jax.tree.map(functools.partial(_put, idx=idx), c,
                                         jax.tree.map(_pairs, f))
                            for c, f in zip(self.chains, final, strict=True))  # fmt: skip
        off = np.abs(self.positions(self.chains[0]) - self.x_home) > self.drift
        if not off.any():
            return
        healthy = np.flatnonzero(~off)
        if len(healthy) >= off.size // 2:
            source = self.chains
            src = rng.choice(healthy, int(off.sum()))
        else:
            source = self.pool
            src = rng.choice(self.pool[0].x.shape[0], int(off.sum()), replace=False)
            self.refilled += int(off.sum())
        dst = jnp.asarray(np.flatnonzero(off))
        self.chains = tuple(jax.tree.map(
            functools.partial(_put, idx=dst), c,
            jax.tree.map(functools.partial(_index, i=jnp.asarray(src)), p))
            for c, p in zip(self.chains, source, strict=True))  # fmt: skip
        self.refreshed += int(off.sum())

    def draw_batch(self, rng, n):
        """``n`` windows stacked on a new leading axis: ``(states, cstates, keys, targets,
        rows, seeds)`` (``rows``, ``seeds`` host lists; with chains ``seeds`` is the chain
        indices)."""
        if self.chains is not None:
            return self._draw_chains(rng, n)
        drawn = [self.draw(rng, with_seed=True) for _ in range(n)]
        states, cstates, keys, targets = (jax.tree.map(flows.stack, *[d[i] for d in drawn])
                                          for i in range(4))  # fmt: skip
        return states, cstates, keys, targets, [d[4] for d in drawn], [d[5] for d in drawn]

    def windows(self, theta, rng, estimator, n):
        """``n`` windows in batches of at most ``window_batch``: per window the pair loss,
        the mean deviation ``[T, K, F]``, the bias rows and the gradient (lists)."""
        pairs, devs, rows_all, grads = [], [], [], []
        while n > 0:
            m = min(n, self.window_batch)
            n -= m
            states, cstates, keys, targets, rows, seeds = self.draw_batch(rng, m)
            if self.burn:
                states, cstates = self.burn_fn(theta, states, cstates, keys)
            if self.chains is None:
                bias = np.stack([self.bias_at(r) for r in rows])
            else:  # the table (the shock's frame) moved to each replica's own cells
                shifts = seeds[1]
                bias = np.stack([np.stack([_shift(self.bias_at(r), int(s)) for s in sh])
                                 for r, sh in zip(rows, shifts, strict=True)])  # fmt: skip
            (_, (pair, d_pair, final)), grad = self.grad_fn[estimator](
                theta,
                states,
                cstates,
                keys,
                targets,
                jnp.asarray(self.weights),
                jnp.asarray(bias),
            )
            if self.chains is not None:
                self._write_back(seeds[0], final, rng)
            pair, d_pair = np.asarray(pair), np.asarray(d_pair)
            for i in range(m):
                pairs.append(float(pair[i]))
                if self.chains is None:
                    devs.append(d_pair[i].mean(axis=0))
                else:
                    devs.append(0.5 * sum(_shift(d_pair[i, r], -int(shifts[i, r]))
                                          for r in range(2)))  # fmt: skip
                rows_all.append(rows[i])
                grads.append(jax.tree.map(functools.partial(_index, i=i), grad))
        return pairs, devs, rows_all, grads

    def update_bias(self, rows, d_mean):
        """Fold one window's mean deviation into the running bias (a running mean for the
        first ``1 / decay`` updates of an entry, then an exponential average)."""
        if self.scale is not None:  # per window block: the deviation over the flux scale
            rows, d_mean = np.arange(len(rows)), d_mean / self.scale[rows]
        self.bias_n[rows] += 1.0
        beta = np.maximum(self.decay, 1.0 / self.bias_n[rows])[:, None, None]
        self.bias[rows] = (1.0 - beta) * self.bias[rows] + beta * d_mean

    def bias_norm(self):
        """``mean(w b^2)``: the squared bias the model is fitting (over the visited rows of a
        shock, over the whole reference series of a wave)."""
        if not (self.bias_n > 0).any():
            return float("nan")
        if self.scale is not None:  # over every window position in the series
            b = self.bias[None] * self.scale[:, None]
        else:
            b = self.bias[self.bias_n > 0]
        return float(np.mean(np.sum(self.weights * b**2, axis=-1)))


def _mean_of(*leaves, n):
    return sum(leaves) / n


def _index(a, i):
    return a[i]


def _scaled(scale, g):
    return scale * g


def model_options(args):
    """:func:`jax_dsmc.collide.relax.init_correction` options of the run's model."""
    opts = {}
    if args.model == "mlp":
        opts.update(width=args.width, depth=args.depth)
    if args.model != "scalar" and args.cubic:
        opts["cubic"] = True
    if args.model != "scalar" and args.outputs == "rich":
        if args.cubic:
            raise SystemExit("--cubic and --outputs rich are exclusive")
        opts["rich"] = True
    if args.model != "scalar" and args.features:
        stats = json.loads((RESULTS / args.features).read_text())
        k = len(stats["features"])
        if stats["features"] != list(relax.FEATURES[:k]):
            raise SystemExit(
                f"{args.features} has features {stats['features']}, the operator "
                f"{list(relax.FEATURES)}: rerun features.py (lcfeat)"
            )
        # a model reads the features its statistics cover (features_v2: the first nine)
        opts["phi_scale"] = stats["phi_scale"]
        opts["n_features"] = k
    return opts


def _fan_in(theta):
    """Each leaf's step divisor: a weight matrix's fan-in (``A``: ``[2, F]``, ``W{i}``:
    ``[fan_in, fan_out]``), 1 for the rest."""
    out = {}
    for k, v in theta.items():
        if k == "A":
            out[k] = float(v.shape[1])
        elif k.startswith("W"):
            out[k] = float(v.shape[0])
        else:
            out[k] = 1.0
    return out


def _divide(u, d):
    return u / d


def _frozen(args):
    """Leaves held fixed: the feature scale always, the global coefficients of a correction
    unless ``--train-intercept``."""
    frozen = {"phi_scale"}
    if args.model != "scalar" and not args.train_intercept:
        frozen |= {"log_tau", "log_pr"}
    return frozen


def _soft_threshold(a, t):
    return jnp.sign(a) * jnp.maximum(jnp.abs(a) - t, 0.0)


def _step(theta, grads, state, opt, args, frozen):
    """An Adam step divided by each leaf's fan-in, then the decoupled weight shrink and the
    proximal L1 (at the linear coefficients' step size)."""
    grads = {k: (jnp.zeros_like(g) if k in frozen else g) for k, g in grads.items()}
    updates, state = opt.update(grads, state, theta)
    fan = _fan_in(theta)
    updates = jax.tree.map(_divide, updates, fan)
    if args.weight_decay > 0.0:
        updates = {k: (u - args.weight_decay * theta[k] if k.startswith("W") else u)
                   for k, u in updates.items()}  # fmt: skip
    theta = optax.apply_updates(theta, updates)
    if args.l1 > 0.0 and "A" in theta:
        theta = dict(theta, A=_soft_threshold(theta["A"], args.lr / fan["A"] * args.l1))
    return theta, state, grads


def _accumulate(total, theta, w):
    return total + w * theta


def validate(theta, args, cases):
    """Forward metrics (:mod:`.evaluate`) of ``theta`` on ``cases``."""
    out = {}
    for name in cases:
        t0 = time.time()
        ref_name = flows.CASES[name][0]
        ref, meta = load_ref(ref_name, args.refs_tag)
        if meta["kind"] == "shear":
            out[name] = ev.eval_shear(name, args.model, theta, ref, meta, args.eval_seeds,
                                      args.ppc)  # fmt: skip
        else:
            res = ev.eval_shock(name, args.model, theta, ref, meta, args.eval_starts, 0.5, 1.0,
                                ppc=args.ppc)  # fmt: skip
            out[name] = {k: v for k, v in res.items() if k not in ("profile", "reference")}
        out[name]["seconds"] = time.time() - t0
        print(f"  validate {name}: " + ", ".join(
            f"{k} {v:.4f}" for k, v in out[name].items()
            if k in ("mu_ratio", "mu_ratio_se", "l2_txx", "l2_qx", "int_txx_ratio",
                     "int_qx_ratio")), flush=True)  # fmt: skip
    return out


def _jsonable(theta):
    return {k: np.asarray(v).tolist() for k, v in theta.items()}


def _iteration(theta, runners, rng, args):
    """One iteration's windows: the summed per-case mean gradients (each over its case's
    starting squared bias with ``--case-norm``), the pair losses, the seconds and the
    gradient norms per case."""
    grads = jax.tree.map(jnp.zeros_like, theta)
    pairs, seconds, norms = {}, {}, {}
    for r in runners:
        t1 = time.time()
        batch = args.shear_batch if r.case.kind == "shear" else args.batch
        vals, devs, rows, gs = r.windows(theta, rng, args.estimator, batch)
        for row, d in zip(rows, devs, strict=True):
            r.update_bias(row, d)
        # each case's mean gradient, equally weighted across cases
        mean = jax.tree.map(functools.partial(_mean_of, n=batch), *gs)
        if args.case_norm:
            mean = jax.tree.map(functools.partial(_scaled, 1.0 / r.bias2_start), mean)
        norms[r.name] = float(optax.global_norm(mean))
        grads = jax.tree.map(jnp.add, grads, mean)
        pairs[r.name] = float(np.mean(vals))
        seconds[r.name] = time.time() - t1
    return grads, pairs, seconds, norms


def _chain_log(runners):
    """The shock chains' positions and replacements so far (``{}`` without chains)."""
    chained = [r for r in runners if r.chains is not None]
    if not chained:
        return {}
    return {"chains": {r.name: {"x_median": float(np.median(r.positions(r.chains[0]))),
                                "x_home": r.x_home, "refreshed": r.refreshed,
                                "refilled": r.refilled}
                       for r in chained}}  # fmt: skip


def _fill_bias(theta, runners, rng, args):
    """The bias estimator's tables from ``--bias-init`` forward windows per case, and each
    case's starting squared bias (``--case-norm``)."""
    if args.case_norm and args.estimator != "bias":
        raise SystemExit("--case-norm needs the bias estimator (its starting squared bias)")
    if args.estimator != "bias":
        return
    t0 = time.time()
    for r in runners:
        _, devs, rows, _ = r.windows(theta, rng, "pair", args.bias_init)
        for row, d in zip(rows, devs, strict=True):
            r.update_bias(row, d)
        r.bias2_start = r.bias_norm()
    print(f"bias tables filled ({args.bias_init} windows per case), "
          f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip


def train(args):
    """The training loop (module docstring); returns the log dict."""
    _, theta, _ = operators.operator(args.model, "shear", jax.random.PRNGKey(args.seed),
                                     **model_options(args))  # fmt: skip
    runners = [CaseRunner(name, args.model, args) for name in args.cases]
    rng = np.random.default_rng(args.seed)
    frozen = _frozen(args)
    t0 = time.time()
    _fill_bias(theta, runners, rng, args)
    opt = optax.adam(args.lr)
    state = opt.init(theta)
    avg_from = args.iters - max(1, round(args.average * args.iters))
    avg = jax.tree.map(jnp.zeros_like, theta)
    log = []
    out_path = RESULTS / f"train_{args.model}_{args.tag}.json"
    out = {"args": vars(args), "theta": _jsonable(theta), "log": log}
    for it in range(args.iters):
        grads, pairs, seconds, norms = _iteration(theta, runners, rng, args)
        grads = jax.tree.map(functools.partial(_scaled, 1.0 / len(runners)), grads)
        theta, state, grads = _step(theta, grads, state, opt, args, frozen)
        row = {"iter": it, "pair": pairs, "bias2": {r.name: r.bias_norm() for r in runners},
               "grad_norm": float(optax.global_norm(grads)), "case_grad_norm": norms,
               "case_seconds": seconds,
               "log_tau": float(theta["log_tau"]), "log_pr": float(theta["log_pr"]),
               "seconds": time.time() - t0}  # fmt: skip
        if "A" in theta:
            row["A"] = np.asarray(theta["A"]).tolist()
        row.update(_chain_log(runners))
        log.append(row)
        print(f"it {it:4d}  bias2 {sum(row['bias2'].values()): .4e}  pair "
              f"{sum(pairs.values()): .4e}  |g| {row['grad_norm']:.3e}  log_tau "
              f"{row['log_tau']:+.4f} log_pr {row['log_pr']:+.4f}  {row['seconds']:.0f} s",
              flush=True)  # fmt: skip
        if it >= avg_from:
            avg = jax.tree.map(functools.partial(_accumulate, w=1.0 / (args.iters - avg_from)),
                               avg, theta)  # fmt: skip
        out["theta"] = _jsonable(theta)
        out_path.write_text(json.dumps(out))
    if args.iters > 0:
        theta = avg
        out["theta_last"], out["theta"] = out["theta"], _jsonable(theta)
        out_path.write_text(json.dumps(out))
    checks = list(args.validate) + (list(args.cases) if args.validate_train else [])
    if checks:
        t1 = time.time()
        out["validation"] = validate(theta, args, checks)
        out["validation_seconds"] = time.time() - t1
        out_path.write_text(json.dumps(out))
    return out


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="linear", choices=operators.MODELS)
    ap.add_argument("--cases", nargs="+", default=list(TRAIN_CASES))
    ap.add_argument("--estimator", default="bias", choices=("bias", "pair"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32, help="shock windows per case per iteration")
    ap.add_argument("--shear-batch", type=int, default=32,
                    help="shear-wave windows per case per iteration (~6x cheaper than a shock's, "
                         "and a low-amplitude wave's flux signal is weak)")  # fmt: skip
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--burn", type=int, default=2)
    ap.add_argument(
        "--shock-burn",
        type=int,
        default=None,
        help="the shocks' burn-in blocks (default --burn; the waves keep --burn)",
    )
    ap.add_argument(
        "--shock-modes",
        type=int,
        default=0,
        help="shock loss on the first K cosine modes (1 = the integrals; 0 = per cell)",
    )
    ap.add_argument("--qx-weight", type=float, default=1.0, help="shock heat-flux weight factor")
    ap.add_argument("--warm-pool", default="", help="shock chains from warm_<case>_<TAG>.npz")
    ap.add_argument(
        "--cubic",
        action="store_true",
        help="a third output: the cubic Fokker-Planck drift coefficient (relax.cubic_drift)",
    )
    ap.add_argument(
        "--outputs",
        default="rates",
        choices=("rates", "rich"),
        help="the correction's outputs: the two rates, or relax.RICH_OUTPUTS (stress and "
        "heat-flux targets, reconstruction gains, noise shape)",
    )
    ap.add_argument(
        "--case-norm",
        action="store_true",
        help="each case's gradient over its squared bias after the fill (relative progress; "
        "lcconv: K1's large, noisy squared bias led while K4's doubled)",
    )
    ap.add_argument(
        "--chain-drift",
        type=float,
        default=6.0,
        help="a chain whose shock moved this far [lambda_1] from the pool's median is replaced",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="per-iteration output change: Adam's step over each weight's fan-in",
    )
    ap.add_argument(
        "--average",
        type=float,
        default=0.25,
        help="final theta = mean of the last fraction of the iterates",
    )
    ap.add_argument("--bias-decay", type=float, default=1e-3,
                    help="the running bias's forgetting rate (a running mean for its first "
                         "1 / decay windows): long memory -- a per-window deviation is mostly "
                         "noise (lcsnr 2026-11-11)")  # fmt: skip
    ap.add_argument("--bias-init", type=int, default=256)
    ap.add_argument(
        "--window-batch", type=int, default=32, help="windows per compiled call (x 2 replicas each)"
    )
    ap.add_argument("--ppc", type=int, default=flows.PPC_COARSE)
    ap.add_argument("--width", type=int, default=64, help="MLP hidden width")
    ap.add_argument("--depth", type=int, default=2, help="MLP hidden layers")
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="per-iteration shrink fraction of the MLP weights (independent of lr)",
    )
    ap.add_argument("--l1", type=float, default=0.0, help="proximal L1 on the linear A")
    ap.add_argument("--train-intercept", action="store_true")
    ap.add_argument(
        "--features",
        default="features_v2.json",
        help="feature statistics (features.py; '' = nominal scales)",
    )
    ap.add_argument(
        "--pool-seeds",
        type=int,
        nargs="*",
        default=[],
        help="reference seeds to draw snapshots from (default all)",
    )
    ap.add_argument("--validate", nargs="*", default=[], help="cases evaluated at the end")
    ap.add_argument("--validate-train", action="store_true")
    ap.add_argument("--eval-seeds", type=int, default=128)
    ap.add_argument("--eval-starts", type=int, default=16)
    ap.add_argument("--refs-tag", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    return ap


def main(argv=None):
    train(parser().parse_args(argv))


if __name__ == "__main__":
    main()
