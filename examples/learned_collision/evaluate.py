"""Forward evaluation of collision operators on the round-1 cases (:mod:`.flows`): the
``lcbase`` baseline table and the ``lceval`` held-out test.

* **Shear wave** -- ``mu_eff / mu_ref``: the decay rate of the seed-mean wave amplitude of
  ``--seeds`` coarse runs from fresh initial waves (:func:`.shear_wave.fit_decay`, skipping
  20 collision times), over the fine reference's (its seed-mean coarse ``u_y`` projected on
  the wave), with a bootstrap standard error over seeds.
* **Shock** -- coarse runs started from thinned reference snapshots (pools), relaxed
  ``--relax`` flow-through times (``L / u_2``) and averaged over ``--sample`` more: the
  coarse profile against the reference's shock-frame profile placed at the coarse runs' mean
  shock position (the mass-integral position, so a uniform shift is not an error):
  normalised L2 errors of ``n``, ``T``, ``tau_xx``, ``q_x``, the shift-invariant integrals
  ``int tau_xx dx`` and ``int q_x dx`` against the reference's, and the shock drift.
  Standard errors over the batch of independent starts.

    PYTHONPATH=. python3 -m examples.learned_collision.evaluate --cases S1 K3 \
        --operators ntc esfp_recon [--theta train_linear_v1.json] [--refs-tag v1]

Writes ``examples/learned_collision/results/eval_<tag>.json`` (one row per case and
operator, rewritten after every row).
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

from examples.learned_collision import flows, moments, operators, refdata
from examples.learned_collision import windows2d as w2
from examples.learned_collision.shear_wave import BLOCK, RESULTS, amplitude, fit_decay
from examples.learned_collision.windows import wave_basis


def load_ref(name, tag):
    """A reference's arrays and metadata (:mod:`.refdata` output)."""
    d = np.load(RESULTS / f"ref2d_{name}_{tag}.npz")
    return {k: d[k] for k in d.files if k != "meta"}, json.loads(str(d["meta"]))


def load_theta(path):
    """Trained parameters from a :mod:`.train` result (``theta`` of its last row)."""
    log = json.loads((RESULTS / path).read_text())
    return {k: jnp.asarray(v, jnp.float64) for k, v in log["theta"].items()}, log["args"]["model"]


def _bootstrap_rate(t, amps, t0, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    gamma = fit_decay(t, amps.mean(axis=0), t0)
    boot = [fit_decay(t, amps[rng.integers(0, len(amps), len(amps))].mean(axis=0), t0)
            for _ in range(n_boot)]  # fmt: skip
    return gamma, float(np.std(boot, ddof=1))


def _skip(t):
    """The start of the decay fit: 20 collision times, or a fifth of a shorter series."""
    return min(20.0, 0.2 * float(t[-1]))


def reference_rate(ref, meta, dx_c):
    """The fine reference's wave decay rate [1 / (lambda / v_th)]."""
    mean = ref[f"mean_dx{dx_c:g}"]  # [T, C, 13]
    amp = np.asarray(wave_basis(mean.shape[1]))[1] @ mean[:, :, 2].T
    return fit_decay(ref["t"], amp, _skip(ref["t"]))


def eval_shear(case_name, op_name, theta, ref, meta, seeds, ppc, batch_particles=2e6):
    """``mu_eff / mu_ref`` (module docstring) of one operator on shear case ``case_name``."""
    ref_name, dx_c, _ = flows.CASES[case_name]
    spec = flows.REFERENCES[ref_name]
    fn, theta0, pairing = operators.operator(op_name, "shear")
    theta = theta0 if theta is None else theta
    _, _, grid, sp, params, static, _, c, bc = flows.setup(spec, dx_c, ppc, 0)
    faces, capacity, res, wall = flows.run_kwargs(bc)
    static = dataclasses.replace(static, pairing=pairing)
    blocks = len(ref["t"]) - 1
    batch = max(1, min(seeds, int(batch_particles // c["n_part"])))
    amp = jax.jit(jax.vmap(functools.partial(amplitude, k=c["k"], vth=c["vth"])))
    series, clipped = [], 0
    for first in range(0, seeds, batch):
        ids = range(first, min(first + batch, seeds))
        cases = [flows.setup(spec, dx_c, ppc, 1000 + s) for s in ids]
        states = jax.tree.map(flows.stack, *[x[0] for x in cases])
        cstates = jax.tree.map(flows.stack, *[x[1] for x in cases])
        keys = jnp.stack([x[6] for x in cases])
        a = [np.asarray(amp(states))]
        for b in range(blocks):
            states, cstates, clip, _ = flows.advance(
                states, cstates, keys, grid, sp, params, b * BLOCK, theta, res, wall,
                static=static, collide_fn=fn, faces=faces, capacity=capacity,
            )  # fmt: skip
            clipped += int(clip)
            a.append(np.asarray(amp(states)))
        series.append(np.stack(a, axis=1))
    amps = np.concatenate(series)
    t = ref["t"][: blocks + 1]
    gamma, se = _bootstrap_rate(t, amps, _skip(t))
    gamma_ref = reference_rate(ref, meta, dx_c)
    return {"mu_ratio": gamma / gamma_ref, "mu_ratio_se": se / gamma_ref, "gamma": gamma,
            "gamma_ref": gamma_ref, "seeds": seeds, "attempts_clipped": clipped}  # fmt: skip


def _fields(raw, k_cells, ppc_up, vth):
    """``(n, T, tau_xx, q_x)`` per cell from mean raw sums ``raw[C, 16]`` (SI velocities)."""
    m = np.asarray(moments.central(jnp.asarray(raw[:, 0]), jnp.asarray(raw[:, 1:4]),
                                   jnp.asarray(raw[:, 4:13].reshape(-1, 3, 3)),
                                   jnp.asarray(raw[:, 13:16]), unbiased=False))  # fmt: skip
    temp = (m[:, 4] + m[:, 5] + m[:, 6]) / 3.0
    return np.stack([raw[:, 0] / (k_cells * ppc_up), temp / vth**2,
                     (m[:, 4] - temp) / vth**2, m[:, 10] / vth**3], axis=1)  # fmt: skip


def _starts(ref, case, n_starts, rng):
    """``n_starts`` coarse initial states from random (seed, snapshot, pool) of the reference
    -- the pools are disjoint and the stripes and thinning random, so independent -- and
    the snapshots' shock positions."""
    s_n, t_n, k_n = ref["pool_x"].shape[:3]
    picks = [(int(rng.integers(s_n)), int(rng.integers(t_n)), int(rng.integers(k_n)))
             for _ in range(n_starts)]  # fmt: skip
    out = [w2.pool_state(ref["pool_x"][s, t, k], ref["pool_v"][s, t, k],
                         ref["pool_alive"][s, t, k], case,
                         jax.random.PRNGKey(int(rng.integers(2**31 - 1))))
           for s, t, k in picks]  # fmt: skip
    if any(int(o[2]) for o in out):
        raise RuntimeError("thinning overflowed the coarse capacity")
    states = jax.tree.map(flows.stack, *[o[0] for o in out])
    cstates = jax.tree.map(flows.stack, *[o[1] for o in out])
    return states, cstates, np.array([ref["snap_xs"][s, t] for s, t, _ in picks])


def eval_shock(case_name, op_name, theta, ref, meta, starts, relax, sample, seed=0,
               ppc=flows.PPC_COARSE):  # fmt: skip
    """Profile errors (module docstring) of one operator on shock case ``case_name``."""
    case = w2.make_case(case_name, meta, ppc)
    fn, theta0, pairing = operators.operator(op_name, "shock")
    theta = theta0 if theta is None else theta
    static = dataclasses.replace(case.static, pairing=pairing)
    faces, capacity, res, wall = flows.run_kwargs(case.bc)
    rng = np.random.default_rng(seed)
    states, cstates, xs0 = _starts(ref, case, starts, rng)
    keys = jnp.stack([jax.random.PRNGKey(int(rng.integers(2**31 - 1))) for _ in range(starts)])
    st = flows.shock_states(flows.REFERENCES[flows.CASES[case_name][0]]["mach"])
    block_time = float(case.params.dt) * BLOCK
    flow_through = (case.hi - case.lo) / st["u2"]
    n_relax = math.ceil(relax * flow_through / block_time)
    n_sample = math.ceil(sample * flow_through / block_time)
    sums = jax.jit(jax.vmap(functools.partial(refdata._fine_sums, length=case.hi,
                                              nx=case.n_coarse)))  # fmt: skip
    x_c = (np.arange(case.n_coarse) + 0.5) * case.dx_c
    acc = np.zeros((starts, case.n_coarse, 16))
    pos, clipped, short = [], 0, 0.0
    for b in range(n_relax + n_sample):
        states, cstates, clip, sh = flows.advance(
            states, cstates, keys, case.grid, case.sp, case.params, b * BLOCK, theta, res, wall,
            static=static, collide_fn=fn, faces=faces, capacity=capacity,
        )  # fmt: skip
        clipped += int(clip)
        short = max(short, float(sh))
        if b >= n_relax:
            s = np.asarray(sums(states))
            acc += s
            pos.append(flows.shock_position(s[:, :, 0], x_c, st["ratio"], case.ppc_up))
    acc /= n_sample
    pos = np.stack(pos)  # [samples, starts]
    return _shock_metrics(acc, pos, xs0, ref, meta, case) | {
        "starts": starts, "attempts_clipped": clipped, "inflow_shortfall": short,
        "relax_blocks": n_relax, "sample_blocks": n_sample}  # fmt: skip


def _shock_metrics(acc, pos, xs0, ref, meta, case):
    """Errors of the coarse mean profile ``acc[starts, C, 16]`` against the reference."""
    vth = meta["vth_ms"]
    per_start = np.stack([_fields(a, 1, case.ppc_up, vth) for a in acc])  # [S, C, 4]
    mean = per_start.mean(axis=0)
    x_s = float(pos.mean())
    raw = w2.placed_raw(ref["prof_raw"], x_s, case, meta)
    tgt = _fields(raw, round(case.dx_c / meta["dx_fine"]), meta["ppc_up"], vth)
    jump = np.abs(tgt[-1] - tgt[0]) + np.abs(tgt).max(axis=0) * 1e-6
    names = ("n", "T", "txx", "qx")
    out = {"x_shock": x_s, "drift": float((pos[-1] - xs0).mean()),
           "drift_se": float((pos[-1] - xs0).std(ddof=1) / math.sqrt(len(xs0)))}  # fmt: skip
    for i, f in enumerate(names):
        scale = jump[i] if f in ("n", "T") else np.abs(tgt[:, i]).max()
        errs = np.sqrt(np.mean((per_start[:, :, i] - tgt[:, i]) ** 2, axis=1)) / scale
        out[f"l2_{f}"] = float(np.sqrt(np.mean((mean[:, i] - tgt[:, i]) ** 2)) / scale)
        out[f"l2_{f}_se"] = float(errs.std(ddof=1) / math.sqrt(len(errs)))
    for i, f in ((2, "txx"), (3, "qx")):
        integ = per_start[:, :, i].sum(axis=1) * case.dx_c
        ref_integ = tgt[:, i].sum() * case.dx_c
        out[f"int_{f}_ratio"] = float(integ.mean() / ref_integ)
        out[f"int_{f}_ratio_se"] = float(integ.std(ddof=1) / math.sqrt(len(integ)) / abs(ref_integ))
    out["profile"] = mean.tolist()
    out["reference"] = tgt.tolist()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", nargs="+", default=list(flows.CASES))
    ap.add_argument("--operators", nargs="*", default=list(operators.BASELINES))
    ap.add_argument("--theta", nargs="*", default=[], help="trained models (train_*.json)")
    ap.add_argument("--refs-tag", default="v1")
    ap.add_argument("--seeds", type=int, default=64, help="shear-wave runs per operator")
    ap.add_argument("--starts", type=int, default=32, help="shock runs per operator")
    ap.add_argument("--relax", type=float, default=0.5, help="shock relaxation, flow-throughs")
    ap.add_argument("--sample", type=float, default=1.0, help="shock sampling, flow-throughs")
    ap.add_argument("--ppc", type=int, default=flows.PPC_COARSE, help="coarse particles/cell")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)
    runs = [(name, None, name) for name in args.operators]
    for path in args.theta:
        theta, model = load_theta(path)
        runs.append((model, theta, path.removesuffix(".json")))
    rows, refs, t0 = [], {}, time.time()
    out_path = RESULTS / f"eval_{args.tag}.json"
    for case_name in args.cases:
        ref_name, dx_c, role = flows.CASES[case_name]
        if ref_name not in refs:
            refs[ref_name] = load_ref(ref_name, args.refs_tag)
        ref, meta = refs[ref_name]
        for op_name, theta, label in runs:
            t1 = time.time()
            if meta["kind"] == "shear":
                res = eval_shear(case_name, op_name, theta, ref, meta, args.seeds, args.ppc)
                summary = f"mu/mu_ref {res['mu_ratio']:.4f} +- {res['mu_ratio_se']:.4f}"
            else:
                res = eval_shock(case_name, op_name, theta, ref, meta, args.starts, args.relax,
                                 args.sample, ppc=args.ppc)  # fmt: skip
                summary = (f"L2 n {res['l2_n']:.3f} T {res['l2_T']:.3f} txx {res['l2_txx']:.3f}"
                           f" qx {res['l2_qx']:.3f}  int txx {res['int_txx_ratio']:.3f} qx "
                           f"{res['int_qx_ratio']:.3f}  drift {res['drift']:+.2f}")  # fmt: skip
            rows.append({"case": case_name, "role": role, "reference": ref_name, "dx_c": dx_c,
                         "operator": label, "seconds": time.time() - t1, **res})  # fmt: skip
            print(f"{case_name} ({role}) {label:14s} {summary}  clipped "
                  f"{res['attempts_clipped']}  {time.time() - t0:.0f} s", flush=True)  # fmt: skip
            out_path.write_text(json.dumps({"args": vars(args), "rows": rows}))


if __name__ == "__main__":
    main()
