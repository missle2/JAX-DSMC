"""Is the window gradient a descent direction of the steady error?  (``lcdirfd``, 2026-11-13.)

Trained on K4 alone, K4 got worse on both integrals and its own window loss rose after ~40
iterations (lccase), while K8-trained models fixed K4 -- K4's gradient leads away from a
better K4.  This tests the estimator where it fails: at iterates of a training run
(``--run``; ``--at`` iteration indices of its log), the window gradient ``g`` exactly as the
run computed it (train.CaseRunner: the run's arguments, warm chains first settled
``--settle`` windows under the iterate, the bias table filled, ``--grad-windows`` windows of
the bias estimator) and the steady loss (:func:`.evaluate.eval_shock`, the same starts and
keys at every point -- common random numbers) at ``theta -+ h g / |g|``.  The steady loss
is the run's objective on the steady profile: ``sum_f w_f mean_cells (f - f_ref)^2 /
mean(f_ref^2)``, ``w = (1, --qx-weight)`` for ``(tau_xx, q_x)``.  Also along the run's own
update direction (``A[i + k] - A[i - k]``, Adam's processed steps).  A descent direction has
``L(theta - h u) < L(theta + h u)``.

    PYTHONPATH=. python3 -m examples.learned_collision.direction_fd \
        --run train_linear_lccase_only_K4.json --at 0 40 150 299

Writes ``examples/learned_collision/results/direction_fd_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples.learned_collision import evaluate as ev
from examples.learned_collision import flows, operators, train
from examples.learned_collision.shear_wave import RESULTS


def steady(theta, name, ref, meta, args, qx_weight, ppc):
    """The steady loss and integrals of model ``theta`` on shock ``name`` (the same seed:
    common random numbers across calls)."""
    res = ev.eval_shock(name, "linear", theta, ref, meta, args.starts, args.relax,
                        args.sample, seed=args.seed, ppc=ppc)  # fmt: skip
    p, r = np.array(res["profile"]), np.array(res["reference"])
    loss = float(np.mean((p[:, 2] - r[:, 2]) ** 2) / np.mean(r[:, 2] ** 2)
                 + qx_weight * np.mean((p[:, 3] - r[:, 3]) ** 2) / np.mean(r[:, 3] ** 2))  # fmt: skip
    return {"loss": loss, "int_txx": res["int_txx_ratio"], "int_qx": res["int_qx_ratio"],
            "l2_txx": res["l2_txx"], "l2_qx": res["l2_qx"]}  # fmt: skip


def window_gradient(theta, run_args, name, args, rng):
    """``g`` (``f64[rows, F]``, the ``A`` leaf) of the run's estimator at ``theta``."""
    r = train.CaseRunner(name, "linear", run_args)
    r.windows(theta, rng, "pair", args.settle)  # the chains relax under this iterate
    _, devs, rows, _ = r.windows(theta, rng, "pair", run_args.bias_init)
    for row, d in zip(rows, devs, strict=True):
        r.update_bias(row, d)
    _, _, _, gs = r.windows(theta, rng, run_args.estimator, args.grad_windows)
    g = np.mean([np.asarray(x["A"]) for x in gs], axis=0)
    se = np.std([np.asarray(x["A"]) for x in gs], axis=0, ddof=1) / np.sqrt(len(gs))
    return g, se, r.bias_norm()


def _directions(log, i, g, k):
    out = {"grad": -g / np.linalg.norm(g)}
    if k <= i < len(log) - k:
        step = np.asarray(log[i + k]["A"]) - np.asarray(log[i - k]["A"])
        out["train"] = step / np.linalg.norm(step)
    return out


def study_point(i, log, base, run_args, name, ref, meta, args, rng):
    a = np.zeros_like(np.asarray(base["A"])) if i == 0 else np.asarray(log[i]["A"])
    theta = dict(base, A=jnp.asarray(a))
    t0 = time.time()
    g, g_se, bias2 = window_gradient(theta, run_args, name, args, rng)
    centre = steady(theta, name, ref, meta, args, run_args.qx_weight, run_args.ppc)
    row = {"iter": i, "bias2": bias2, "grad": g.tolist(), "grad_se": g_se.tolist(),
           "grad_snr": float(np.linalg.norm(g) / np.linalg.norm(g_se)), "centre": centre,
           "directions": {}}  # fmt: skip
    for label, u in _directions(log, i, g, args.k).items():
        cuts = []
        for h in args.h:
            down = steady(dict(theta, A=jnp.asarray(a + h * u)), name, ref, meta, args,
                          run_args.qx_weight, run_args.ppc)  # fmt: skip
            up = steady(dict(theta, A=jnp.asarray(a - h * u)), name, ref, meta, args,
                        run_args.qx_weight, run_args.ppc)  # fmt: skip
            slope = (down["loss"] - up["loss"]) / (2.0 * h)
            cuts.append({"h": h, "plus": down, "minus": up, "slope": slope})
            print(f"iter {i:3d} {label:5s} h {h:.3f}: L(theta + h u) {down['loss']:.4e} "
                  f"L(theta) {centre['loss']:.4e} L(theta - h u) {up['loss']:.4e} -> slope "
                  f"{slope:+.3e} ({'DESCENT' if slope < 0 else 'ascent'}); int txx "
                  f"{up['int_txx']:.3f}/{down['int_txx']:.3f} qx {up['int_qx']:.3f}/"
                  f"{down['int_qx']:.3f}", flush=True)  # fmt: skip
        row["directions"][label] = cuts
    if "train" in row["directions"]:
        u_g = -g / np.linalg.norm(g)
        step = np.asarray(log[i + args.k]["A"]) - np.asarray(log[i - args.k]["A"])
        row["cos_grad_train"] = float(u_g.ravel() @ step.ravel() / np.linalg.norm(step))
    print(f"iter {i:3d}: window bias2 {bias2:.3e}, |g| {np.linalg.norm(g):.3e} (snr "
          f"{row['grad_snr']:.1f}), steady loss {centre['loss']:.4e} int txx "
          f"{centre['int_txx']:.3f} qx {centre['int_qx']:.3f}  {time.time() - t0:.0f} s",
          flush=True)  # fmt: skip
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="train_linear_lccase_only_K4.json")
    ap.add_argument("--case", default="K4")
    ap.add_argument("--at", type=int, nargs="+", default=[0, 40, 150, 299])
    ap.add_argument("--h", type=float, nargs="+", default=[0.02, 0.05])
    ap.add_argument("--k", type=int, default=10, help="the run's update over iterations i -+ k")
    ap.add_argument("--starts", type=int, default=64)
    ap.add_argument("--relax", type=float, default=0.5)
    ap.add_argument("--sample", type=float, default=1.0)
    ap.add_argument("--settle", type=int, default=512, help="chain windows under the iterate")
    ap.add_argument("--grad-windows", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)
    run = json.loads((RESULTS / args.run).read_text())
    run_args = argparse.Namespace(**(vars(train.parser().parse_args([])) | run["args"]))
    run_args.cases = [args.case]
    base = operators.operator("linear", "shock", jax.random.PRNGKey(run_args.seed),
                              **train.model_options(run_args))[1]  # fmt: skip
    ref, meta = ev.load_ref(flows.CASES[args.case][0], run_args.refs_tag)
    rng = np.random.default_rng(args.seed)
    rows = []
    for at in args.at:
        i = at if at >= 0 else len(run["log"]) + at
        rows.append(study_point(i, run["log"], base, run_args, args.case, ref, meta, args, rng))
        (RESULTS / f"direction_fd_{args.tag}.json").write_text(
            json.dumps({"args": vars(args), "rows": rows})
        )


if __name__ == "__main__":
    main()
