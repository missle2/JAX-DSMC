"""The window-gradient gate (``lcsnr``): signal-to-noise of the two window-gradient
estimators (:mod:`.windows2d`) at the base operator, against window length, **including the
running bias's own noise** (2026-11-11).

The running-bias gradient ``b grad m`` multiplies a bias table filled from earlier windows by
the current windows' sensitivities.  Freezing one table and measuring the spread over fresh
windows (the first version of this gate) leaves the table's noise out: a noise-dominated table
times low-noise sensitivities gave a tight, wrong gradient (SNR 3-5 that flipped sign with the
window length).  So, per case and window length, ``--fills`` independent repetitions of:

1. fill a fresh table from ``--bias-windows`` windows (forward-only);
2. ``--windows`` fresh windows, each giving both estimators' gradients on the *same* window
   (same pools, stripes, thinning and noise).

The pair estimator's windows are all independent: its mean and standard error pool every
fill.  The bias estimator's standard error comes from the spread of its per-fill means (each
fill: a new table and new windows), so it carries the table's noise.  Also reported: the pair
loss over all windows (an unbiased squared bias: is the bias detected at all?), the tables'
squared bias (their noise floor shows as an excess over the pair loss), and each
parameter's agreement between the estimators (same expectation).

    PYTHONPATH=. python3 -m examples.learned_collision.grad_snr2d [--cases S3 S4 K1 K8]
        [--blocks-list 5 10] [--fills 6] [--windows 256] [--bias-windows 512]

Writes ``examples/learned_collision/results/grad_snr2d_<tag>.json``.
"""

from __future__ import annotations

import json
import time

import jax
import numpy as np

from examples.learned_collision import operators
from examples.learned_collision.shear_wave import RESULTS
from examples.learned_collision.train import CaseRunner, parser


def _flat(grad):
    return np.concatenate([np.asarray(grad[k], dtype=float).ravel() for k in sorted(grad)])


def _names(theta):
    out = []
    for k in sorted(theta):
        n = int(np.asarray(theta[k]).size)
        out += [k] if n == 1 else [f"{k}[{i}]" for i in range(n)]
    return out


def study(name, blocks, args):
    """Both estimators' gradient statistics on case ``name`` at window length ``blocks``."""
    args.blocks = blocks
    runner = CaseRunner(name, args.model, args)
    _, theta, _ = operators.operator(args.model, runner.case.kind, jax.random.PRNGKey(0))
    pair_g, fill_means, within_se, pair_loss, table_b2 = [], [], [], [], []
    for f in range(args.fills):
        runner.bias[:] = 0.0
        runner.bias_n[:] = 0.0
        rng = np.random.default_rng((args.seed, f, 1))
        _, devs, rows, _ = runner.windows(theta, rng, "pair", args.bias_windows)
        for r, d in zip(rows, devs, strict=True):
            runner.update_bias(r, d)
        table_b2.append(runner.bias_norm())
        out = {}
        for est in ("pair", "bias"):  # identical windows: the same generator state
            pairs, _, _, grads = runner.windows(theta, np.random.default_rng((args.seed, f, 2)),
                                                est, args.windows)  # fmt: skip
            out[est] = np.stack([_flat(g) for g in grads])
            if est == "pair":
                pair_loss += pairs
        pair_g.append(out["pair"])
        fill_means.append(out["bias"].mean(axis=0))
        within_se.append(out["bias"].std(axis=0, ddof=1) / np.sqrt(len(out["bias"])))
    pair_g = np.concatenate(pair_g)
    p_mean, p_se = pair_g.mean(0), pair_g.std(0, ddof=1) / np.sqrt(len(pair_g))
    fill_means = np.stack(fill_means)
    b_mean = fill_means.mean(0)
    b_se = fill_means.std(0, ddof=1) / np.sqrt(len(fill_means))
    loss = np.array(pair_loss)
    return {
        "case": name, "blocks": blocks, "params": _names(theta),
        "pair_loss": float(loss.mean()),
        "pair_loss_se": float(loss.std(ddof=1) / np.sqrt(len(loss))),
        "table_bias2": table_b2,
        "pair_mean": p_mean.tolist(), "pair_se": p_se.tolist(),
        "pair_snr": (np.abs(p_mean) / p_se).tolist(),
        "bias_mean": b_mean.tolist(), "bias_se": b_se.tolist(),
        "bias_snr": (np.abs(b_mean) / b_se).tolist(),
        "bias_within_se": np.mean(within_se, axis=0).tolist(),
        "agreement": ((p_mean - b_mean) / np.hypot(p_se, b_se)).tolist(),
        "windows": len(pair_g),
    }  # fmt: skip


def main(argv=None):
    ap = parser()
    ap.description = __doc__.split("\n\n")[0]
    ap.set_defaults(cases=["S3", "S4", "K1", "K8"], model="linear", features="")
    ap.add_argument("--blocks-list", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--fills", type=int, default=6)
    ap.add_argument("--windows", type=int, default=256, help="gradient windows per fill")
    ap.add_argument("--bias-windows", type=int, default=512, help="table windows per fill")
    args = ap.parse_args(argv)
    rows, t0 = [], time.time()
    path = RESULTS / f"grad_snr2d_{args.tag}.json"
    for name in args.cases:
        for blocks in args.blocks_list:
            row = study(name, blocks, args)
            row["seconds"] = time.time() - t0
            rows.append(row)
            idx = [row["params"].index(k) for k in ("log_tau", "log_pr")]
            a_idx = [i for i, p in enumerate(row["params"]) if p.startswith("A[")]
            print(f"{name} blocks {blocks:2d}: pair loss {row['pair_loss']:+.3e} +- "
                  f"{row['pair_loss_se']:.1e}, tables' bias2 {np.mean(row['table_bias2']):.3e}"
                  f" | log_tau snr pair {row['pair_snr'][idx[0]]:.1f} bias "
                  f"{row['bias_snr'][idx[0]]:.1f} | log_pr snr pair {row['pair_snr'][idx[1]]:.1f}"
                  f" bias {row['bias_snr'][idx[1]]:.1f} | A median snr pair "
                  f"{np.median(np.array(row['pair_snr'])[a_idx]):.1f} bias "
                  f"{np.median(np.array(row['bias_snr'])[a_idx]):.1f} | rms agreement "
                  f"{np.sqrt(np.mean(np.square(row['agreement']))):.2f} sigma | "
                  f"{row['windows']} windows, {row['seconds']:.0f} s", flush=True)  # fmt: skip
            path.write_text(json.dumps({"args": vars(args), "rows": rows}))


if __name__ == "__main__":
    main()
