"""The ``lcsnr`` gate (plan 2026-11-11): signal-to-noise of the two window-gradient
estimators (:mod:`.windows2d`) on the round-1 cases, against window length.

Per case and window length: ``--bias-windows`` windows fill a running-bias table (forward
only), then ``--windows`` fresh windows each give both estimators' gradients *on the same
window* (same pools, stripes, thinning and noise) -- the two-replica ``pair`` and the
running-bias ``bias`` (the table from the first, independent set, then frozen: no lag, no
reuse bias).
Reported per parameter: the mean gradient, its standard error and ``|mean| / se``, at the
base operator (``theta`` from the model's zero initialisation).

    PYTHONPATH=. python3 -m examples.learned_collision.grad_snr2d [--cases S2 K1]
        [--blocks-list 2 5 10] [--windows 64] [--model linear]

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
    return {k: np.asarray(v, dtype=float).ravel() for k, v in grad.items()}


def study(name, blocks, args):
    """Both estimators' gradient statistics on case ``name`` at window length ``blocks``."""
    args.blocks = blocks
    runner = CaseRunner(name, args.model, args)
    _, theta, _ = operators.operator(args.model, runner.case.kind, jax.random.PRNGKey(0))
    rng = np.random.default_rng(args.seed)
    for _ in range(args.bias_windows):
        _, d_mean, rows, _ = runner.window(theta, rng, "pair")
        runner.update_bias(rows, d_mean)
    grads = {"pair": [], "bias": []}
    pairs = []
    for i in range(args.windows):
        for est in ("pair", "bias"):
            pair, _, _, g = runner.window(theta, np.random.default_rng((args.seed, 7, i)), est)
            grads[est].append(_flat(g))
        pairs.append(pair)
    out = {"case": name, "blocks": blocks, "pair_loss": float(np.mean(pairs)),
           "pair_loss_se": float(np.std(pairs, ddof=1) / np.sqrt(len(pairs))),
           "bias2": runner.bias_norm()}  # fmt: skip
    for est, rows in grads.items():
        for leaf in rows[0]:
            g = np.stack([r[leaf] for r in rows])
            mean, se = g.mean(0), g.std(0, ddof=1) / np.sqrt(len(g))
            out[f"{est}_{leaf}_mean"] = mean.tolist()
            out[f"{est}_{leaf}_se"] = se.tolist()
            out[f"{est}_{leaf}_snr"] = (np.abs(mean) / np.maximum(se, 1e-300)).tolist()
    return out


def main(argv=None):
    ap = parser()
    ap.description = __doc__.split("\n\n")[0]
    ap.set_defaults(cases=["S2", "K1"], model="linear")
    ap.add_argument("--blocks-list", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--bias-windows", type=int, default=64)
    args = ap.parse_args(argv)
    rows, t0 = [], time.time()
    path = RESULTS / f"grad_snr2d_{args.tag}.json"
    for name in args.cases:
        for blocks in args.blocks_list:
            row = study(name, blocks, args)
            row["seconds"] = time.time() - t0
            rows.append(row)
            print(f"{name} blocks {blocks:2d}  pair loss {row['pair_loss']:.3e} +- "
                  f"{row['pair_loss_se']:.1e}  bias2 {row['bias2']:.3e}  log_tau snr pair "
                  f"{row['pair_log_tau_snr'][0]:.1f} bias {row['bias_log_tau_snr'][0]:.1f}  "
                  f"log_pr snr pair {row['pair_log_pr_snr'][0]:.1f} bias "
                  f"{row['bias_log_pr_snr'][0]:.1f}  {row['seconds']:.0f} s",
                  flush=True)  # fmt: skip
            path.write_text(json.dumps({"args": vars(args), "rows": rows}))


if __name__ == "__main__":
    main()
