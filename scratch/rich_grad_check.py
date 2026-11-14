"""Per output row of the rich linear correction at zero (the start of training): the window
gradient of the trainer's bias estimator (train.CaseRunner, snapshot windows), its norm over
the features and its signal-to-noise -- finite everywhere, and which rows the windows push.

    PYTHONPATH=. python scratch/rich_grad_check.py K8 K1 S3 [--windows 16]
"""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from examples.learned_collision import operators, train
from jax_dsmc.collide import relax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--window-batch", type=int, default=4)
    args = ap.parse_args()
    targs = train.parser().parse_args(["--outputs", "rich", "--features", "features_v4.json",
                                       "--window-batch", str(args.window_batch)])  # fmt: skip
    theta = operators.operator("linear", "shock", jax.random.PRNGKey(0),
                               **train.model_options(targs))[1]  # fmt: skip
    rng = np.random.default_rng(0)
    for name in args.cases:
        t0 = time.time()
        r = train.CaseRunner(name, "linear", targs)
        _, devs, rows, _ = r.windows(theta, rng, "pair", args.windows)
        for row, d in zip(rows, devs, strict=True):
            r.update_bias(row, d)
        _, _, _, gs = r.windows(theta, rng, "bias", args.windows)
        g = np.stack([np.asarray(x["A"]) for x in gs])  # [W, rows, F]
        assert np.isfinite(g).all(), f"{name}: non-finite gradient"
        mean = g.mean(axis=0)
        se = g.std(axis=0, ddof=1) / np.sqrt(len(g))
        print(f"{name}: bias2 {r.bias_norm():.3e}  {time.time() - t0:.0f} s")
        for i, out in enumerate(relax.RICH_OUTPUTS):
            print(f"  {out:22s} |g| {np.linalg.norm(mean[i]):.3e}  snr "
                  f"{np.linalg.norm(mean[i]) / max(np.linalg.norm(se[i]), 1e-300):5.1f}")


if __name__ == "__main__":
    main()
