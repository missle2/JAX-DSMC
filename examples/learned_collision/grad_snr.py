"""Gradient signal-to-noise of the window loss (plan step 7): per-window gradients of ES-FP's
``(log_tau, log_pr)`` over many random windows, for each particles-per-cell and window
length -- the mean, its standard error and ``|mean| / se`` per reference.

    PYTHONPATH=. python3 -m examples.learned_collision.grad_snr [--ppc 10 50] [--blocks 2 5]
        [--windows 64] [--dx 4]

Writes ``examples/learned_collision/results/grad_snr_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from examples.learned_collision import train_scalar as ts
from examples.learned_collision import windows
from examples.learned_collision.shear_wave import RESULTS


def study(refs, dx, ppc, blocks, burn, n_windows, operator_name, seed):
    """Per-reference gradient statistics over ``n_windows`` windows."""
    args = argparse.Namespace(dx=dx, ppc=ppc, blocks=blocks, burn=burn)
    collide_fn, theta, _ = ts.operator(operator_name)
    out = {}
    for name, (ref, meta) in refs.items():
        case = windows.coarse_case(meta["wave"], meta["length_mfp"], dx, ppc,
                                   ref["snap_x"].shape[2])  # fmt: skip
        fns = ts.compiled(case, collide_fn, args, ts.FIELD[meta["wave"]])
        rng = np.random.default_rng(seed)
        losses, grads = [], []
        for _ in range(n_windows):
            loss, g = ts.window(ref, meta, case, rng, fns, theta, args)
            losses.append(float(loss))
            grads.append([float(g["log_tau"]), float(g["log_pr"])])
        grads = np.array(grads)
        mean, se = grads.mean(0), grads.std(0, ddof=1) / np.sqrt(n_windows)
        out[name] = {"loss_mean": float(np.mean(losses)),
                     "loss_se": float(np.std(losses, ddof=1) / np.sqrt(n_windows)),
                     "grad_mean": mean.tolist(), "grad_se": se.tolist(),
                     "snr": (np.abs(mean) / se).tolist(),
                     "finite": bool(np.all(np.isfinite(grads)))}  # fmt: skip
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--refs", nargs="+", default=["reference_shear_L64_v1.npz", "reference_thermal_L64_v1.npz"]
    )
    ap.add_argument("--dx", type=float, default=4.0)
    ap.add_argument("--ppc", type=int, nargs="+", default=[10, 50])
    ap.add_argument("--blocks", type=int, nargs="+", default=[2, 5])
    ap.add_argument("--burn", type=int, default=2)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--operator", default="esfp", choices=("esfp", "esfp_recon"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)  # fmt: skip
    refs = {name: ts.load(name) for name in args.refs}
    rows = []
    for ppc in args.ppc:
        for blocks in args.blocks:
            t0 = time.time()
            res = study(refs, args.dx, ppc, blocks, args.burn, args.windows, args.operator,
                        args.seed)  # fmt: skip
            rows.append({"ppc": ppc, "blocks": blocks, "dx": args.dx, "results": res,
                         "seconds": time.time() - t0})  # fmt: skip
            for name, r in res.items():
                print(f"ppc {ppc:3d} blocks {blocks}  {name[:22]:22s} loss {r['loss_mean']:9.3f}"
                      f" +- {r['loss_se']:7.3f}  dL/dlog_tau {r['grad_mean'][0]:9.3f} +-"
                      f" {r['grad_se'][0]:7.3f}  dL/dlog_pr {r['grad_mean'][1]:9.3f} +-"
                      f" {r['grad_se'][1]:7.3f}  finite {r['finite']}", flush=True)  # fmt: skip
            (RESULTS / f"grad_snr_{args.tag}.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
