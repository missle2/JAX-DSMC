"""Does the window objective point where the steady shocks need to go?  (``lcsteadyfd``,
2026-11-12.)

Training lowered the shock stress errors but raised the heat-flux ones (lctime, lcburn: int qx
0.97 -> 0.87-0.91 on K4, 0.94 -> 0.82-0.84 on K8), with any burn-in from 2 to 30 blocks.  At
the base operator the window gradient (lcsnr v2/v3) asks for a higher ``log_pr``, a faster
heat-flux relaxation and so a lower conductivity, while the steady profiles' heat flux is
already too low.  This measures the steady side directly: the forward shock metrics
(:func:`.evaluate.eval_shock`) of the scalar operator at the base coefficients and at
``log_tau`` / ``log_pr`` +- ``--h``, common random numbers (the same starts and keys), and
the central differences of each metric and of the steady squared error
``l2_txx^2 + l2_qx^2``.  Its sign against the window gradient's settles whether the window
objective or its weighting is at fault.

    PYTHONPATH=. python3 -m examples.learned_collision.steady_fd [--cases K1 K4 K8] [--h 0.1]

Writes ``examples/learned_collision/results/steady_fd_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import time

import jax.numpy as jnp

from examples.learned_collision import evaluate as ev
from examples.learned_collision import flows
from examples.learned_collision.shear_wave import RESULTS
from jax_dsmc.collide import relax

METRICS = ("l2_txx", "l2_qx", "int_txx_ratio", "int_qx_ratio")


def _theta(param=None, step=0.0):
    theta = {k: jnp.float64(v) for k, v in relax.BASELINE.items()}
    if param is not None:
        theta[param] = theta[param] + step
    return theta


def _loss(res):
    return res["l2_txx"] ** 2 + res["l2_qx"] ** 2


def study(name, args):
    """The base and the four perturbed runs of case ``name`` and their differences."""
    ref, meta = ev.load_ref(flows.CASES[name][0], args.refs_tag)
    runs = {"base": (None, 0.0)}
    for p in ("log_tau", "log_pr"):
        runs[f"{p}+"] = (p, args.h)
        runs[f"{p}-"] = (p, -args.h)
    out = {}
    for label, (param, step) in runs.items():
        t0 = time.time()
        res = ev.eval_shock(name, "scalar", _theta(param, step), ref, meta, args.starts, 0.5,
                            1.0, seed=args.seed, ppc=args.ppc)  # fmt: skip
        out[label] = {k: v for k, v in res.items() if k not in ("profile", "reference")}
        out[label]["loss"] = _loss(res)
        print(f"{name} {label:9s} " + " ".join(f"{k} {res[k]:.4f}" for k in METRICS)
              + f"  loss {out[label]['loss']:.3e}  {time.time() - t0:.0f} s", flush=True)  # fmt: skip
    fd = {}
    for p in ("log_tau", "log_pr"):
        fd[p] = {k: (out[f"{p}+"][k] - out[f"{p}-"][k]) / (2.0 * args.h)
                 for k in (*METRICS, "loss")}  # fmt: skip
        print(f"{name} d/d{p}: " + " ".join(f"{k} {v:+.4f}" for k, v in fd[p].items()), flush=True)
    return {"case": name, "runs": out, "fd": fd}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", nargs="+", default=["K1", "K4", "K8"])
    ap.add_argument("--h", type=float, default=0.1)
    ap.add_argument("--starts", type=int, default=32)
    ap.add_argument("--ppc", type=int, default=flows.PPC_COARSE)
    ap.add_argument("--refs-tag", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)
    rows = []
    for name in args.cases:
        rows.append(study(name, args))
        (RESULTS / f"steady_fd_{args.tag}.json").write_text(
            json.dumps({"args": vars(args), "rows": rows})
        )


if __name__ == "__main__":
    main()
