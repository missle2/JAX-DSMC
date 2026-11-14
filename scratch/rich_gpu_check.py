"""GPU validation of the rich correction (2026-11-14): steady coarse shocks run to a sampled
steady state with every rich output pushed to its bounds, per-cell varied (random linear
weights on the eleven standardised features) or pinned one way -- finite, positive
temperature, shock held, stress and heat flux bounded -- and a zero rich model against the
rate-only zero model (the same run, bitwise metrics).

    PYTHONPATH=. python scratch/rich_gpu_check.py K8 K1 [--starts 8]
"""

from __future__ import annotations

import argparse
import json
import time

import jax.numpy as jnp
import numpy as np

from examples.learned_collision import evaluate as ev
from examples.learned_collision import flows
from examples.learned_collision.shear_wave import RESULTS
from jax_dsmc.collide import relax


def thetas(phi_scale):
    k = len(phi_scale)
    base = relax.init_correction("linear", phi_scale=phi_scale, rich=True)
    rates = relax.init_correction("linear", phi_scale=phi_scale)
    out = {"rates_zero": rates, "rich_zero": base}
    for seed in range(3):
        a = 5.0 * np.random.default_rng(seed).standard_normal((len(relax.RICH_OUTPUTS), k))
        out[f"adv{seed}"] = dict(base, A=jnp.asarray(a))
    even = relax.FEATURES.index("div_u")  # nonnegative features: a fixed sign in every cell
    for name, sign in (("pin_plus", 1.0), ("pin_minus", -1.0)):
        a = np.zeros((len(relax.RICH_OUTPUTS), k))
        a[:, :even] = 20.0 * sign
        out[name] = dict(base, A=jnp.asarray(a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--relax", type=float, default=0.5)
    ap.add_argument("--sample", type=float, default=0.5)
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    phi_scale = json.loads((RESULTS / "features_v4.json").read_text())["phi_scale"]
    models = thetas(phi_scale)
    if args.only:
        models = {k: v for k, v in models.items() if k in args.only}
    for name in args.cases:
        ref, meta = ev.load_ref(flows.CASES[name][0], "v1")
        for label, theta in models.items():
            t0 = time.time()
            r = ev.eval_shock(name, "linear", theta, ref, meta, args.starts, args.relax,
                              args.sample, seed=0)  # fmt: skip
            prof = np.array(r["profile"])  # [C, (n, T, txx, qx)]
            ok = np.isfinite(prof).all() and prof[:, 1].min() > 0.0 and prof[:, 0].min() > 0.0
            print(f"{name} {label:10s} {'OK  ' if ok else 'FAIL'} int txx "
                  f"{r['int_txx_ratio']:.3f} qx {r['int_qx_ratio']:.3f}  l2 n {r['l2_n']:.3f} "
                  f"T {r['l2_T']:.3f} txx {r['l2_txx']:.3f} qx {r['l2_qx']:.3f}  drift "
                  f"{r['drift']:+.2f}  Tmin {prof[:, 1].min():.3g} clipped "
                  f"{r['attempts_clipped']} short {r['inflow_shortfall']:.2g}  "
                  f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
