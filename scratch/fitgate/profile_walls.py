"""Why is the diffuse shape gate ~55x slower than the specular one (2026-11-02: c10_diff_kn2
16.5 ks / seed against c10_spec_kn2's 300 s)?  A tiny version of each gate (fnum x SCALE, a
short warm-up and capture) runs the captured forward + backward + shape terms and the FD's
forwards for two seeds, counting the XLA compilations per stage and wall (jax_log_compiles):
a stage that compiles again on the second seed, or once per FD forward, is the culprit.

    PYTHONPATH=. python3 scratch/fitgate/profile_walls.py [--gates c10_spec_kn2 c10_diff_kn2]
"""

from __future__ import annotations

import argparse
import json
import collections
import logging
import sys
import time

import jax

sys.path.insert(0, "scratch/fitgate")
import variant_gate as V  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402

COMPILES = collections.Counter()
STAGE = ["setup"]


class CompileCounter(logging.Handler):
    """Counts jax_log_compiles' records per stage, keeping the compiled function's name."""

    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith("Compiling "):
            name = msg.split()[1]
            COMPILES[(STAGE[0], name)] += 1


def stage(name):
    STAGE[0] = name
    return time.time()


def report(gate, seed, timings):
    print(f"\n{gate} seed {seed}: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    for (st, name), n in sorted(COMPILES.items()):
        print(f"  {st:14s} {n:3d} x {name}")
    COMPILES.clear()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gates", nargs="+", default=["c10_spec_kn2", "c10_diff_kn2"])
    ap.add_argument("--scale", type=float, default=16.0, help="fnum x this (fewer particles)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--variants", nargs="+", default=["along_pool_plane"])
    args = ap.parse_args(argv)
    jax.config.update("jax_log_compiles", True)
    handler = CompileCounter()
    for logger in ("jax._src.dispatch", "jax._src.interpreters.pxla", "jax"):
        logging.getLogger(logger).addHandler(handler)
        logging.getLogger(logger).setLevel(logging.WARNING)
    for gate in args.gates:
        cfg, parameter, fd_steps, seed_start, _rows = V.gate_config(gate)
        # the 4 GB laptop GPU: slots and the attempt bound cut with the particle count (A_max
        # regrows if the flow needs more)
        cfg = {**cfg, "fnum_scale": args.scale * cfg["fnum_scale"], "warmup": 120, "steps": 40,
               "window": 20, "slots": int(cfg.get("slots", 400_000) / args.scale),
               "a_max": 8}  # fmt: skip
        runs = [(gate, cfg)]
        if cfg.get("silhouette_band", 0.0) > 0.0:
            # the diffuse wall's silhouette band term off: its share of the shape stage
            runs.append((f"{gate} silhouette off", {**cfg, "silhouette_band": 0.0}))
        for label, run_cfg in runs:
            profile(label, run_cfg, parameter, fd_steps, seed_start, args.seeds, args.variants)


def profile(label, cfg, parameter, fd_steps, seed_start, seeds, variants):
    """Two (or ``seeds``) seeds of the shape pipeline and the FD's forwards, compiles counted."""
    value = float(EM.nominal(cfg, parameter))
    for seed in range(seeds):
        key = jax.random.key(seed_start + seed)
        timings = {}
        t0 = stage("shape_terms")
        all_terms, info, s = EM.shape_terms(
            cfg, key, None, parameter, 32, cfg.get("timing_band", 10.0), tuple(variants)
        )
        timings["shape_terms"] = time.time() - t0
        terms, volume = V.terms_row(cfg, parameter, all_terms, s)
        for name, t in terms.items():
            print(f"  {label} seed {seed} {name}: " + json.dumps(t), flush=True)
        print(f"  {label} seed {seed} volume {volume!r}", flush=True)
        timings |= {k: v for k, v in info.items() if k.startswith("seconds")}
        for sign in (1.0, -1.0):
            t0 = stage(f"fd_{'+' if sign > 0 else '-'}")
            EM.forward(cfg, parameter, value + sign * fd_steps[0], key, False)
            timings[f"fd_{'+' if sign > 0 else '-'}"] = time.time() - t0
        report(label, seed, timings)


if __name__ == "__main__":
    main()
