"""The wall / volume split of a shape FD (2026-11-28): where is c10_spec's +11 % radius excess?

A shape change reaches the forward through two channels: the wall (hits, reflections, tallies)
and the collision cells' flow volumes ``grid.volume`` (every NTC probability scales as 1 / V_c).
The adjoint splits the same way -- its ``volume`` term is the pull-back through ``grid.volume``,
every other term the wall's -- so with common random numbers

    FD_wall = [J(p + h, V(p)) - J(p - h, V(p))] / 2h     the body moved, the volumes frozen
    FD_vol  = [J(p, V(p + h)) - J(p, V(p - h))] / 2h     the body fixed, the volumes moved
    FD_tot  = [J(p + h, V(p + h)) - J(p - h, V(p - h))] / 2h   (the gate's own FD; additivity)

against the adjoint's (total - volume) and volume.  The old volume gate (volume_gate.py) scaled
whole sets of cells uniformly -- the ``cut`` set alone was FD noise -- while a radius change
moves volume almost only in the cut cells, with the geometry's own pattern: this is that
pattern's test.  ``--parameter`` defaults to the gate's (radius for c10_spec; mode2 for
c10_spec_m2, which closes: the control).  Rows to ``splitfd_<gate>_s<a>-<b>.jsonl``.

    PYTHONPATH=. python3 scratch/fitgate/split_fd.py --gate c10_spec --seeds 0 8
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "scratch/fitgate")
import variant_gate as V  # noqa: E402
import volume_gate as VG  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402

jax.config.update("jax_enable_x64", True)
_BUILD = EM.build
SWAP = {}


def build_swapped(cfg, parameter, value, steps, body=None):
    """``EM.build`` with ``grid.volume`` replaced by ``SWAP["volume"]`` when set."""
    out = _BUILD(cfg, parameter, value, steps, body)
    if "volume" in SWAP:
        grid = out[0]
        if SWAP["volume"].shape != grid.volume.shape:
            raise ValueError("collision-cell layout changed with the control")
        out = (dataclasses.replace(grid, volume=SWAP["volume"]), *out[1:])
    return out


def volume_at(cfg, parameter, value):
    return jnp.asarray(_BUILD(cfg, parameter, value, 1)[0].volume)


def objective_with(cfg, parameter, value, key, volume):
    SWAP["volume"] = volume
    try:
        return VG.objective(cfg, parameter, value, key)
    finally:
        SWAP.clear()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, required=True, help="first, end (exclusive)")
    ap.add_argument("--h", type=float, default=None, help="FD step (default: the gate's first)")
    args = ap.parse_args(argv)
    cfg, parameter, fd_steps, seed_start, _rows = V.gate_config(args.gate)
    h = args.h or fd_steps[0]
    p0 = float(EM.nominal(cfg, parameter))
    EM.build = build_swapped
    vols = {s: volume_at(cfg, parameter, p0 + s * h) for s in (-1.0, 0.0, 1.0)}
    dv = np.asarray(vols[1.0] - vols[-1.0])
    print(f"{args.gate} {parameter} h {h}: {int(np.sum(dv != 0))} cells change volume, "
          f"sum dV/dp {dv.sum() / (2 * h):+.4g}", flush=True)  # fmt: skip
    out = VG.OUT_DIR / f"splitfd_{args.gate}_s{args.seeds[0]}-{args.seeds[1]}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["seed"] for line in out.read_text().splitlines()}
    for seed in range(*args.seeds):
        if seed in done:
            continue
        key = jax.random.key(seed_start + seed)
        t0 = time.time()
        j = {
            "wall+": objective_with(cfg, parameter, p0 + h, key, vols[0.0]),
            "wall-": objective_with(cfg, parameter, p0 - h, key, vols[0.0]),
            "vol+": objective_with(cfg, parameter, p0, key, vols[1.0]),
            "vol-": objective_with(cfg, parameter, p0, key, vols[-1.0]),
            "tot+": objective_with(cfg, parameter, p0 + h, key, vols[1.0]),
            "tot-": objective_with(cfg, parameter, p0 - h, key, vols[-1.0]),
        }
        row = {"gate": args.gate, "parameter": parameter, "seed": seed, "h": h,
               "fd_wall": (j["wall+"] - j["wall-"]) / (2 * h),
               "fd_vol": (j["vol+"] - j["vol-"]) / (2 * h),
               "fd_tot": (j["tot+"] - j["tot-"]) / (2 * h),
               "objectives": j, "seconds": time.time() - t0}  # fmt: skip
        print(json.dumps(row), flush=True)
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
