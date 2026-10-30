"""Variant gate of the shape adjoint's fits (2026-10-29): every grad-lambda estimator of
``jax_dsmc.shape.VARIANTS`` (the per-query k-nearest fits and the per-cell polynomial ones)
under one or more collision-field bases, on the same realisations, against central
differences -- to choose the fits by bias AND per-seed spread.

One forward per seed is shared by every collision-field basis (``EM.forward`` cached for the
nominal captured run); each basis runs its own backward and the shape VJP of all variants.
Per (seed, basis) one JSON line: the objective, every variant's own terms (``pathwise_x``,
``pathwise_v``, ``pathwise_tally``, ``score``, ``timing``, ``silhouette``, ``order``), the
basis' cut-volume term and the central differences -- copied from the FD gate's artifact
(examples/paper/results/fd/<gate>.json, same config and seeds) when it has the seed, else
computed here.  A rerun skips the (seed, basis) pairs already written.

    PYTHONPATH=. python3 scratch/fitgate/variant_gate.py --gate af_drag_radius --seeds 0 4 \\
        --bases base pool1
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import jax

from examples import ensemble_mesh as EM
from jax_dsmc import ensemble_backward, shape

jax.config.update("jax_enable_x64", True)
FD_DIR = Path("examples/paper/results/fd")
OUT_DIR = Path("examples/mesh/results/fitgate")
BASES = {
    "base": ensemble_backward.CELL_BASIS,
    # the collision field pooled over the 3^D block (tricube-weighted around each cell)
    "pool1": ensemble_backward.CELL_BASIS + ":pool1",
}
# gates without an FD artifact: (config overrides of EM.default_config, parameter, FD steps)
NEW_GATES = {
    # the Newton box at Kn 0.5, M 10 with SPECULAR walls (the paper's nw_drag_radius otherwise)
    "nw_spec_radius": (
        {"case": "newton", "levelset_h": 0.1, "fnum_scale": 1.0, "warmup": 1000, "mach": 10.0,
         "dt": 2.5e-5, "boundary": "free", "a_max": 16, "acc": 0.0, "silhouette_band": 0.0,
         "knudsen": 0.5, "body": "levelset", "backward_normals": True, "grazing": 0.01,
         "timing_band": 10.0},
        "radius",
        [0.05, 0.02],
    ),
}  # fmt: skip
_CACHE = {}


def cached_forward(original, cfg, parameter, value, key, capture, *args, **kwargs):
    """``EM.forward`` with the nominal captured run of the current seed kept for the next
    basis (the FD runs, capture off, are not cached)."""
    tag = (repr(key), float(value), bool(capture), args, tuple(sorted(kwargs.items())))
    if capture and tag in _CACHE:
        return _CACHE[tag]
    out = original(cfg, parameter, value, key, capture, *args, **kwargs)
    if capture:
        _CACHE.clear()
        _CACHE[tag] = out
    return out


def gate_config(gate):
    """``(cfg, parameter, fd_steps, seed_start, artifact rows by seed)``."""
    art = FD_DIR / f"{gate}.json"
    if gate in NEW_GATES:
        over, parameter, steps = NEW_GATES[gate]
        return {**EM.default_config(), **over}, parameter, steps, 32, {}
    data = json.loads(art.read_text())
    conf = data["config"]
    cfg = {**EM.default_config(), **{k: v for k, v in conf.items() if k in EM.default_config()}}
    rows = {r["seed"]: r for r in data["rows"]}
    return cfg, conf["parameter"], data["fd_steps"], conf["seed_start"], rows


def terms_row(cfg, parameter, all_terms, s):
    pick = functools.partial(EM.control_gradient, cfg, parameter, s["grid"])
    names = ("pathwise_x", "pathwise_v", "pathwise_tally", "score", "timing", "silhouette")
    out = {}
    for name, t in all_terms.items():
        out[name] = {k: pick(getattr(t, k)) for k in names}
        out[name]["order"] = 0.0 if t.order is None else pick(t.order)
    volume = 0.0 if s["volume"] is None else pick(s["volume"])
    return out, volume


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, required=True, help="first, end (exclusive)")
    ap.add_argument("--bases", nargs="+", default=["base"], choices=list(BASES))
    ap.add_argument("--variants", nargs="*", default=list(shape.VARIANTS))
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="the plumbing only: fnum x 16, 150 + 40 steps, one FD step, *_smoke",
    )
    args = ap.parse_args(argv)
    cfg, parameter, fd_steps, seed_start, art_rows = gate_config(args.gate)
    name = args.gate
    if args.smoke:
        cfg = {**cfg, "fnum_scale": 16.0 * cfg["fnum_scale"], "warmup": 150, "steps": 40,
               "window": 20}  # fmt: skip
        fd_steps, art_rows, name = fd_steps[:1], {}, f"{args.gate}_smoke"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}_{'-'.join(args.bases)}_s{args.seeds[0]}-{args.seeds[1]}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["seed"], r["basis"]))
    EM.forward = functools.partial(cached_forward, EM.forward)
    for seed in range(*args.seeds):
        key = jax.random.key(seed_start + seed)
        fd = art_rows[seed]["fd"] if seed in art_rows else None
        for basis in args.bases:
            if (seed, basis) in done:
                continue
            ensemble_backward.configure("virtual", 4, 8, "staged", BASES[basis])
            t0 = time.time()
            all_terms, info, s = EM.shape_terms(
                cfg, key, None, parameter, 32, cfg.get("timing_band", 10.0),
                tuple(args.variants),
            )  # fmt: skip
            terms, volume = terms_row(cfg, parameter, all_terms, s)
            if fd is None:
                fd = EM.finite_differences(cfg, parameter, key, fd_steps)
            ref = art_rows.get(seed, {})
            row = {
                "gate": args.gate, "seed": seed, "basis": basis, "basis_string": BASES[basis],
                "objective": info["objective"], "objective_artifact": ref.get("objective"),
                "fd": fd, "fd_steps": fd_steps, "volume": volume, "terms": terms,
                "hits": info["hits"], "lambda_absmax": info["lambda_absmax"],
                "seconds_backward": info["seconds_backward"],
                "seconds_shape": info["seconds_shape"], "seconds": time.time() - t0,
            }  # fmt: skip
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            prod = terms.get("along_pool_plane")
            msg = "" if prod is None else f" along_pool_plane x {prod['pathwise_x']:.4g}"
            print(f"{args.gate} seed {seed} {basis}: J {info['objective']:.6g} "
                  f"(artifact {ref.get('objective')}) FD {fd}{msg} "
                  f"{row['seconds']:.0f}s", flush=True)  # fmt: skip
        _CACHE.clear()


if __name__ == "__main__":
    main()
