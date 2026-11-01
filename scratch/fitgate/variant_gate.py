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
    # the 5^D block (tricube support 3 cells)
    "pool2": ensemble_backward.CELL_BASIS + ":pool2",
}
# gates without an FD artifact: (config overrides of EM.default_config, parameter, FD steps),
# 2-D specular / diffuse pairs on identical bodies and boxes (production fit settings:
# one-sided backward, grazing floor 0.01; diffuse: T_w = T_inf and the silhouette band)
SQUASH = {
    "case": "ellipse",
    "fnum_scale": 1.0,
    "body": "levelset",
    "levelset_h": 0.0625,
    "levelset_samples": 16,
    "knudsen": 0.5,
    "a_max": 64,
    "squash": 0.6,
}
CIRCLE_M10 = {
    "case": "circle_long",
    "fnum_scale": 2.0,
    "warmup": 2800,
    "boundary": "free",
    "knudsen": 0.5,
    "mach": 10.0,
    "dt": 2.5e-5,
    "a_max": 256,
    "body": "levelset",
    "levelset_h": 0.0625,
    "levelset_samples": 16,
}
COMMON = {"backward_normals": True, "grazing": 0.01, "timing_band": 10.0}
DIFFUSE = {"acc": 1.0, "t_wall": 273.15, "silhouette_band": 0.03}
NEW_GATES = {
    # the squash gate that passed specular (-1.4 % vs FD, 2026-10-19): s = 0.6, M 0.32, Kn 0.5
    "sq_spec": ({**SQUASH, **COMMON}, "squash_y", [0.1, 0.05]),
    "sq_diff": ({**SQUASH, **COMMON, **DIFFUSE}, "squash_y", [0.1, 0.05]),
    # the L / D showcase's box: the R = 3 circle at M 10, Kn 0.5
    "c10_spec": ({**CIRCLE_M10, **COMMON}, "radius", [0.1, 0.05]),
    "c10_diff": ({**CIRCLE_M10, **COMMON, **DIFFUSE}, "radius", [0.1, 0.05]),
    # the diffuse M 10 circle with dt halved (the along-flight identity's O(nu dt) error; the
    # same physical warm-up) and with half the particles (the fits' kernel width)
    "c10_diff_dt2": ({**CIRCLE_M10, **COMMON, **DIFFUSE, "dt": 1.25e-5, "warmup": 5600},
                     "radius", [0.1, 0.05]),
    # the collisionality sweep of the diffuse M 10 circle: a collisional modelling error grows
    # as Kn falls and vanishes toward free-molecular flow
    "c10_diff_kn0.2": ({**CIRCLE_M10, **COMMON, **DIFFUSE, "knudsen": 0.2}, "radius", [0.1, 0.05]),
    # Kn 2 ran out of particle slots in the warm-up (inflow_shortfall ~2.4e5, 2026-10-31: the
    # default 1.3 x the free stream's ~1.5e5): twice the slots
    "c10_diff_kn2": ({**CIRCLE_M10, **COMMON, **DIFFUSE, "knudsen": 2.0, "slots": 400_000},
                     "radius", [0.1, 0.05]),
    "c10_diff_f2": ({**CIRCLE_M10, **COMMON, **DIFFUSE, "fnum_scale": 4.0}, "radius",
                    [0.1, 0.05]),
    # the specular M 10 circle with dt halved (2026-10-30: every variant +11-14 % on c10_spec,
    # the same with either collision field, where the Kn 10 Newton box passes at 0.1 %): an
    # O(nu dt) error of the collision pull-back halves, free of the diffuse wall's terms
    "c10_spec_dt2": ({**CIRCLE_M10, **COMMON, "dt": 1.25e-5, "warmup": 5600}, "radius",
                     [0.1, 0.05]),
    # the captured horizon (2026-11-01: dt / 2 left c10_spec's +12-18 % and every term but
    # pathwise_v unchanged, N / 2 c10_diff's): the same forward (warm-up + steps = 3100, the
    # window the last 200) with the backward over its last 1000 steps instead of 300 -- the
    # adjoint treats the state at the capture's start as fixed, so a shock layer slower than
    # the 300 steps (7.5 ms) is missed.  Paired with c10_spec / c10_diff seed by seed; their FD
    # rows are reused (FD_FROM)
    "c10_spec_h1000": ({**CIRCLE_M10, **COMMON, "warmup": 2100, "steps": 1000}, "radius",
                       [0.1, 0.05]),
    "c10_diff_h1000": ({**CIRCLE_M10, **COMMON, **DIFFUSE, "warmup": 2100, "steps": 1000},
                       "radius", [0.1, 0.05]),
    # collision cells half the size (2026-11-01: the user's hypothesis -- the M 10 bias lives
    # in the compressed shock layer, whose cells are too coarse for the dynamics there; the
    # horizon gave -3 %, the in-situ collision-volume gate -6 %): grid_scale 2 at the same
    # particles per cell (4x the particles, a whole card) and at the same particle count
    # (fnum x 4: a quarter the particles per cell)
    "c10_spec_g2": ({**CIRCLE_M10, **COMMON, "grid_scale": 2}, "radius", [0.1, 0.05]),
    "c10_spec_g2n": ({**CIRCLE_M10, **COMMON, "grid_scale": 2, "fnum_scale": 8.0}, "radius",
                     [0.1, 0.05]),
    # 2026-11-01: g2n halved every along / cell variant's bias (+11 -> +5 %, cellxv_along_pool
    # +9 -> +1 %), one_pool_plane unchanged: the scaling with the cell (g4n, a quarter the cell,
    # the same particles) and the diffuse circle on the half cells
    # a_max: the captured pair record is (steps, A_max, cells[, 3]) -- c10_spec_g4n's first try
    # kept CIRCLE_M10's 256 on 16x the cells and died allocating one 35.4 GB velocity array
    # (300 x 256 x 19 200 x 3 x 8 B, 2026-11-01).  NTC attempts per cell scale as N_c^2 / V_c,
    # i.e. 1 / k^D at the same particle count: <= 16 at k = 4 (c10_spec never outgrew 256),
    # ~125 for the diffuse circle at k = 2 (peak 499 at k = 1).  The cap is only the first
    # guess -- a clipped run is redone at 1.4x its peak, the same draws (_run_grown)
    "c10_spec_g4n": ({**CIRCLE_M10, **COMMON, "grid_scale": 4, "fnum_scale": 32.0, "a_max": 24},
                     "radius", [0.1, 0.05]),
    "c10_diff_g2n": ({**CIRCLE_M10, **COMMON, **DIFFUSE, "grid_scale": 2, "fnum_scale": 8.0,
                      "a_max": 160}, "radius", [0.1, 0.05]),
    # 2026-11-01: the specular bias is NOT cell size -- the adjoint stays ~0.56 at k = 1 / 2 / 4
    # (g2n's 1.05 was its FD 3 % high); the terms trade (pathwise_x + volume constant), the total
    # does not.  Is it collisional at all on THIS body and box?  (The Kn 10 Newton box passes,
    # but another body.)  slots: a free box at high Kn holds the back-reflected gas (Kn 2
    # diffuse ran out of slots); a_max: attempts scale with the density
    "c10_spec_kn10": ({**CIRCLE_M10, **COMMON, "knudsen": 10.0, "slots": 400_000, "a_max": 32},
                      "radius", [0.1, 0.05]),
    "c10_spec_kn2": ({**CIRCLE_M10, **COMMON, "knudsen": 2.0, "slots": 400_000, "a_max": 96},
                     "radius", [0.1, 0.05]),
    # the derivative-only settings, paired with c10_spec seed by seed (its forward, its FD): the
    # grazing cut-off 0.01 -> 0.001 (the hit derivative's floor), two-sided backward fits, and
    # 4 x 8 -> 8 x 16 virtual substeps x partners (the volume gate's -6 %)
    "c10_spec_gz001": ({**CIRCLE_M10, **COMMON, "grazing": 0.001}, "radius", [0.1, 0.05]),
    "c10_spec_two": ({**CIRCLE_M10, **COMMON, "backward_normals": False}, "radius",
                     [0.1, 0.05]),
    "c10_spec_v816": ({**CIRCLE_M10, **COMMON, "virtual": [8, 16]}, "radius", [0.1, 0.05]),
    # 2026-11-01: the cell-membership term (shape.membership_cotangent, a band of 0.1 cell
    # about each face), paired with c10_spec / c10_diff (their forwards and FD), and a second
    # dt point for the specular circle (2 dt, the same physical warm-up; the term on too)
    "c10_spec_mem": ({**CIRCLE_M10, **COMMON, "membership_band": 0.1}, "radius", [0.1, 0.05]),
    "c10_diff_mem": ({**CIRCLE_M10, **COMMON, **DIFFUSE, "membership_band": 0.1}, "radius",
                     [0.1, 0.05]),
    "c10_spec_2dt": ({**CIRCLE_M10, **COMMON, "dt": 5e-5, "warmup": 1400,
                      "membership_band": 0.1}, "radius", [0.1, 0.05]),
}  # fmt: skip
# gates whose forward is another gate's (the same objective, seeds and run): FD reused from
# that gate's rows in OUT_DIR when it has the seed, else computed
FD_FROM = {
    "c10_spec_h1000": "c10_spec",
    "c10_diff_h1000": "c10_diff",
    "c10_spec_gz001": "c10_spec",
    "c10_spec_two": "c10_spec",
    "c10_spec_v816": "c10_spec",
    "c10_spec_mem": "c10_spec",
    "c10_diff_mem": "c10_diff",
}
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
        return {**EM.default_config(), **over}, parameter, steps, 32, reused_fd(gate)
    data = json.loads(art.read_text())
    conf = data["config"]
    cfg = {**EM.default_config(), **{k: v for k, v in conf.items() if k in EM.default_config()}}
    rows = {r["seed"]: r for r in data["rows"]}
    return cfg, conf["parameter"], data["fd_steps"], conf["seed_start"], rows


def reused_fd(gate):
    """``{seed: {"fd": ..}}`` of :data:`FD_FROM`'s gate (its rows in OUT_DIR), ``{}`` else."""
    rows = {}
    if gate in FD_FROM:
        for path in sorted(OUT_DIR.glob(f"{FD_FROM[gate]}_*.jsonl")):
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if r["gate"] == FD_FROM[gate]:
                    rows[r["seed"]] = {"fd": r["fd"], "objective": r["objective"]}
    return rows


def terms_row(cfg, parameter, all_terms, s):
    pick = functools.partial(EM.control_gradient, cfg, parameter, s["grid"])
    names = ("pathwise_x", "pathwise_v", "pathwise_tally", "score", "timing", "silhouette")
    out = {}
    for name, t in all_terms.items():
        out[name] = {k: pick(getattr(t, k)) for k in names}
        out[name]["order"] = 0.0 if t.order is None else pick(t.order)
        if t.membership is not None:
            # reported, not in the analyses' totals until gated (2026-11-01)
            out[name]["membership"] = pick(t.membership)
    volume = 0.0 if s["volume"] is None else pick(s["volume"])
    return out, volume


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, required=True, help="first, end (exclusive)")
    ap.add_argument("--stride", type=int, default=1, help="every stride-th seed of the range")
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
    for seed in range(*args.seeds, args.stride):
        key = jax.random.key(seed_start + seed)
        fd = art_rows[seed]["fd"] if seed in art_rows else None
        for basis in args.bases:
            if (seed, basis) in done:
                continue
            sk = cfg.get("virtual", (4, 8))  # substeps x partners of the virtual pull-back
            ensemble_backward.configure("virtual", sk[0], sk[1], "staged", BASES[basis])
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
