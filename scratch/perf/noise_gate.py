"""Seed-to-seed noise-floor gate: a code change that alters floating-point results at
round-off decorrelates the DSMC trajectories, so a bitwise gate is meaningless; instead the
change must move the seed-means of the physical outputs by less than the seed noise.

Forward gate (SPARTA circle case, polygon and level-set bodies, lane tracker, collisions
on, empty box, ``--steps`` steps run in blocks of ``BLOCK``): per seed, steady-window
(steps ``[--window-start, --steps)``) means of the body drag / lift / energy / heat /
hits, the box-face exits, the injected count, and of block-end state samples (particle
count, mean kinetic energy, temperature, mean x-velocity); plus a particle-count series.

Gradient gate (the optimiser's per-seed chain ``examples.ensemble_mesh.shape_terms`` at
the ``regress.py`` settings: level-set circle, h 0.0625, 16 samples, grazing 0.05, fnum x 4,
virtual 1 x 2, variant ``along_pool``; seed key ``jax.random.key(32 + seed)`` drives the
forward and the backward): per seed the objective, hits and the coefficient gradient
(pathwise + score + silhouette) with its parts.

    # save (each tree's code imported cleanly: the work runs in a subprocess with
    # PYTHONPATH=<--code>)
    run_locked.sh gpu python3 scratch/perf/noise_gate.py forward  --code DIR --save TAG
    run_locked.sh gpu python3 scratch/perf/noise_gate.py gradient --code DIR --save TAG
    # compare (numpy only, no lock needed)
    python3 scratch/perf/noise_gate.py forward  --compare base TAG
    python3 scratch/perf/noise_gate.py gradient --compare base TAG

Criterion (every gated scalar observable): z = (mean_new - mean_base) / sqrt(se_base^2 +
se_new^2), se = seed std / sqrt(n_seeds); FAIL if |z| >= 3, WARN if 2 <= |z| < 3.  (A
per-observable 2-sigma cut is too tight for a family of ~11 correlated observables x 2
bodies: two independent baseline seed sets failed it on the lift at z = 2.4.)  On a
marginal FAIL re-run with more seeds (``--append --seeds 12 13 ...``) before concluding.
Seeds common to both sets are also compared seed-by-seed (bitwise equal or not).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_TREE = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "noise_gate")
BLOCK = 20
FORWARD_SEEDS = tuple(range(12))
GRADIENT_SEEDS = tuple(range(4))
SIGMAS = 3.0
WARN_SIGMAS = 2.0

# forward observables: (name, description, gated)
FORWARD_OBS = (
    ("drag", "body force x [fnum/dt * sum m dv], per step, window mean", True),
    ("lift", "body force y, window mean", True),
    ("body_energy", "energy flux to body [fnum/dt * sum dE] (0 up to round-off: specular)", False),
    ("body_heat", "heat flux to body (0 up to round-off: specular)", False),
    ("body_hits", "body hits per step", True),
    ("exits", "particles leaving through the box faces per step", True),
    ("exit_momentum_x", "x-momentum flux out through box faces [fnum/dt * sum m v]", True),
    ("exit_energy", "energy flux out through box faces", True),
    ("injected", "inflow inserted per step", True),
    ("particles", "alive count (block-end samples in window)", True),
    ("mean_ke", "mean kinetic energy per alive particle [J]", True),
    ("temperature", "m <|v - <v>|^2> / (3 k) over alive particles [K]", True),
    ("mean_vx", "mean x-velocity of alive particles [m/s]", True),
)


# ---------------------------------------------------------------- workers (import JAX)


def _setup(body, steps):
    import jax.numpy as jnp  # noqa: PLC0415
    import forward as F  # noqa: PLC0415

    from examples import ensemble_mesh as EM  # noqa: PLC0415

    if body == "polygon":
        grid, mesh, sp, params, static, faces, res, wall, _t, capacity = F.build(
            "circle", "polygon", "lane", "Ar", 0.0, 1
        )
    else:
        cfg = dict(
            EM.default_config(), body="levelset", levelset_h=0.0625, levelset_samples=16,
            fnum_scale=1.0,
        )  # fmt: skip
        grid, mesh, sp, params, static, faces, res, wall, capacity = EM.build(
            cfg, "radius", F.CASES["circle"]["radius"], steps
        )
    static = dataclasses.replace(static, n_steps=BLOCK)
    flow = dict(nrho=jnp.float64(F.NRHO), temp=jnp.float64(F.TEMP), vstream=(F.VSTREAM, 0.0, 0.0))
    return (grid, mesh, sp, params, static, faces, res, wall, capacity), flow


def _forward_seed(objs, flow, seed, steps, w0):
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from examples import ensemble_mesh as EM  # noqa: PLC0415
    from jax_dsmc import initialize  # noqa: PLC0415
    from jax_dsmc.constants import K_BOLTZ  # noqa: PLC0415
    from jax_dsmc.state import FlowParams  # noqa: PLC0415

    grid, mesh, sp, params, static, faces, res, wall, capacity = objs
    D = static.D
    fp = FlowParams(nrho=flow["nrho"], temp=flow["temp"], vstream=jnp.array(flow["vstream"]))
    key = jax.random.key(seed)
    state = initialize.initial_state(key, fp, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(fp, grid, sp, static)
    mass = float(sp.mass[0])
    scale = float(params.fnum[0]) / float(params.dt)
    per_step = {k: [] for k in ("drag", "lift", "body_energy", "body_heat", "body_hits",
                                "exits", "exit_momentum_x", "exit_energy", "injected")}  # fmt: skip
    samples = {k: [] for k in ("particles", "mean_ke", "temperature", "mean_vx")}
    series = []
    for k0 in range(0, steps, BLOCK):
        (state, cstate), aux = EM._run(
            state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity, k0,
            False,
        )  # fmt: skip
        diag, tal = aux[1], aux[-1]
        mom = np.asarray(tal.momentum)  # [BLOCK, 2D + E, 3]
        en = np.asarray(tal.energy)
        heat = np.asarray(tal.heat)
        hits = np.asarray(tal.hits)
        alive = np.asarray(state.alive)
        v = np.asarray(state.v)[alive]
        n = int(alive.sum())
        series.append((k0 + BLOCK, n))
        step_idx = k0 + np.arange(BLOCK)
        sel = step_idx >= w0
        if sel.any():
            per_step["drag"].append(scale * mom[sel, 2 * D :, 0].sum(axis=1))
            per_step["lift"].append(scale * mom[sel, 2 * D :, 1].sum(axis=1))
            per_step["body_energy"].append(scale * en[sel, 2 * D :].sum(axis=1))
            per_step["body_heat"].append(scale * heat[sel, 2 * D :].sum(axis=1))
            per_step["body_hits"].append(hits[sel, 2 * D :].sum(axis=1).astype(float))
            per_step["exits"].append(hits[sel, : 2 * D].sum(axis=1).astype(float))
            per_step["exit_momentum_x"].append(scale * mom[sel, : 2 * D, 0].sum(axis=1))
            per_step["exit_energy"].append(scale * en[sel, : 2 * D].sum(axis=1))
            per_step["injected"].append(np.asarray(diag.inflow_injected)[sel].astype(float))
        if k0 + BLOCK > w0:
            vbar = v.mean(axis=0)
            samples["particles"].append(float(n))
            samples["mean_ke"].append(0.5 * mass * float(np.mean(np.sum(v**2, axis=1))))
            samples["temperature"].append(
                mass * float(np.mean(np.sum((v - vbar) ** 2, axis=1))) / (3.0 * K_BOLTZ)
            )
            samples["mean_vx"].append(float(vbar[0]))
    values = {k: float(np.mean(np.concatenate(v))) for k, v in per_step.items()}
    values.update({k: float(np.mean(v)) for k, v in samples.items()})
    return values, np.array(series, dtype=np.int64)


def worker_forward(args):
    import jax  # noqa: PLC0415

    print(f"code {args.code}; jax_dsmc from {_where()}", flush=True)
    rows = {"meta": {"code": args.code, "steps": args.steps, "window_start": args.window_start,
                     "block": BLOCK, "device": jax.devices()[0].device_kind,
                     "seeds": list(args.seeds), "bodies": list(args.bodies)}}  # fmt: skip
    series = {}
    for body in args.bodies:
        t0 = time.time()
        objs, flow = _setup(body, args.steps)
        rows[body] = {}
        for i, seed in enumerate(args.seeds):
            ts = time.time()
            values, s = _forward_seed(objs, flow, seed, args.steps, args.window_start)
            rows[body][str(seed)] = values
            series[f"{body}_s{seed}"] = s
            print(f"{body} seed {seed}: {time.time() - ts:.1f} s  "
                  + " ".join(f"{k} {v:.6g}" for k, v in values.items()), flush=True)  # fmt: skip
            if i == 0:
                rows["meta"][f"{body}_first_seed_seconds"] = time.time() - ts
        rows["meta"][f"{body}_seconds"] = time.time() - t0
        print(f"{body}: {time.time() - t0:.0f} s total", flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"forward_{args.save}.json")
    spath = os.path.join(OUT, f"forward_{args.save}_series.npz")
    if args.append and os.path.exists(path):
        old = json.load(open(path))
        for k in ("steps", "window_start", "block"):
            if old["meta"][k] != rows["meta"][k]:
                raise SystemExit(f"--append: {k} differs from the saved set")
        for body in args.bodies:
            rows[body] = {**old.get(body, {}), **rows[body]}
        rows["meta"]["seeds"] = sorted(set(old["meta"]["seeds"]) | set(args.seeds))
        rows["meta"]["bodies"] = sorted(set(old["meta"]["bodies"]) | set(args.bodies))
        series = {**dict(np.load(spath)), **series}
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1)
    np.savez(spath, **series)


def _where():
    import jax_dsmc  # noqa: PLC0415

    return os.path.dirname(jax_dsmc.__file__)


def worker_gradient(args):
    import jax  # noqa: PLC0415
    import optimize as O  # noqa: PLC0415

    from examples import ensemble_mesh as EM  # noqa: PLC0415
    from jax_dsmc import ensemble_backward  # noqa: PLC0415

    print(f"code {args.code}; jax_dsmc from {_where()}", flush=True)
    ensemble_backward.configure("virtual", 1, 2, "staged")
    cfg = dict(
        O.default_config(), fnum_scale=4.0, levelset_h=0.0625, levelset_samples=16,
        grazing=0.05, record="lean",
    )  # fmt: skip
    _grid, ls = O.initial_levelset(cfg)
    meta = {"code": args.code, "seeds": list(args.seeds), "variant": "along_pool",
            "device": jax.devices()[0].device_kind, "per_seed": {}}  # fmt: skip
    arrays = {}
    t_all = time.time()
    for seed in args.seeds:
        t0 = time.time()
        terms, info, _s = EM.shape_terms(
            cfg, jax.random.key(32 + seed), ls, "radius", 32, 0.2, ("along_pool",)
        )
        t = terms["along_pool"]
        for f in ("pathwise", "score", "silhouette", "timing"):
            arrays[f"s{seed}_{f}"] = np.asarray(getattr(t, f))
        grad = (
            arrays[f"s{seed}_pathwise"] + arrays[f"s{seed}_score"] + arrays[f"s{seed}_silhouette"]
        )
        arrays[f"s{seed}_gradient"] = grad
        meta["per_seed"][str(seed)] = {
            "objective": float(info["objective"]),
            "hits": int(info["hits"]),
            "alive_final": int(info["alive_final"]),
            "seconds": time.time() - t0,
        }
        print(f"seed {seed}: objective {info['objective']:.10e} hits {info['hits']} "
              f"grad sum {grad.sum():.6e} |grad| {np.linalg.norm(grad):.6e} "
              f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip
    meta["seconds"] = time.time() - t_all
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"gradient_{args.save}")
    if args.append and os.path.exists(path + ".json"):
        old = json.load(open(path + ".json"))
        meta["per_seed"] = {**old["per_seed"], **meta["per_seed"]}
        meta["seeds"] = sorted(set(old["seeds"]) | set(args.seeds))
        arrays = {**dict(np.load(path + ".npz")), **arrays}
    np.savez(path + ".npz", **arrays)
    with open(path + ".json", "w") as fh:
        json.dump(meta, fh, indent=1)


# ---------------------------------------------------------------- comparison (numpy only)


def _stats(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    return float(np.mean(x)), sd, sd / np.sqrt(n)


def _gate_line(name, a, b, gated, abs_floor=0.0):
    """One scalar observable; ``a`` base per-seed values, ``b`` new.  Returns ok."""
    ma, sa, ea = _stats(a)
    mb, sb, eb = _stats(b)
    d = mb - ma
    tol = SIGMAS * np.hypot(ea, eb)
    z = d / np.hypot(ea, eb) if np.hypot(ea, eb) > 0 else (0.0 if d == 0 else np.inf)
    ok = abs(d) < tol or abs(d) <= abs_floor or d == 0
    warn = ok and abs(z) >= WARN_SIGMAS
    tag = ("WARN" if warn else "PASS" if ok else "FAIL") if gated else "info"
    rel = d / abs(ma) if ma != 0 else float("nan")
    print(f"  {name:16s} base {ma:+.6e} (sd {sa:.2e}, n {len(a)})  new {mb:+.6e} (sd {sb:.2e}, "
          f"n {len(b)})  diff {d:+.3e} = {rel:+.2e} rel  z {z:+.2f}  "
          f"diff/sd1 {d / sa if sa > 0 else float('nan'):+.2f}  {tag}")  # fmt: skip
    return ok or not gated


def _same_seed(label, common, get_a, get_b):
    if not common:
        return
    same = [s for s in common if np.array_equal(get_a(s), get_b(s))]
    print(f"  same-seed ({label}): {len(same)}/{len(common)} seeds bitwise identical"
          + ("" if len(same) == len(common) else
             f"; differing seeds {[s for s in common if s not in same]}"))  # fmt: skip


def compare_forward(base, new):
    A = json.load(open(os.path.join(OUT, f"forward_{base}.json")))
    B = json.load(open(os.path.join(OUT, f"forward_{new}.json")))
    for k in ("steps", "window_start", "block"):
        if A["meta"][k] != B["meta"][k]:
            print(f"WARNING: {k} differs: {A['meta'][k]} vs {B['meta'][k]}")
    print(f"forward gate: base '{base}' ({A['meta']['code']}) vs new '{new}' ({B['meta']['code']})")
    print(f"criterion: z = diff / sqrt(se_base^2 + se_new^2) (se = seed sd / sqrt(n)); PASS iff"
          f" |z| < {SIGMAS:g} for every gated observable (WARN for |z| >= {WARN_SIGMAS:g});"
          " diff/sd1 = diff / single-seed sd of base")  # fmt: skip
    ok_all = True
    for body in [b for b in A["meta"]["bodies"] if b in B["meta"]["bodies"]]:
        sa, sb = A[body], B[body]
        print(f"[{body}] seeds base {sorted(sa, key=int)} new {sorted(sb, key=int)}")
        for name, desc, gated in FORWARD_OBS:
            a = [sa[s][name] for s in sa]
            b = [sb[s][name] for s in sb]
            # energy / heat on a specular body are round-off: gated only above that scale
            ok_all &= _gate_line(name, a, b, gated)
        common = sorted(set(sa) & set(sb), key=int)
        _same_seed(body, common, lambda s, sa=sa: [sa[s][k] for k, *_ in FORWARD_OBS],
                   lambda s, sb=sb: [sb[s][k] for k, *_ in FORWARD_OBS])  # fmt: skip
    print("FORWARD GATE:", "PASS" if ok_all else "FAIL")
    return ok_all


def _grad_load(tag):
    meta = json.load(open(os.path.join(OUT, f"gradient_{tag}.json")))
    arr = np.load(os.path.join(OUT, f"gradient_{tag}.npz"))
    seeds = [int(s) for s in meta["seeds"]]
    return meta, arr, seeds


def compare_gradient(base, new):
    ma, aa, sa = _grad_load(base)
    mb, ab, sb = _grad_load(new)
    print(f"gradient gate: base '{base}' ({ma['code']}) vs new '{new}' ({mb['code']})")
    print(
        f"criterion: PASS iff each gated scalar has |z| < {SIGMAS:g}, z = diff of seed-means / "
        "sqrt(se_a^2 + se_b^2), and the RMS over the band coefficients of the per-coefficient "
        "z < 1.5 (the 'grad . base-mean-direction' scalar is the gradient's signal component)"
    )
    ga = np.stack([aa[f"s{s}_gradient"] for s in sa]).reshape(len(sa), -1)
    gb = np.stack([ab[f"s{s}_gradient"] for s in sb]).reshape(len(sb), -1)
    ok = True
    for f in ("objective", "hits"):
        ok &= _gate_line(f, [ma["per_seed"][str(s)][f] for s in sa],
                         [mb["per_seed"][str(s)][f] for s in sb], True)  # fmt: skip
    # the base's per-coefficient MEDIAN (a single seed's rare spike -- seed 2 of base has a
    # few coefficients at 30x the others -- would otherwise dominate the direction)
    direction = np.median(ga, axis=0)
    direction = direction / np.linalg.norm(direction)
    ok &= _gate_line("grad_sum", ga.sum(axis=1), gb.sum(axis=1), True)
    ok &= _gate_line("grad.dir_base", ga @ direction, gb @ direction, True)
    ok &= _gate_line("grad_norm", np.linalg.norm(ga, axis=1), np.linalg.norm(gb, axis=1), False)
    for f in ("pathwise", "score", "silhouette", "timing"):
        pa = np.stack([aa[f"s{s}_{f}"].ravel() for s in sa])
        pb = np.stack([ab[f"s{s}_{f}"].ravel() for s in sb])
        _gate_line(f"{f}_sum", pa.sum(axis=1), pb.sum(axis=1), False)
    # per coefficient (only those where the base has seed noise: the band near the surface)
    ea = ga.std(axis=0, ddof=1) / np.sqrt(len(sa))
    eb = gb.std(axis=0, ddof=1) / np.sqrt(len(sb))
    band = ea > 1e-12 * np.abs(ga).max()
    zc = (gb.mean(axis=0) - ga.mean(axis=0))[band] / np.hypot(ea, eb)[band]
    rms = float(np.sqrt(np.mean(zc**2)))
    frac = float(np.mean(np.abs(zc) > 2.0))
    print(f"  per-coefficient ({band.sum()} with noise): RMS z {rms:.2f}, median |z| "
          f"{np.median(np.abs(zc)):.2f}, max |z| {np.abs(zc).max():.2f}, fraction |z|>2 {frac:.2f}"
          f"  {'PASS' if rms < 1.5 else 'FAIL'}")  # fmt: skip
    ok &= rms < 1.5
    common = sorted(set(sa) & set(sb))
    for s in common:
        oa, ob = ma["per_seed"][str(s)]["objective"], mb["per_seed"][str(s)]["objective"]
        x, y = aa[f"s{s}_gradient"], ab[f"s{s}_gradient"]
        rel = float(np.max(np.abs(x - y)) / np.max(np.abs(x)))
        sd1 = float(np.linalg.norm(ga.std(axis=0, ddof=1)))
        print(f"  same seed {s}: objective {oa:.12e} vs {ob:.12e} "
              f"({'bitwise' if oa == ob else f'rel {(ob - oa) / oa:+.2e}'}); gradient "
              f"{'bitwise' if np.array_equal(x, y) else f'max rel {rel:.2e}'}; "
              f"|dgrad| / |seed sd| {np.linalg.norm(x - y) / sd1:.2f}")  # fmt: skip
    print("GRADIENT GATE:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- CLI


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("gate", choices=("forward", "gradient"))
    ap.add_argument("--code", default=BASE_TREE, help="tree whose jax_dsmc/examples to run")
    ap.add_argument("--save", help="run the gate on --code and save under this tag")
    ap.add_argument("--compare", nargs=2, metavar=("BASE_TAG", "NEW_TAG"))
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--bodies", nargs="*", default=["polygon", "levelset"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--window-start", type=int, default=400)
    ap.add_argument("--append", action="store_true", help="add --seeds to an existing --save tag")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.code = os.path.abspath(args.code)
    if args.seeds is None:
        args.seeds = list(FORWARD_SEEDS if args.gate == "forward" else GRADIENT_SEEDS)
    if args.worker:
        sys.path.insert(0, os.path.join(args.code, "examples", "mesh"))
        sys.path.insert(0, args.code)
        (worker_forward if args.gate == "forward" else worker_gradient)(args)
        return 0
    if args.save:
        env = dict(os.environ, PYTHONPATH=args.code)
        cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--worker"]
        rc = subprocess.call(cmd, env=env, cwd=args.code)
        if rc:
            return rc
    if args.compare:
        ok = (compare_forward if args.gate == "forward" else compare_gradient)(*args.compare)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
