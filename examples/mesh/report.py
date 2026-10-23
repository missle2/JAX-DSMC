"""Forward gates FG1-FG7 of the mesh plan: our SPARTA duplicates vs the rebuilt SPARTA runs.

Reads ``results/forward_*.json`` (from ``forward.py``) and the SPARTA logs / dumps under
``results/sparta/``; prints markdown tables and writes ``figures/circle_fill.png``,
``figures/sphere_fill.png``, ``figures/circle_surface_pressure.png``,
``figures/sphere_surface_pressure.png``, ``figures/circle_fields_sparta_t1000.png``,
``figures/circle_fields_difference.png`` and ``results/forward_gates.json``.  Nothing here
is typed by hand: every number comes from an artifact.

    python3 examples/mesh/report.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import jax.numpy as jnp
import numpy as np

from jax_dsmc import viz
from jax_dsmc.grid import cell_of
from jax_dsmc.mesh import cut
from jax_dsmc.state import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figure_paths import FIGURES, figure_path, forward_folder  # noqa: E402
from forward import CASES, SPARTA  # noqa: E402
from sparta_io import read_grid_dump, read_log_stats, read_surf_dump, surf_columns  # noqa: E402

RESULTS = os.path.join(HERE, "results")
SPARTA_RESULTS = os.path.join(RESULTS, "sparta")


def load_ours(case: str, body="polygon", tracker="stencil", gas="Ar", acc=0.0) -> list[dict]:
    pattern = os.path.join(RESULTS, f"forward_{case}_{body}_{tracker}_{gas}_acc{acc:g}_*.json")
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            runs.append(json.load(fh))
    return runs


def load_sparta_logs(case: str, gas="ar") -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(os.path.join(SPARTA_RESULTS, f"log.{case}_{gas}.*"))):
        out[path.rsplit(".", 1)[1]] = read_log_stats(path)
    return out


def grid_for(case):
    c = CASES[case]
    return GridSpec(lo=jnp.array(c["lo"]), hi=jnp.array(c["hi"]), n=c["n"])


def _mean_se(rows):
    a = np.asarray(rows, dtype=float)
    n = a.shape[0]
    se = a.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.full(a.shape[1:], np.nan)
    return a.mean(axis=0), se, n


# --- FG1 diagnostics, FG2 fill curves ------------------------------------------------


def gate_diagnostics(runs) -> dict:
    keys = (
        "stencil_overrun",
        "walk_exhausted",
        "inside_body",
        "wall_recrossed",
        "attempts_clipped",
        "inflow_shortfall",
        "inflow_tracking_exhausted",
    )
    totals = {k: sum(r["diagnostics_total"].get(k, 0.0) for r in runs) for k in keys}
    return {"totals": totals, "pass": bool(all(v == 0 for v in totals.values())), "runs": len(runs)}


def gate_fill(case, runs, logs) -> dict:
    """Np(t) and Nscoll(t): ours vs SPARTA at every logged step, in units of the seed spread."""
    steps = np.asarray([r["step"] for r in runs[0]["stats"]])
    ours_np = np.asarray([[r["np"] for r in run["stats"]] for run in runs], float)
    # SPARTA logs the count of the logged step alone; compare like with like
    ours_ns = np.asarray([[r["nscoll_step"] for r in run["stats"]] for run in runs], float)
    sp_np, sp_ns = [], []
    for st in logs.values():
        idx = np.searchsorted(st["step"], steps)
        sp_np.append(st["np"][idx].astype(float))
        sp_ns.append(st["nscoll"][idx].astype(float))
    sp_np, sp_ns = np.asarray(sp_np), np.asarray(sp_ns)
    out = {"steps": steps.tolist()}
    for name, a, b in (("np", ours_np, sp_np), ("nscoll", ours_ns, sp_ns)):
        ma, sa, na = _mean_se(a)
        mb, sb, nb = _mean_se(b)
        z = (ma - mb) / np.sqrt(sa**2 + sb**2)
        tail = steps > steps[-1] - 300
        out[name] = {
            "ours_final": float(ma[-1]),
            "ours_se_final": float(sa[-1]),
            "sparta_final": float(mb[-1]),
            "sparta_se_final": float(sb[-1]),
            "z_final": float(z[-1]),
            "max_abs_z": float(np.nanmax(np.abs(z))),
            "max_abs_z_tail": float(np.nanmax(np.abs(z[tail]))),
            "ours_tail_mean": float(ma[tail].mean()),
            "sparta_tail_mean": float(mb[tail].mean()),
            "tail_z": float(
                (ma[tail].mean() - mb[tail].mean())
                / math.sqrt(((sa[tail] ** 2).mean() + (sb[tail] ** 2).mean()) / tail.sum())
            ),
            "seeds": [na, nb],
        }
    # Np by the seed-to-seed spread; the surface-collision rate by the window totals of
    # gate_surface (SPARTA's logged `nscoll` is a single-step sample, ten times noisier
    # than our block record) -- filled in by main()
    out["pass"] = bool(abs(out["np"]["tail_z"]) <= 2.0)
    return out, (steps, ours_np, ours_ns, sp_np, sp_ns)


def figure_fill(case, curves, air_log, path):
    steps, ours_np, ours_ns, sp_np, sp_ns = curves
    fig, (ax0, ax1) = viz.make_figure(1, 2, figsize=(11.2, 4.4))
    for k in range(sp_np.shape[0]):
        ax0.plot(
            steps, sp_np[k], color=viz.REFERENCE, lw=1, label="SPARTA argon" if k == 0 else None
        )
        ax1.plot(
            steps, sp_ns[k], color=viz.REFERENCE, lw=1, label="SPARTA argon" if k == 0 else None
        )
    for k in range(ours_np.shape[0]):
        ax0.plot(
            steps,
            ours_np[k],
            color=viz.SERIES_MEASURED,
            lw=1.2,
            label="JAX-DSMC" if k == 0 else None,
        )
        ax1.plot(
            steps,
            ours_ns[k],
            color=viz.SERIES_MEASURED,
            lw=1.2,
            label="JAX-DSMC" if k == 0 else None,
        )
    if air_log is not None:
        idx = np.searchsorted(air_log["step"], steps)
        ax0.plot(
            steps,
            air_log["np"][idx],
            color=viz.SERIES_ANALYTIC,
            lw=1,
            ls="--",
            label="SPARTA shipped N2/O2",
        )
        ax1.plot(
            steps,
            air_log["nscoll"][idx],
            color=viz.SERIES_ANALYTIC,
            lw=1,
            ls="--",
            label="SPARTA shipped N2/O2",
        )
    ax0.set_xlabel("step")
    ax0.set_ylabel("particles")
    ax1.set_xlabel("step")
    ax1.set_ylabel("surface collisions per step")
    ax0.legend(fontsize=8)
    ax0.set_title(f"{case}: fill from empty", fontsize=10)
    viz.savefig(fig, path)


# --- FG3 surface loads ------------------------------------------------------------


def sparta_surface(case, seeds) -> dict:
    per_seed = {}
    for seed in seeds:
        path = os.path.join(SPARTA_RESULTS, f"surf_ave.{case}_ar.{seed}.txt")
        s = surf_columns(read_surf_dump(path))
        order = np.argsort(s["id"])
        per_seed[seed] = {
            k: np.asarray(s[k])[order]
            for k in (
                "n",
                "press",
                "px",
                "py",
                "pz",
                "shx",
                "shy",
                "shz",
                "fx",
                "fy",
                "fz",
                "ke",
                "area",
            )
        }
    return per_seed


def gate_surface(case, runs, sp) -> dict:
    E = len(runs[0]["loads"]["press"])
    ours_press = np.asarray([r["loads"]["press"] for r in runs])
    ours_n = np.asarray([np.asarray(r["loads"]["hits"]) / r["window"] for r in runs], float)
    ours_fx = np.asarray([r["total_force"][0] for r in runs])
    sp_press = np.asarray([v["press"] for v in sp.values()])
    sp_n = np.asarray([v["n"] for v in sp.values()])
    sp_fx = np.asarray([v["fx"].sum() for v in sp.values()])
    mp, sep, _ = _mean_se(ours_press)
    mq, seq, _ = _mean_se(sp_press)
    z_el = (mp - mq) / np.sqrt(sep**2 + seq**2)
    z_el = z_el[np.isfinite(z_el)]
    mf, sef, _ = _mean_se(ours_fx[:, None])
    mg, seg, _ = _mean_se(sp_fx[:, None])
    z_fx = float((mf[0] - mg[0]) / math.sqrt(sef[0] ** 2 + seg[0] ** 2))
    return {
        "elements": E,
        "fx_ours": float(mf[0]),
        "fx_ours_se": float(sef[0]),
        "fx_sparta": float(mg[0]),
        "fx_sparta_se": float(seg[0]),
        "fx_z": z_fx,
        "fx_rel_diff": float(mf[0] / mg[0] - 1.0),
        "hits_per_step_ours": float(ours_n.sum(axis=1).mean()),
        "hits_per_step_sparta": float(sp_n.sum(axis=1).mean()),
        "press_rms_z": float(np.sqrt(np.mean(z_el**2))),
        "press_max_abs_z": float(np.abs(z_el).max()),
        "press_frac_within_2": float(np.mean(np.abs(z_el) <= 2.0)),
        "pass": bool(abs(z_fx) <= 2.0 and np.sqrt(np.mean(z_el**2)) <= 1.5),
    }, (mp, sep, mq, seq)


def figure_surface(case, runs, sp, stats, path):
    mp, sep, mq, seq = stats
    body = CASES[case]
    if body["dim"] == 2:
        from jax_dsmc.mesh import surface as S  # noqa: PLC0415

        surf = S.read_sparta_surf(os.path.join(SPARTA, body["surf"]), 2)
        cen = np.asarray(S.centroids(surf.points, surf.elements)) - np.asarray(body["centre"])
        angle = np.degrees(np.arctan2(cen[:, 1], cen[:, 0]))
        xlabel = "polar angle [deg] (0 = downstream)"
    else:
        from jax_dsmc.mesh import surface as S  # noqa: PLC0415

        surf = S.read_sparta_surf(os.path.join(SPARTA, body["surf"]), 3)
        cen = np.asarray(S.centroids(surf.points, surf.elements))
        angle = np.degrees(np.arccos(np.clip(-cen[:, 0] / np.linalg.norm(cen, axis=1), -1, 1)))
        xlabel = "angle from the stagnation point [deg]"
    order = np.argsort(angle)
    fig, (ax0, ax1) = viz.make_figure(1, 2, figsize=(11.2, 4.4))
    ax0.errorbar(
        angle[order],
        mq[order],
        yerr=seq[order],
        fmt="o",
        ms=3,
        color=viz.REFERENCE,
        label="SPARTA argon",
    )
    ax0.errorbar(
        angle[order],
        mp[order],
        yerr=sep[order],
        fmt="s",
        ms=3,
        color=viz.SERIES_MEASURED,
        label="JAX-DSMC",
    )
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel("normal pressure [N/m^2]")
    ax0.legend(fontsize=8)
    ax0.set_title(f"{case}: window-averaged surface pressure", fontsize=10)
    z = (mp - mq) / np.sqrt(sep**2 + seq**2)
    ax1.plot(angle[order], z[order], "o", ms=3, color=viz.SERIES_ERROR)
    ax1.axhline(2, color=viz.REFERENCE, lw=1)
    ax1.axhline(-2, color=viz.REFERENCE, lw=1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("(ours - SPARTA) / combined SE")
    viz.savefig(fig, path)


# --- FG4 fields ---------------------------------------------------------------------


def sparta_fields(case, seeds, grid):
    out = []
    for seed in seeds:
        d = read_grid_dump(os.path.join(SPARTA_RESULTS, f"grid_ave.{case}_ar.{seed}.txt"))
        D = len(grid.n)
        cols = [d["xc"], d["yc"]] + ([d["zc"]] if D == 3 else [])
        cid = np.asarray(cell_of(jnp.asarray(np.stack(cols, axis=1)), grid))
        C = int(np.prod(grid.n))
        f = {}
        for k, name in enumerate(("n", "nrho", "u", "v", "w", "temp"), start=1):
            arr = np.zeros(C)
            arr[cid] = d[f"f_ag[{k}]"]
            f[name] = arr
        out.append(f)
    return out


def gate_fields(case, runs, sp_fields) -> dict:
    """Per-cell window averages vs SPARTA's ``fix ave/grid`` of ``compute grid n nrho u v w temp``.

    SPARTA's ``temp`` is the **un-centred** kinetic temperature ``(m / 3k) <|v|^2>``
    (``compute_grid.cpp`` ``case TEMPERATURE``: ``tprefactor * sum(m v^2) / count``; the
    thermal one is ``compute thermal/grid``), so ours is compared as
    ``temp + m |u|^2 / (3 k)``.
    """
    res = {}
    for name, ours_key, comp in (("nrho", "nrho", None), ("u", "u", 0), ("temp", "temp", None)):
        a = np.asarray([np.asarray(r["fields"][ours_key]) for r in runs], float)
        if comp is not None:
            a = a[:, :, comp]
        if name == "temp":
            a = np.asarray([np.asarray(r["fields"]["temp_kinetic"]) for r in runs], float)
        b = np.asarray([f[name] for f in sp_fields], float)
        ma, sea, _ = _mean_se(a)
        mb, seb, _ = _mean_se(b)
        occupied = (mb != 0) & (ma != 0)
        z = (ma - mb) / np.sqrt(sea**2 + seb**2)
        z = z[occupied & np.isfinite(z)]
        res[name] = {
            "cells": int(z.size),
            "rms_z": float(np.sqrt(np.mean(z**2))),
            "frac_within_2": float(np.mean(np.abs(z) <= 2)),
            "mean_ours": float(ma[occupied].mean()),
            "mean_sparta": float(mb[occupied].mean()),
            "ours": ma,
            "sparta": mb,
            "z": (ma - mb) / np.sqrt(sea**2 + seb**2),
        }
    res["pass"] = bool(all(res[k]["rms_z"] <= 1.5 for k in ("nrho", "u", "temp")))
    return res


def figure_fields(case, grid, fields, overlay, path_sparta, path_diff, mask=None):
    n = tuple(int(x) for x in grid.n)
    plane = viz.Plane(2, fraction=0.5) if len(n) == 3 else None
    viz.field_snapshot(
        path_sparta,
        {
            "nrho": fields["nrho"]["sparta"],
            "u_x": fields["u"]["sparta"],
            "temp": fields["temp"]["sparta"],
        },
        n,
        grid.lo,
        grid.hi,
        title=f"{case}: SPARTA argon, window average (4 seeds)",
        plane=plane,
        surface=overlay,
        mask=mask,
    )
    fig, axes = viz.make_figure(1, 3, figsize=(13.8, 3.8))
    for ax, name in zip(axes, ("nrho", "u", "temp"), strict=True):
        z = np.nan_to_num(fields[name]["z"])
        im = viz.draw_field(
            ax,
            z,
            n,
            grid.lo,
            grid.hi,
            label=f"{name}: (ours - SPARTA)/SE",
            cmap=viz.FIELD_CMAP_DIVERGING,
            vmin=-3,
            vmax=3,
            plane=plane,
            mask=mask,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        viz.draw_surface(ax, overlay[0], overlay[1], plane=plane, lo=grid.lo, hi=grid.hi)
    viz.savefig(fig, path_diff)


# --- FG7 polygon vs analytic, tracker equivalence -------------------------------------


def compare_variants(case, base, other, label) -> dict:
    if not base or not other:
        return {"available": False}
    keys = {
        "np_final": lambda r: r["stats"][-1]["np"],
        "nscoll_tail": lambda r: (
            sum(s["nscoll"] for s in r["stats"] if s["step"] > r["steps"] - 300) / 300
        ),
        "fx": lambda r: r["total_force"][0],
    }
    out = {"available": True, "label": label}
    for k, f in keys.items():
        a = np.asarray([f(r) for r in base], float)
        b = np.asarray([f(r) for r in other], float)
        sa = a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else np.nan
        out[k] = {
            "base": float(a.mean()),
            "other": float(b.mean()),
            "rel_diff": float(b.mean() / a.mean() - 1),
            "base_se": float(sa),
        }
    return out


# --- G3a: the epsilon hit-timing score on a steady window -----------------------------


def window_score(path_json, path_fig) -> dict | None:
    """`scratch/mesh/probe_window_score.py` artifact: 1-D closed box, windowed impulse."""
    if not os.path.exists(path_json):
        return None
    with open(path_json) as fh:
        rows = json.load(fh)
    ws = sorted(int(w) for w in rows)
    fig, (ax,) = viz.make_figure(1, 1)
    fd = [rows[str(w)]["fd"][0] for w in ws]
    ax.errorbar(
        ws,
        fd,
        yerr=[rows[str(w)]["fd"][1] for w in ws],
        fmt="o-",
        color=viz.REFERENCE,
        label="ensemble FD (= -J/gap)",
    )
    for eps, color in (("0.02", viz.SERIES_ANALYTIC), ("0.01", viz.SERIES_MEASURED)):
        m = [rows[str(w)]["score"][eps][0] for w in ws]
        e = [rows[str(w)]["score"][eps][1] for w in ws]
        ax.errorbar(
            ws,
            m,
            yerr=e,
            fmt="s--",
            color=color,
            label=f"value-weighted eps-hit score, eps = {eps}",
        )
    ax.axhline(0, color=viz.GRID, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("window W [steps]")
    ax.set_ylabel("dJ/dgap")
    ax.legend(fontsize=8)
    ax.set_title(
        "G3a: the eps hit-timing score carries no resolvable mean on a steady window", fontsize=9
    )
    viz.savefig(fig, path_fig)
    out = {}
    for w in ws:
        r = rows[str(w)]
        out[w] = {
            "fd": r["fd"],
            "sd_fd": r["sd_fd"],
            "analytic": r["analytic"],
            "score": r["score"],
            "score_sd_over_fd_sd": {e: v[2] / r["sd_fd"] for e, v in r["score"].items()},
            "score_mean_in_fd_sd": {e: abs(v[0]) / r["sd_fd"] for e, v in r["score"].items()},
            "score_z": {e: v[0] / v[1] for e, v in r["score"].items()},
        }
    return out


# --- report -----------------------------------------------------------------------


GRADIENT_RESULTS = os.path.join(os.path.dirname(HERE), "gradient_bench", "results")
ADJOINT_GATES = (
    ("G1", "Couette windowed v_y^2, speed", "ensemble_couette_speed_*_dev_energy-window*.json"),
    ("G2", "Couette windowed shear, speed", "ensemble_couette_speed_*_dev_shear-window*.json"),
    (
        "G2",
        "Couette windowed shear, T_left",
        "ensemble_couette_temperature_left_*_dev_shear-window*.json",
    ),
    ("G3", "embedded plane position, windowed v_y^2", "ensemble_mesh_plane_energy-window*.json"),
    ("G3", "embedded plane position, windowed shear", "ensemble_mesh_plane_shear-window*.json"),
    ("D", "analytic circle radius, windowed drag", "ensemble_mesh_circle_radius_*.json"),
    ("D", "analytic circle centre_x, windowed drag", "ensemble_mesh_circle_centre_x_*.json"),
    (
        "LS2",
        "level-set circle radius, windowed drag",
        "ensemble_mesh_circle_levelset*_radius_*.json",
    ),
    (
        "LS2",
        "level-set circle cos 2θ mode, windowed drag",
        "ensemble_mesh_circle_levelset*_mode2_*.json",
    ),
    (
        "GP",
        "diffuse capsule plate half-width, windowed drag (silhouette term)",
        "ensemble_mesh_plate_levelset*_width_*.json",
    ),
    (
        "GD",
        "diffuse level-set circle radius, windowed drag (hybrid normal + silhouette terms)",
        "ensemble_mesh_circle_levelset*_acc1_*_radius_*.json",
    ),
    (
        "LS3",
        "level-set sphere radius, windowed drag",
        "ensemble_mesh_sphere_levelset*_radius_*.json",
    ),
)


def _figure_levelset(out, common, path_fig):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    x = np.arange(len(common))
    titles = ("windowed drag J", "ensemble gradient dJ/dr", "CRN FD dJ/dr")
    for ax, k, title in zip(axes, ("objective", "gradient", "fd"), titles, strict=True):
        a, b = np.array(out[k]["analytic"]), np.array(out[k]["levelset"])
        ax.plot(x, a, "o", color="0.35", label="analytic circle")
        ax.plot(x, b, "x", color="tab:blue", label="level set")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in common], fontsize=7)
        ax.set_xlabel("seed")
        ax.set_title(f"{title}\nmax |Δ| / max = {out[k]['max_rel_diff']:.2e}", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)


def _rel_diff(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-300))


def levelset_vs_analytic(path_fig) -> dict | None:
    """LS1/LS2: the level-set circle against the analytic circle **seed by seed** (common
    random numbers: the forwards differ only by the level set's surface error, so the
    objective, the CRN finite differences, the exact tally / timing terms and every
    ``grad_x lambda`` variant's gradient must agree far inside the seed-to-seed spread).
    The analytic rows come from the ``_variants`` artifact (all four estimators stored per
    seed).  Figure ``levelset_gradient.png``."""
    pat_a = os.path.join(GRADIENT_RESULTS, "ensemble_mesh_circle_radius_*_dev_variants.json")
    pat_l = os.path.join(GRADIENT_RESULTS, "ensemble_mesh_circle_levelset*_radius_*_dev.json")
    paths_a, paths_l = sorted(glob.glob(pat_a)), sorted(glob.glob(pat_l))
    if not paths_a or not paths_l:
        return None
    with open(paths_a[0]) as fh:
        da = json.load(fh)
    with open(paths_l[0]) as fh:
        dl = json.load(fh)
    rows_a = {r["seed"]: r for r in da["rows"]}
    rows_l = {r["seed"]: r for r in dl["rows"]}
    common = sorted(set(rows_a) & set(rows_l))
    if not common:
        return None
    out = {
        "seeds": common,
        "artifact_analytic": os.path.basename(paths_a[0]),
        "artifact_levelset": os.path.basename(paths_l[0]),
    }
    for k in ("objective", "pathwise_tally", "timing", "hits"):
        a = [rows_a[s][k] for s in common]
        b = [rows_l[s][k] for s in common]
        out[k] = {"analytic": a, "levelset": b, "max_rel_diff": _rel_diff(a, b)}
    fd_a = [rows_a[s]["fd"][0] for s in common]
    fd_l = [rows_l[s]["fd"][0] for s in common]
    out["fd"] = {"analytic": fd_a, "levelset": fd_l, "max_rel_diff": _rel_diff(fd_a, fd_l)}
    out["variants"] = {}
    for name in rows_l[common[0]]["variants"]:
        entry = {}
        for k in ("gradient", "pathwise_x", "pathwise_v"):
            a = [rows_a[s]["variants"][name][k] for s in common]
            b = [rows_l[s]["variants"][name][k] for s in common]
            entry[k] = {"analytic": a, "levelset": b, "max_rel_diff": _rel_diff(a, b)}
        out["variants"][name] = entry
    variant = dl["config"].get("variant", "along_pool")
    out["variant"] = variant
    ga = np.array(out["variants"][variant]["gradient"]["analytic"])
    gl = np.array(out["variants"][variant]["gradient"]["levelset"])
    out["gradient"] = {
        "analytic": ga.tolist(),
        "levelset": gl.tolist(),
        "max_rel_diff": _rel_diff(ga, gl),
        "seed_sd_analytic": float(np.std(ga, ddof=1)) if len(common) > 1 else float("nan"),
    }
    out["gradient"]["max_abs_diff_over_seed_sd"] = float(
        np.max(np.abs(ga - gl)) / out["gradient"]["seed_sd_analytic"]
    )
    st_l = _gate_stats(dl["rows"], dl["fd_steps"])
    out["levelset_stats"] = st_l
    # the analytic reference of the same variant, over the same seeds
    out["analytic_mean_same_seeds"] = float(ga.mean())
    out["levelset_mean_same_seeds"] = float(gl.mean())
    _figure_levelset(out, common, path_fig)
    return out


def grazing_regularisation(path_fig) -> dict | None:
    """``scratch/mesh/probe_grazing_sweep.py`` (copied to ``results/grazing_sweep.json``):
    the pathwise tally term of the circle's radius gradient vs the grazing cut-off ``eps``
    of the hit-time derivative, per seed -- its seed-to-seed sd, the paired bias against
    ``eps = 0`` and the largest per-hit weight.  Figure ``grazing_regularisation.png``."""
    path = os.path.join(RESULTS, "grazing_sweep.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    rows = d["rows"]
    eps = [str(e) for e in d["eps"]]
    ref = np.array([r["eps"]["0.0"]["tally"] for r in rows])
    n = len(rows)
    out = {"seeds": n, "eps": {}}
    for e in eps:
        t = np.array([r["eps"][e]["tally"] for r in rows])
        bias = t - ref
        out["eps"][e] = {
            "mean": float(t.mean()),
            "sd": float(t.std(ddof=1)),
            "sd_rel": float(t.std(ddof=1) / abs(t.mean())),
            "bias": float(bias.mean()),
            "bias_se": float(bias.std(ddof=1) / math.sqrt(n)),
            "bias_rel": float(bias.mean() / abs(ref.mean())),
            "max_per_hit": float(np.mean([r["eps"][e]["max_abs_per_hit"] for r in rows])),
            "rms_per_hit": float(np.mean([r["eps"][e]["rms_per_hit"] for r in rows])),
            "hits_below": float(np.mean([r["eps"][e]["hits_below_cutoff"] for r in rows])),
        }
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    x = [max(float(e), 3e-3) for e in eps]
    axes[0].semilogx(
        x, [out["eps"][e]["sd_rel"] * 100 for e in eps], "o-", label="seed-to-seed sd (%)"
    )
    axes[0].errorbar(
        x,
        [out["eps"][e]["bias_rel"] * 100 for e in eps],
        yerr=[out["eps"][e]["bias_se"] / abs(ref.mean()) * 100 for e in eps],
        fmt="s--",
        capsize=3,
        label="paired bias vs eps = 0 (%)",
    )
    axes[0].axhline(0.0, color="0.6", lw=1)
    axes[0].set_xlabel("grazing cut-off eps (eps = 0 drawn at 3e-3)")
    axes[0].set_ylabel("% of the tally term")
    axes[0].set_title(f"tally term of dJ/dr, {n} seeds", fontsize=8)
    axes[0].legend(fontsize=7)
    axes[1].loglog(
        x, [out["eps"][e]["max_per_hit"] for e in eps], "o-", label="largest per-hit weight"
    )
    axes[1].loglog(
        x, [out["eps"][e]["rms_per_hit"] for e in eps], "s--", label="rms per-hit weight"
    )
    axes[1].set_xlabel("grazing cut-off eps")
    axes[1].set_ylabel("|d v_x,post / d r| per hit")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)
    return out


def print_grazing(gz):
    print("\n### Grazing regularisation of the tally term (circle radius, CRN seeds)\n")
    print(
        "| eps | tally mean | seed sd (%) | paired bias vs eps = 0 (%) | largest per-hit weight | hits below cut-off / window |"
    )
    print("|---|---|---|---|---|---|")
    for e, r in gz["eps"].items():
        print(
            f"| {e} | {r['mean']:.4g} | {r['sd_rel'] * 100:.1f} | {r['bias_rel'] * 100:+.2f} ± "
            f"{r['bias_se'] / abs(gz['eps']['0.0']['mean']) * 100:.2f} | {r['max_per_hit']:.3g} | {r['hits_below']:.1f} |"
        )


SCORE_FORMS = (
    ("_prefloor", "score form, hit-time cut-off only"),
    ("_scoreform", "score form + emission-side floor"),
)


def score_floor_comparison() -> dict | None:
    """GP: the diffuse plate gate's three estimators of the normal dependence **seed by
    seed** (CRN: the forward, the FD, the silhouette and the timing terms are identical):
    the score form with the hit-time cut-off only (``_prefloor``), the score form with the
    emission-side floor of ``1 / (v_out . n)`` (``_scoreform``), and the shipped hybrid
    (the hit's own reward pathwise through the reparameterised sample, the value-to-go
    after the hit by the score; the default artifact).  Per seed ``score +
    pathwise_tally`` (the whole normal dependence) and the total, their seed sd and the
    paired difference against the hybrid."""
    pat = os.path.join(GRADIENT_RESULTS, "ensemble_mesh_plate_levelset*_gz*_width_*.json")
    out = {}
    for path in sorted(glob.glob(pat)):
        if any(suffix in path for suffix, _ in SCORE_FORMS):
            continue
        with open(path) as fh:
            d_h = json.load(fh)
        hyb = {r["seed"]: r for r in d_h["rows"]}
        entry = {"forms": {}}
        for suffix, label in SCORE_FORMS:
            path_f = path.replace("_width_", f"{suffix}_width_")
            if not os.path.exists(path_f):
                continue
            with open(path_f) as fh:
                d_f = json.load(fh)
            form = {r["seed"]: r for r in d_f["rows"]}
            common = sorted(set(hyb) & set(form))
            if len(common) < 2:
                continue
            f_entry = {"label": label, "seeds": common, "artifact": os.path.basename(path_f)}
            for k in ("objective", "fd", "silhouette", "timing"):
                a = [form[s_][k] for s_ in common]
                b = [hyb[s_][k] for s_ in common]
                f_entry[k] = {"max_rel_diff": _rel_diff(a, b)}
            for k in ("normal", "gradient"):
                if k == "normal":
                    a = np.array([form[s_]["score"] + form[s_]["pathwise_tally"] for s_ in common])
                    b = np.array([hyb[s_]["score"] + hyb[s_]["pathwise_tally"] for s_ in common])
                else:
                    a = np.array([form[s_][k] for s_ in common], dtype=float)
                    b = np.array([hyb[s_][k] for s_ in common], dtype=float)
                n = len(common)
                f_entry[k] = {
                    "form": a.tolist(),
                    "hybrid": b.tolist(),
                    "sd_form": float(a.std(ddof=1)),
                    "sd_hybrid": float(b.std(ddof=1)),
                    "paired_diff": float((b - a).mean()),
                    "paired_diff_se": float((b - a).std(ddof=1) / math.sqrt(n)),
                }
            entry["forms"][suffix] = f_entry
        if not entry["forms"]:
            continue
        seeds = sorted(hyb)
        entry["seeds"] = seeds
        entry["hybrid"] = {
            "score": [hyb[s_]["score"] for s_ in seeds],
            "pathwise_tally": [hyb[s_]["pathwise_tally"] for s_ in seeds],
            "gradient": [hyb[s_]["gradient"] for s_ in seeds],
        }
        st = _gate_stats(d_h["rows"], d_h["fd_steps"])
        entry["hybrid_stats"] = st
        out[os.path.basename(path)] = entry
    return out or None


def print_score_floor(sf):
    print("\n### GP: the diffuse normal term, three estimators seed by seed (CRN)\n")
    for name, e in sf.items():
        st = e["hybrid_stats"]
        h = str(st["reference_h"])
        print(
            f"`{name}` (hybrid, shipped): {st['seeds']} seeds, ensemble {st['ensemble']:.4g} ± "
            f"{st['ensemble_se']:.2g} vs FD {st['fd'][h]['mean']:.4g} ± {st['fd'][h]['se']:.2g}, "
            f"z = {st['z']:+.2f}; per seed score {', '.join(f'{v:.3g}' for v in e['hybrid']['score'])}; "
            f"tally {', '.join(f'{v:.3g}' for v in e['hybrid']['pathwise_tally'])}\n"
        )
        print(
            "| estimator | seeds | identical: J, FD, silhouette, timing (max rel Δ) | normal term per seed | seed sd (form → hybrid) | paired Δ (hybrid - form) | total seed sd (form → hybrid) |"
        )
        print("|---|---|---|---|---|---|---|")
        for suffix, f in e["forms"].items():
            ident = max(f[k]["max_rel_diff"] for k in ("objective", "fd", "silhouette", "timing"))
            r, g = f["normal"], f["gradient"]
            print(
                f"| {f['label']} (`{suffix}`) | {f['seeds']} | {ident:.1e} | "
                f"{', '.join(f'{v:.3g}' for v in r['form'])} | {r['sd_form']:.2g} → {r['sd_hybrid']:.2g} | "
                f"{r['paired_diff']:+.3g} ± {r['paired_diff_se']:.2g} | {g['sd_form']:.2g} → {g['sd_hybrid']:.2g} |"
            )


def plate_fd_steps() -> dict | None:
    """GP: the CRN finite difference of the plate's width gradient vs the FD step on the
    gate's seeds (``results/plate_fd_steps.json`` from
    ``scratch/mesh/probe_plate_fd_steps.py`` merged with the gate artifact's two steps),
    against the shipped hybrid estimate -- the FD's noise grows as ``1 / h`` (the
    hit-set jumps under common random numbers) while its mean shows no drift."""
    path = os.path.join(RESULTS, "plate_fd_steps.json")
    pat = os.path.join(GRADIENT_RESULTS, "ensemble_mesh_plate_levelset*_gz*_width_*.json")
    paths = [q for q in sorted(glob.glob(pat)) if not any(sfx in q for sfx, _ in SCORE_FORMS)]
    if not os.path.exists(path) or not paths:
        return None
    with open(path) as fh:
        d = json.load(fh)
    with open(paths[0]) as fh:
        g = json.load(fh)
    ens = {r["seed"]: r["gradient"] for r in g["rows"]}
    seeds = sorted(set(ens) & {r["seed"] for r in d["rows"]})
    if len(seeds) < 2:
        return None
    e = np.array([ens[s_] for s_ in seeds])
    fd = {r["seed"]: r["fd"] for r in d["rows"]}
    out = {
        "seeds": seeds,
        "ensemble": float(e.mean()),
        "ensemble_se": float(e.std(ddof=1) / math.sqrt(len(seeds))),
        "steps": {},
    }
    for h in d["steps"]:
        a = np.array([fd[s_][h] for s_ in seeds], dtype=float)
        z = (e.mean() - a.mean()) / math.sqrt((e.var(ddof=1) + a.var(ddof=1)) / len(seeds))
        out["steps"][h] = {
            "per_seed": a.tolist(),
            "mean": float(a.mean()),
            "se": float(a.std(ddof=1) / math.sqrt(len(seeds))),
            "z": float(z),
        }
    return out


def print_plate_fd_steps(fs):
    print(
        f"\n### GP: CRN finite difference vs step (seeds {fs['seeds']}; hybrid ensemble "
        f"{fs['ensemble']:.4g} ± {fs['ensemble_se']:.2g})\n"
    )
    print("| h | FD per seed | mean ± SE | z(ensemble - FD) |")
    print("|---|---|---|---|")
    for h, r in fs["steps"].items():
        print(
            f"| {h} | {', '.join(f'{v:.3g}' for v in r['per_seed'])} | {r['mean']:.4g} ± "
            f"{r['se']:.2g} | {r['z']:+.2f} |"
        )


def figure_plate_estimators(sf, fs, path_fig):  # noqa: PLR0915 -- three panels, one figure
    """``plate_estimators.png``: the GP plate's three estimators of the diffuse normal term
    seed by seed (left), their totals against the per-seed CRN FD (middle) and the FD vs
    its step with the hybrid's ensemble band (right).  All from the artifacts of
    :func:`score_floor_comparison` and :func:`plate_fd_steps`."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    name, e = next(iter(sf.items()))
    seeds = e["seeds"]
    x = np.arange(len(seeds))
    forms = [(f["label"], f) for f in e["forms"].values()]
    hyb_normal = np.array(e["hybrid"]["score"]) + np.array(e["hybrid"]["pathwise_tally"])
    hyb_total = np.array(e["hybrid"]["gradient"])
    colours = {"hybrid": "tab:blue", 0: "0.55", 1: "tab:orange"}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

    ax = axes[0]
    for j, (label, f) in enumerate(forms):
        a = np.array(f["normal"]["form"])
        ax.plot(
            x,
            a,
            "s" if j else "^",
            color=colours[j],
            label=f"{label}: sd {f['normal']['sd_form']:.2g}",
        )
        ax.axhline(a.mean(), color=colours[j], lw=0.8, ls=":")
    ax.plot(
        x,
        hyb_normal,
        "o",
        color=colours["hybrid"],
        label=f"hybrid (shipped): sd {np.std(hyb_normal, ddof=1):.2g}",
    )
    ax.axhline(hyb_normal.mean(), color=colours["hybrid"], lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s_) for s_ in seeds])
    ax.set_xlabel("seed (common random numbers)")
    ax.set_ylabel("normal term  dJ/dw")
    ax.set_title("diffuse normal term, three estimators", fontsize=9)
    ax.legend(fontsize=6.5)

    ax = axes[1]
    fd_steps = list(fs["steps"])
    for h, marker in zip(fd_steps[:3], ("x", "+", "1"), strict=False):
        ax.plot(x, fs["steps"][h]["per_seed"], marker, color="k", ms=8, label=f"CRN FD, h = {h}")
    for j, (label, f) in enumerate(forms):
        ax.plot(
            x, f["gradient"]["form"], "s" if j else "^", color=colours[j], label=label, alpha=0.8
        )
    ax.plot(x, hyb_total, "o", color=colours["hybrid"], label="hybrid (shipped)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s_) for s_ in seeds])
    ax.set_xlabel("seed")
    ax.set_ylabel("total  dJ/dw")
    ax.set_title("totals vs the same seed's finite difference", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")

    ax = axes[2]
    hs = np.array([float(h) for h in fd_steps])
    m = np.array([fs["steps"][h]["mean"] for h in fd_steps])
    se = np.array([fs["steps"][h]["se"] for h in fd_steps])
    ax.errorbar(hs, m, yerr=se, fmt="o", color="k", capsize=3, label="CRN FD ± SE")
    ax.axhspan(
        fs["ensemble"] - fs["ensemble_se"],
        fs["ensemble"] + fs["ensemble_se"],
        color=colours["hybrid"],
        alpha=0.25,
        label="hybrid ensemble ± SE",
    )
    ax.axhline(fs["ensemble"], color=colours["hybrid"])
    for j, (label, f) in enumerate(forms):
        a = np.array(f["gradient"]["form"])
        ax.axhline(a.mean(), color=colours[j], ls="--", lw=0.9, label=f"{label} (mean)")
    ax.set_xscale("log")
    ax.set_xlabel("FD step h")
    ax.set_ylabel("dJ/dw")
    ax.set_title(f"FD vs step, {len(seeds)} seeds", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.suptitle(f"GP: {name}", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)


def print_levelset(ls):
    print("\n### LS1/LS2 level-set circle vs analytic circle, seed by seed (CRN)\n")
    print("| quantity | max |Δ| / max |analytic| over common seeds |")
    print("|---|---|")
    for k in ("objective", "fd", "hits", "pathwise_tally", "timing"):
        print(f"| {k} | {ls[k]['max_rel_diff']:.2e} |")
    for name, entry in ls["variants"].items():
        print(
            f"| gradient, variant `{name}` (x-part, v-part) | {entry['gradient']['max_rel_diff']:.2e} "
            f"({entry['pathwise_x']['max_rel_diff']:.2e}, {entry['pathwise_v']['max_rel_diff']:.2e}) |"
        )
    st = ls["levelset_stats"]
    h = str(st["reference_h"])
    print(
        f"\nlevel set (`{ls['variant']}`): {st['seeds']} seeds, ensemble {st['ensemble']:+.4g} ± "
        f"{st['ensemble_se']:.2g}, FD {st['fd'][h]['mean']:+.4g} ± {st['fd'][h]['se']:.2g}, "
        f"z(FD) = {st['z']:+.2f}; same seeds analytic mean {ls['analytic_mean_same_seeds']:+.4g} "
        f"vs level set {ls['levelset_mean_same_seeds']:+.4g}; per-seed gradient difference ≤ "
        f"{ls['gradient']['max_abs_diff_over_seed_sd']:.4f} seed-sd (seeds {ls['seeds']})"
    )


def _gate_stats(rows, hs):
    g = np.array([r["gradient"] for r in rows], dtype=float)
    fd = np.array([r["fd"] for r in rows], dtype=float)
    n = len(g)
    out = {
        "seeds": n,
        "objective": float(np.mean([r["objective"] for r in rows])),
        "ensemble": float(g.mean()),
        "ensemble_se": float(g.std(ddof=1) / math.sqrt(n)),
        "fd": {},
    }
    for j, h in enumerate(hs):
        m, sd = fd[:, j].mean(), fd[:, j].std(ddof=1)
        z = (g.mean() - m) / math.sqrt((g.var(ddof=1) + sd**2) / n)
        out["fd"][str(h)] = {"mean": float(m), "se": float(sd / math.sqrt(n)), "z": float(z)}
    if "pathwise" in rows[0]:
        out["pathwise"] = float(np.mean([r["pathwise"] for r in rows]))
        out["score"] = float(np.mean([r["score"] for r in rows]))
        out["timing"] = float(np.mean([r.get("timing", 0.0) for r in rows]))
    if "analytic_F_over_r" in rows[0]:
        out["analytic"] = float(np.mean([r["analytic_F_over_r"] for r in rows]))
    for part in ("pathwise_x", "pathwise_v", "pathwise_tally", "timing", "silhouette"):
        if part in rows[0]:
            a = np.array([r[part] for r in rows], dtype=float)
            out[f"{part}_mean"] = float(a.mean())
            out[f"{part}_se"] = float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    # the gate: |z| <= 2 against the finest FD step that is still resolved (SE < 0.5 |mean|),
    # else the largest step
    resolved = [h for h in hs if out["fd"][str(h)]["se"] < 0.5 * abs(out["fd"][str(h)]["mean"])]
    h_ref = min(resolved) if resolved else max(hs)
    out["reference_h"] = h_ref
    out["z"] = out["fd"][str(h_ref)]["z"]
    out["pass"] = bool(abs(out["z"]) <= 2.0)
    return out


def adjoint_gates(path_fig) -> dict:
    """Gates G1-G3 (plan section 4) from the ``gradient_bench`` artifacts of
    ``examples.ensemble_couette --objective ...`` and ``examples.mesh.ensemble_plane``."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    gates = {}
    for gate, label, pattern in ADJOINT_GATES:
        for path in sorted(glob.glob(os.path.join(GRADIENT_RESULTS, pattern))):
            with open(path) as fh:
                d = json.load(fh)
            if not d["rows"]:
                continue
            name = os.path.basename(path)
            if gate == "LS2" and "_acc" in name:
                continue  # the diffuse circle has its own row (GD)
            st = _gate_stats(d["rows"], d["fd_steps"])
            note = " [two-sided fit, no timing term: superseded]" if "twosided" in name else ""
            note += " [one-sided ghost fits]" if "_1s" in name else ""
            for suffix, form_label in SCORE_FORMS:
                note += f" [{form_label}]" if suffix in name else ""
            if gate == "GP":
                note += " [specular wall: no silhouette]" if "_acc" not in name else ""
                note += " [coarse h_φ = 0.125]" if "levelset0.125" in name else ""
            # the grad-lambda fits: stratified by the hit-ahead class (2026-09-22) or the
            # mixed pool of the earlier artifacts (a -27 % bias on the capsule)
            mixed = d.get("config", {}).get("value_class", "none") == "none"
            if name.startswith("ensemble_mesh_") and "plane" not in name and mixed:
                note += " [mixed-pool ∇λ fits]"
            st.update({"gate": gate, "label": label + note, "artifact": name})
            gates[os.path.basename(path)] = st
    if not gates:
        return gates
    fig, axes = plt.subplots(1, len(gates), figsize=(3.2 * len(gates), 3.4), squeeze=False)
    for ax, st in zip(axes[0], gates.values(), strict=True):
        hs = list(st["fd"])
        ax.errorbar(
            range(len(hs)),
            [st["fd"][h]["mean"] for h in hs],
            yerr=[st["fd"][h]["se"] for h in hs],
            fmt="o",
            color="0.35",
            capsize=3,
            label="CRN FD",
        )
        ax.axhspan(
            st["ensemble"] - st["ensemble_se"],
            st["ensemble"] + st["ensemble_se"],
            color="tab:blue",
            alpha=0.25,
            label="ensemble ± SE",
        )
        ax.axhline(st["ensemble"], color="tab:blue")
        ax.set_xticks(range(len(hs)))
        ax.set_xticklabels([f"h={h}" for h in hs], fontsize=8)
        ax.set_title(
            f"{st['gate']}: {st['label']}\n{st['seeds']} seeds, z = {st['z']:+.2f}", fontsize=8
        )
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)
    return gates


def figure_shape_terms(gates, path_fig):
    """Per-term breakdown of the shape cases (plan section 5a, ``shape_circle_*.png``):
    the x / v / tally parts of the pathwise term, the timing term, their sum, the CRN FD
    and the analytic ``F/r``."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cases = [st for st in gates.values() if "pathwise_x_mean" in st]
    if not cases:
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 3.6), squeeze=False)
    for ax, st in zip(axes[0], cases, strict=True):
        names = ["x", "v", "tally", "timing", "ensemble", "FD", "F/r"]
        h = str(st["reference_h"])
        means = [
            st["pathwise_x_mean"],
            st["pathwise_v_mean"],
            st["pathwise_tally_mean"],
            st["timing_mean"],
            st["ensemble"],
            st["fd"][h]["mean"],
            st.get("analytic", float("nan")),
        ]
        errs = [
            st["pathwise_x_se"],
            st["pathwise_v_se"],
            st["pathwise_tally_se"],
            st["timing_se"],
            st["ensemble_se"],
            st["fd"][h]["se"],
            0.0,
        ]
        colors = ["0.6", "0.6", "0.6", "0.8", "tab:blue", "0.35", "tab:red"]
        ax.bar(names, means, yerr=errs, color=colors, capsize=3)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{st['label']}\n{st['seeds']} seeds, z = {st['z']:+.2f}", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)


def _truncated_fd(st):
    """The truncated-horizon FD of a plane artifact (``scratch/mesh/probe_plane_truncated_fd.py``:
    warm-up at theta = 0, captured steps at theta in {0, h, 2h} from the same warm state) --
    the derivative the truncated adjoint estimates.  ``None`` if not measured."""
    name = st["artifact"]
    if not name.startswith("ensemble_mesh_plane_"):
        return None
    objective = "energy-window" if "energy-window" in name else "shear-window"
    m = name.split("_M")[1].split("_")[0]
    w = name.split("-window")[1].split("_")[0]
    pattern = os.path.join(RESULTS, f"plane_truncated_fd_{objective}_M{m}_W{w}*.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            d = json.load(fh)
        rows += d["rows"] if isinstance(d, dict) else d
    if not rows:
        return None
    a = np.array([r["fd1_0.04"] for r in rows], dtype=float)
    return {
        "mean": float(a.mean()),
        "se": float(a.std(ddof=1) / math.sqrt(len(a))),
        "seeds": len(a),
    }


def _fit_bias(suffix=""):
    """``scratch/mesh/probe_plane_fitbias.py``: the x/v-part of the plane estimator with
    fitted (one-/two-sided) vs replica-exact value gradients, per seed."""
    path = os.path.join(RESULTS, f"plane_fit_bias{suffix}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = json.load(fh)

    def pooled(key):
        m = np.array([r[key][0] for r in rows])
        se = np.array([r[key][1] for r in rows])
        wgt = 1.0 / se**2
        return float(np.sum(wgt * m) / wgt.sum()), float(1.0 / math.sqrt(wgt.sum()))

    return {
        "seeds": len(rows),
        "x_part_true": pooled("x_part_true"),
        "x_part_fit_onesided": pooled("x_part_fit_onesided"),
        "x_part_fit_twosided": pooled("x_part_fit_twosided"),
        "bias_onesided": pooled("bias_onesided"),
        "bias_twosided": pooled("bias_twosided"),
    }


FIT_CLASSES = ("hit", "near-out", "near-in", "wall0-in", "wall0-out", "interior")


def fit_classes(path_fig) -> dict | None:
    """``scratch/mesh/probe_fit_classes.py``: fitted value / grad_x / grad_v against
    replica-exact truths per particle class (plane case), two-sided vs one-sided fits."""
    path = os.path.join(RESULTS, "plane_fit_classes.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        seeds = json.load(fh)
    rows = [r for sd in seeds for r in sd["rows"]]
    out = {"seeds": len(seeds)}
    for c in FIT_CLASSES:
        rr = [r for r in rows if r["cls"] == c]
        if not rr:
            continue
        tv = np.array([r["true"][0] for r in rr])
        tgx = np.array([r["true"][1] for r in rr])[:, 0]  # the wall-normal component
        entry = {
            "n": len(rr),
            "scale_value": float(np.sqrt(np.mean(tv**2))),
            "scale_gx": float(np.sqrt(np.mean(tgx**2))),
        }
        for nm in ("two", "one"):
            fv = np.array([r[f"fit_{nm}"][0] for r in rr])
            fgx = np.array([r[f"fit_{nm}"][1] for r in rr])[:, 0]
            for q, f, t in (("value", fv, tv), ("gx", fgx, tgx)):
                e = f - t
                entry[f"{nm}_{q}_bias"] = float(e.mean())
                entry[f"{nm}_{q}_bias_se"] = float(e.std(ddof=1) / math.sqrt(len(e)))
                entry[f"{nm}_{q}_rms"] = float(np.sqrt(np.mean(e**2)))
        out[c] = entry
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    classes = [c for c in FIT_CLASSES if c in out]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, q, title in zip(
        axes, ("gx", "value"), ("d lambda / dx (wall-normal)", "value"), strict=True
    ):
        xs = np.arange(len(classes))
        for off, nm, col in ((-0.2, "two", "0.6"), (0.2, "one", "tab:blue")):
            ax.bar(
                xs + off,
                [out[c][f"{nm}_{q}_rms"] / max(out[c][f"scale_{q}"], 1e-300) for c in classes],
                0.4,
                color=col,
                label=f"{nm}-sided rms / scale",
            )
            ax.errorbar(
                xs + off,
                [
                    abs(out[c][f"{nm}_{q}_bias"]) / max(out[c][f"scale_{q}"], 1e-300)
                    for c in classes
                ],
                yerr=[
                    out[c][f"{nm}_{q}_bias_se"] / max(out[c][f"scale_{q}"], 1e-300) for c in classes
                ],
                fmt="k_",
                capsize=2,
                label="|bias| / scale" if nm == "two" else None,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(classes, fontsize=8, rotation=20)
        ax.set_yscale("log")
        ax.set_title(f"fit error of {title} by particle class", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)
    return out


def print_fit_classes(fc):
    print(f"\n### Fit error by particle class (plane, replica-exact truth, {fc['seeds']} seeds)\n")
    print(
        "| class | n | scale d lambda/dx | two-sided bias | one-sided bias | rms two / one | value scale | value bias two / one | value rms two / one |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for c in FIT_CLASSES:
        if c not in fc:
            continue
        e = fc[c]
        print(
            f"| {c} | {e['n']} | {e['scale_gx']:.2e} | {e['two_gx_bias']:+.2e} ± {e['two_gx_bias_se']:.1e} | "
            f"{e['one_gx_bias']:+.2e} ± {e['one_gx_bias_se']:.1e} | {e['two_gx_rms']:.2e} / {e['one_gx_rms']:.2e} | "
            f"{e['scale_value']:.2e} | {e['two_value_bias']:+.2e} / {e['one_value_bias']:+.2e} | "
            f"{e['two_value_rms']:.2e} / {e['one_value_rms']:.2e} |"
        )


VARIANT_NAMES = ("two", "one", "one_pool", "along", "along_pool")
VARIANT_LABELS = {
    "two": "two-sided fit",
    "one": "one-sided fit",
    "one_pool": "one-sided, pooled 3^D (k = 96)",
    "along": "one-sided + along-flight identity",
    "along_pool": "pooled + along-flight identity",
}


def variants_probe(suffix=""):
    """``scratch/mesh/probe_plane_fitbias.py`` (variants form): the x/v-part of the plane
    estimator per estimator variant against the replica-exact truth, pooled over seeds
    (inverse-variance weights on the per-seed sums)."""
    path = os.path.join(RESULTS, f"plane_fit_variants{suffix}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = json.load(fh)

    def pooled(key):
        m = np.array([r[key][0] for r in rows])
        se = np.array([r[key][1] for r in rows])
        wgt = 1.0 / np.maximum(se, 1e-300) ** 2
        return float(np.sum(wgt * m) / wgt.sum()), float(1.0 / math.sqrt(wgt.sum()))

    out = {"seeds": len(rows), "true": pooled("x_part_true")}
    for v in VARIANT_NAMES:
        if f"x_part_{v}" in rows[0]:
            out[v] = {
                "x_part": pooled(f"x_part_{v}"),
                "bias": pooled(f"bias_{v}"),
                "rms_per_hit": float(np.mean([r[f"rms_{v}"] for r in rows])),
                "se_per_seed": float(np.mean([r[f"x_part_{v}"][1] for r in rows])),
            }
    ident = [r["identity"] for r in rows if "identity" in r]
    if ident:
        out["identity"] = {
            "bias": float(np.mean([i["bias"] for i in ident])),
            "bias_se": float(np.sqrt(np.sum([i["bias_se"] ** 2 for i in ident])) / len(ident)),
            "rms_diff": float(np.mean([i["rms_diff"] for i in ident])),
            "scale": float(np.mean([i["scale"] for i in ident])),
        }
    return out


def variants_runs(gates):
    """Per-variant statistics of the production artifacts that carry ``rows[*].variants``:
    mean ± SE and per-seed sd of the gradient, z against the reference (truncated FD
    for the plane, CRN FD for the circle)."""
    out = {}
    for name, st in gates.items():
        path = os.path.join(GRADIENT_RESULTS, name)
        with open(path) as fh:
            rows = json.load(fh)["rows"]
        if not rows or "variants" not in rows[0]:
            continue
        ref = st.get("truncated_fd") or st["fd"][str(st["reference_h"])]
        entry = {"label": st["label"], "seeds": len(rows), "reference": ref}
        for v in rows[0]["variants"]:
            g = np.array([r["variants"][v]["gradient"] for r in rows])
            gx = np.array([r["variants"][v]["pathwise_x"] for r in rows])
            se = g.std(ddof=1) / math.sqrt(len(g))
            entry[v] = {
                "mean": float(g.mean()),
                "se": float(se),
                "sd": float(g.std(ddof=1)),
                "x_part": float(gx.mean()),
                "x_part_se": float(gx.std(ddof=1) / math.sqrt(len(g))),
                "z": float((g.mean() - ref["mean"]) / math.sqrt(se**2 + ref["se"] ** 2)),
            }
        out[name] = entry
    return out


def figure_variants(probe, runs, path_fig):  # noqa: PLR0915 -- two rows, one figure
    """Top row: the x/v-part of each estimator against the replica-exact gradient (probe)
    and the total gradient of the production runs against their reference FD; bottom row:
    the x-part alone on the production runs (the only quantity the variants change), with
    its SE and the per-seed sd."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    panels = [(k, p) for k, p in probe.items() if p is not None]
    n_cols = max(len(panels), len(runs))
    if n_cols == 0:
        return
    fig, axes = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 7.0), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for col, (label, pr) in enumerate(panels):
        ax = axes[0][col]
        ax.set_visible(True)
        names = [v for v in VARIANT_NAMES if v in pr]
        xs = np.arange(len(names))
        ax.bar(
            xs,
            [pr[v]["x_part"][0] for v in names],
            yerr=[pr[v]["x_part"][1] for v in names],
            color=["0.6" if v == "two" else "tab:blue" for v in names],
            capsize=3,
        )
        lo, hi = pr["true"][0] - pr["true"][1], pr["true"][0] + pr["true"][1]
        ax.axhspan(lo, hi, color="tab:red", alpha=0.25)
        ax.axhline(pr["true"][0], color="tab:red", label="replica-exact ± SE")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=25, fontsize=8)
        ax.set_title(f"x/v-part vs exact gradient\n{label}, {pr['seeds']} seeds", fontsize=8)
        ax.legend(fontsize=7)
    for col, run in enumerate(runs.values()):
        names = [v for v in VARIANT_NAMES if v in run]
        xs = np.arange(len(names))
        ref = run["reference"]
        ax = axes[1][col]
        ax.set_visible(True)
        ax.bar(
            xs,
            [run[v]["mean"] for v in names],
            yerr=[run[v]["se"] for v in names],
            color="tab:blue",
            capsize=3,
        )
        ax.errorbar(
            xs,
            [run[v]["mean"] for v in names],
            yerr=[run[v]["sd"] for v in names],
            fmt="none",
            ecolor="0.5",
            alpha=0.7,
            label="per-seed sd",
        )
        ax.axhspan(ref["mean"] - ref["se"], ref["mean"] + ref["se"], color="tab:red", alpha=0.25)
        ax.axhline(ref["mean"], color="tab:red", label="reference FD ± SE")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=25, fontsize=8)
        ax.set_title(f"total gradient: {run['label']}\n{run['seeds']} seeds", fontsize=8)
        ax.legend(fontsize=7)
        # the x-part alone, inset as a second axis to the right of the bars
        ax2 = ax.twinx()
        ax2.errorbar(
            xs + 0.35,
            [run[v]["x_part"] for v in names],
            yerr=[run[v]["x_part_se"] for v in names],
            fmt="s",
            color="tab:green",
            capsize=2,
            label="x-part ± SE (right axis)",
        )
        ax2.tick_params(axis="y", colors="tab:green", labelsize=7)
        ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)


def print_variants(probe, runs):
    for label, pr in probe.items():
        if pr is None:
            continue
        print(
            f"\n### Estimators of grad_x lambda vs the replica-exact gradient ({label}, {pr['seeds']} seeds)\n"
        )
        print(f"replica-exact x/v-part: {pr['true'][0]:+.4f} ± {pr['true'][1]:.4f}")
        print("| variant | x/v-part ± SE | bias vs exact | per-seed SE (variance) | rms per hit |")
        print("|---|---|---|---|---|")
        for v in VARIANT_NAMES:
            if v in pr:
                e = pr[v]
                print(
                    f"| {VARIANT_LABELS[v]} | {e['x_part'][0]:+.4f} ± {e['x_part'][1]:.4f} | {e['bias'][0]:+.4f} ± {e['bias'][1]:.4f} | {e['se_per_seed']:.4f} | {e['rms_per_hit']:.2e} |"
                )
        if "identity" in pr:
            i = pr["identity"]
            print(
                f"along-flight identity per particle: (-T) - v.grad V = {i['bias']:+.2e} ± {i['bias_se']:.1e} (rms diff {i['rms_diff']:.2e}, scale of v.grad V {i['scale']:.2e})"
            )
    if runs:
        print(
            "\n### Estimator variants on the production runs (same seeds; reference = truncated FD for the plane, CRN FD for the circle)\n"
        )
        print(
            "| case | seeds | variant | gradient ± SE | per-seed sd | x-part ± SE | z vs reference |"
        )
        print("|---|---|---|---|---|---|---|")
        for run in runs.values():
            for v in VARIANT_NAMES:
                if v in run:
                    e = run[v]
                    print(
                        f"| {run['label']} | {run['seeds']} | {VARIANT_LABELS[v]} | {e['mean']:+.4g} ± {e['se']:.2g} | {e['sd']:.2g} | {e['x_part']:+.3g} ± {e['x_part_se']:.2g} | {e['z']:+.2f} |"
                    )


def print_adjoint_gates(gates):
    print("\n### Adjoint gates G1-G3 and the shape cases (ensemble adjoint vs CRN FD)\n")
    print(
        "| gate | case | seeds | J | ensemble ± SE | FD ± SE (h) | analytic / truncated FD | z | pass |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for st in gates.values():
        h = str(st["reference_h"])
        f = st["fd"][h]
        tfd = _truncated_fd(st)
        if tfd is not None:
            st["truncated_fd"] = tfd
            st["z_truncated"] = (st["ensemble"] - tfd["mean"]) / math.sqrt(
                st["ensemble_se"] ** 2 + tfd["se"] ** 2
            )
            # the truncated adjoint is compared with the derivative it is defined to estimate
            st["pass"] = bool(abs(st["z_truncated"]) <= 2.0)
        terms = (
            f" (pathwise {st['pathwise']:+.4g}, score {st['score']:+.4g}, timing {st['timing']:+.4g}"
            + (f", silhouette {st['silhouette_mean']:+.4g}" if "silhouette_mean" in st else "")
            + ")"
            if "pathwise" in st
            else ""
        )
        analytic = f"{st['analytic']:+.4g} (F/r)" if "analytic" in st else ""
        if tfd is not None:
            analytic = (
                f"truncated FD {tfd['mean']:+.4g} ± {tfd['se']:.2g} ({tfd['seeds']} seeds), "
                f"z = {st['z_truncated']:+.2f}"
            )
        print(
            f"| {st['gate']} | {st['label']} | {st['seeds']} | {st['objective']:.4g} | "
            f"{st['ensemble']:+.4g} ± {st['ensemble_se']:.2g}{terms} | {f['mean']:+.4g} ± {f['se']:.2g} ({h}) | "
            f"{analytic} | {st['z']:+.2f} | {st['pass']} |"
        )
    for suffix, label in (("", "shear"), ("_energy", "v_y^2")):
        fb = _fit_bias(suffix)
        if fb is not None:
            print(
                f"\nFit bias on the plane (replica-exact value gradients, {fb['seeds']} seeds, "
                f"x/v-part of the windowed {label} estimator): "
                f"true {fb['x_part_true'][0]:+.4f} ± {fb['x_part_true'][1]:.4f}; "
                f"one-sided fit {fb['x_part_fit_onesided'][0]:+.4f} ± "
                f"{fb['x_part_fit_onesided'][1]:.4f} "
                f"(bias {fb['bias_onesided'][0]:+.4f} ± {fb['bias_onesided'][1]:.4f}); "
                f"two-sided fit {fb['x_part_fit_twosided'][0]:+.4f} ± "
                f"{fb['x_part_fit_twosided'][1]:.4f} "
                f"(bias {fb['bias_twosided'][0]:+.4f} ± {fb['bias_twosided'][1]:.4f})"
            )


def _fmt(x, nd=3):
    return f"{x:.{nd}g}" if isinstance(x, float) else str(x)


def main() -> int:  # noqa: PLR0912, PLR0915 -- one report pass, kept whole
    os.makedirs(FIGURES, exist_ok=True)
    report = {}
    for case in ("circle", "sphere"):
        runs = load_ours(case)
        if not runs:
            print(f"{case}: no forward artifacts yet")
            continue
        logs = load_sparta_logs(case)
        seeds = sorted(logs)
        grid = grid_for(case)
        g = {
            "runs": len(runs),
            "seeds": [r["seed"] for r in runs],
            "seconds_per_run": float(np.mean([r["seconds"] for r in runs])),
            "backend": runs[0]["backend"],
        }
        g["FG1_diagnostics"] = gate_diagnostics(runs)
        g["FG2_fill"], curves = gate_fill(case, runs, logs)
        air = load_sparta_logs(case, "air")
        figure_fill(
            case,
            curves,
            next(iter(air.values())) if air else None,
            figure_path(forward_folder(case), f"{case}_fill.png"),
        )
        sp = sparta_surface(case, seeds)
        g["FG3_surface"], stats = gate_surface(case, runs, sp)
        rate = g["FG3_surface"]["hits_per_step_ours"] / g["FG3_surface"]["hits_per_step_sparta"] - 1
        g["FG2_fill"]["hit_rate_rel_diff"] = rate
        g["FG2_fill"]["pass"] = bool(g["FG2_fill"]["pass"] and abs(rate) <= 0.02)
        figure_surface(
            case, runs, sp, stats, figure_path(forward_folder(case), f"{case}_surface_pressure.png")
        )
        fields = gate_fields(case, runs, sparta_fields(case, seeds, grid))
        g["FG4_fields"] = {
            k: {kk: vv for kk, vv in v.items() if kk not in ("ours", "sparta", "z")}
            if isinstance(v, dict)
            else v
            for k, v in fields.items()
        }
        from jax_dsmc.mesh import surface as S  # noqa: PLC0415

        surf = S.read_sparta_surf(os.path.join(SPARTA, CASES[case]["surf"]), CASES[case]["dim"])
        overlay = (np.asarray(surf.points), np.asarray(surf.elements))
        mask = cut.flow_mask(
            cut.build_mesh(surf, grid, sub=8), grid, sub=8 if CASES[case]["dim"] == 2 else 4
        )
        figure_fields(
            case,
            grid,
            fields,
            overlay,
            figure_path(forward_folder(case), f"{case}_fields_sparta_t1000.png"),
            figure_path(forward_folder(case), f"{case}_fields_difference.png"),
            mask=mask,
        )
        g["FG7_polygon_vs_analytic"] = compare_variants(
            case, runs, load_ours(case, body="analytic"), "analytic body"
        )
        g["tracker_walk_vs_stencil"] = compare_variants(
            case, runs, load_ours(case, tracker="walk"), "cell walk"
        )
        report[case] = g
        _print_case(case, g)
    ws = window_score(
        os.path.join(RESULTS, "window_score.json"), figure_path("gates", "window_score.png")
    )
    if ws is not None:
        report["G3a_window_score"] = ws
        print("\n### G3a window score (1-D closed box, windowed impulse; 12 seeds)\n")
        print(
            "| W | FD ± SE (sd) | analytic | eps-score mean ± SE, eps = 0.02 | eps = 0.01 | score sd / FD sd |"
        )
        print("|---|---|---|---|---|---|")
        for w, r in ws.items():
            s02, s01 = r["score"]["0.02"], r["score"]["0.01"]
            print(
                f"| {w} | {r['fd'][0]:+.4f} ± {r['fd'][1]:.4f} ({r['sd_fd']:.4f}) | {r['analytic']:+.4f} | {s02[0]:+.4f} ± {s02[1]:.4f} (z {r['score_z']['0.02']:+.1f}) | {s01[0]:+.4f} ± {s01[1]:.4f} (z {r['score_z']['0.01']:+.1f}) | {r['score_sd_over_fd_sd']['0.02']:.1f} / {r['score_sd_over_fd_sd']['0.01']:.1f} |"
            )
    ag = adjoint_gates(figure_path("gates", "adjoint_gates.png"))
    if ag:
        report["adjoint_gates"] = ag
        print_adjoint_gates(ag)
        figure_shape_terms(ag, figure_path("gates", "shape_circle_terms.png"))
        ls = levelset_vs_analytic(figure_path("gates", "levelset_gradient.png"))
        if ls is not None:
            report["levelset_vs_analytic"] = ls
            print_levelset(ls)
        gz = grazing_regularisation(figure_path("gates", "grazing_regularisation.png"))
        if gz is not None:
            report["grazing_regularisation"] = gz
            print_grazing(gz)
        sf = score_floor_comparison()
        if sf is not None:
            report["score_floor_comparison"] = sf
            print_score_floor(sf)
        fs = plate_fd_steps()
        if fs is not None:
            report["plate_fd_steps"] = fs
            print_plate_fd_steps(fs)
        if sf is not None and fs is not None:
            figure_plate_estimators(sf, fs, figure_path("gates", "plate_estimators.png"))
        for suffix in ("", "_energy"):
            fb = _fit_bias(suffix)
            if fb is not None:
                report[f"plane_fit_bias{suffix}"] = fb
        fc = fit_classes(figure_path("gates", "fit_classes.png"))
        if fc is not None:
            report["plane_fit_classes"] = fc
            print_fit_classes(fc)
        probe = {"windowed shear": variants_probe(""), "windowed v_y^2": variants_probe("_energy")}
        runs = variants_runs(ag)
        print_variants(probe, runs)
        report["estimator_variants"] = {"probe": probe, "runs": runs}
        figure_variants(probe, runs, figure_path("gates", "estimator_variants.png"))
    with open(os.path.join(RESULTS, "forward_gates.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", os.path.join(RESULTS, "forward_gates.json"))
    return 0


def _print_case(case, g):
    f2, f3, f4 = g["FG2_fill"], g["FG3_surface"], g["FG4_fields"]
    print(f"\n### {case} ({g['runs']} runs, {g['seconds_per_run']:.0f} s/run on {g['backend']})\n")
    print("| gate | measurement | ours | SPARTA | z / criterion | pass |")
    print("|---|---|---|---|---|---|")
    d = g["FG1_diagnostics"]["totals"]
    print(
        f"| FG1 | diagnostics (sum over runs) | {', '.join(f'{k}={_fmt(v)}' for k, v in d.items())} | 0 | all zero | {g['FG1_diagnostics']['pass']} |"
    )
    print(
        f"| FG2 | final Np | {f2['np']['ours_final']:.0f} ± {f2['np']['ours_se_final']:.0f} | {f2['np']['sparta_final']:.0f} ± {f2['np']['sparta_se_final']:.0f} | z = {f2['np']['z_final']:.2f}, z(tail mean) = {f2['np']['tail_z']:.2f}, max\\|z\\| last 300 = {f2['np']['max_abs_z_tail']:.2f} | {f2['pass']} |"
    )
    print(
        f"| FG2 | Nscoll/step, last 300 | {f2['nscoll']['ours_tail_mean']:.1f} | {f2['nscoll']['sparta_tail_mean']:.1f} | z(tail mean) = {f2['nscoll']['tail_z']:.2f}, max\\|z\\| = {f2['nscoll']['max_abs_z_tail']:.2f} | |"
    )
    print(
        f"| FG3 | total drag fx [N{'/m' if case == 'circle' else ''}] | {f3['fx_ours']:.4g} ± {f3['fx_ours_se']:.2g} | {f3['fx_sparta']:.4g} ± {f3['fx_sparta_se']:.2g} | z = {f3['fx_z']:.2f} ({100 * f3['fx_rel_diff']:+.2f} %) | {f3['pass']} |"
    )
    print(
        f"| FG3 | per-element pressure | rms z {f3['press_rms_z']:.2f}, max {f3['press_max_abs_z']:.2f}, {100 * f3['press_frac_within_2']:.0f} % within 2 | | hits/step {f3['hits_per_step_ours']:.1f} vs {f3['hits_per_step_sparta']:.1f} | |"
    )
    for k in ("nrho", "u", "temp"):
        v = f4[k]
        print(
            f"| FG4 | {k} per cell ({v['cells']} cells) | mean {v['mean_ours']:.4g} | mean {v['mean_sparta']:.4g} | rms z {v['rms_z']:.2f}, {100 * v['frac_within_2']:.0f} % within 2 | {f4['pass']} |"
        )
    for key, label in (
        ("FG7_polygon_vs_analytic", "FG7 polygon vs analytic"),
        ("tracker_walk_vs_stencil", "walk vs stencil"),
    ):
        c = g[key]
        if c["available"]:
            print(
                f"| {label} | Np / Nscoll / fx | {c['np_final']['base']:.0f} / {c['nscoll_tail']['base']:.1f} / {c['fx']['base']:.4g} | {c['np_final']['other']:.0f} / {c['nscoll_tail']['other']:.1f} / {c['fx']['other']:.4g} | rel {100 * c['np_final']['rel_diff']:+.2f} / {100 * c['nscoll_tail']['rel_diff']:+.1f} / {100 * c['fx']['rel_diff']:+.2f} % | |"
            )


if __name__ == "__main__":
    sys.exit(main())
