"""Geometric verification of the cut-cell tables against SPARTA -- no particles.

Runs the checks of the plan's section 1a on SPARTA's shipped ``circle`` (2-D) and ``sphere``
(3-D) surfaces: cell-type counts, the cut-cell set, total and per-cell flow volumes against
the ``compute property/grid vol`` dumps of the rebuilt argon runs, candidate lists against a
brute-force clip, watertightness, sub-sample convergence.  Writes
``results/geometry_checks.json`` and the figures ``mesh_cells_circle.png``,
``mesh_cells_sphere_slice.png``, ``mesh_volume_error.png``, ``mesh_volume_convergence.png``.

    JAX_PLATFORMS=cpu python3 examples/mesh/geometry_checks.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import jax.numpy as jnp
import numpy as np
from matplotlib.colors import ListedColormap

from jax_dsmc import viz
from jax_dsmc.grid import cell_of, cell_volume
from jax_dsmc.mesh import analytic, cut
from jax_dsmc.mesh import levelset as L
from jax_dsmc.mesh import surface as S
from jax_dsmc.mesh.types import CUT, INSIDE, OUTSIDE, AnalyticBody
from jax_dsmc.state import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sparta_io import read_grid_dump  # noqa: E402

SPARTA = os.path.join(HERE, "sparta")
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

# SPARTA log values (examples/mesh/results/sparta/log.*_ar.12345): outside / inside / overlap
SPARTA_COUNTS = {"circle": (264, 88, 48), "sphere": (7256, 304, 440)}
SPARTA_FLOW_VOLUME = {"circle": 71.8, "sphere": 59.8617}
TYPE_CMAP = ListedColormap([viz.SERIES_ANALYTIC, viz.INK_SECONDARY, viz.SERIES_MEASURED])


def grid_for(case: str) -> GridSpec:
    if case == "circle":
        return GridSpec(lo=jnp.array([0.0, 0.0]), hi=jnp.array([10.0, 10.0]), n=(20, 20))
    return GridSpec(lo=jnp.array([-2.0] * 3), hi=jnp.array([2.0] * 3), n=(20, 20, 20))


def sparta_volumes(case: str, grid: GridSpec, seed: int = 12345) -> np.ndarray:
    d = read_grid_dump(os.path.join(RESULTS, "sparta", f"grid_vol.{case}_ar.{seed}.txt"), 0)
    D = len(grid.n)
    cols = [d["xc"], d["yc"]] + ([d["zc"]] if D == 3 else [])
    cid = np.asarray(cell_of(jnp.asarray(np.stack(cols, axis=1)), grid))
    ref = np.zeros(int(np.prod(grid.n)))
    np.add.at(ref, cid, d["vol"])
    return ref


def counts(cell_type) -> tuple[int, int, int]:
    ct = np.asarray(cell_type)
    return tuple(int((ct == k).sum()) for k in (OUTSIDE, INSIDE, CUT))


def check_case(case: str, sub: int | None) -> dict:
    grid = grid_for(case)
    D = len(grid.n)
    surf = S.read_sparta_surf(os.path.join(SPARTA, f"data.{case}"), D)
    S.check_watertight(surf)
    t0 = time.time()
    mesh = cut.build_mesh(surf, grid, sub=sub)
    t_build = time.time() - t0
    ref = sparta_volumes(case, grid)
    ours = np.asarray(mesh.volume)
    ct = np.asarray(mesh.cell_type)
    full = float(cell_volume(grid)[0])
    partial = (ref > 1e-12) & (ref < full - 1e-12)
    err = ours - ref
    exact = cut.exact_flow_volume(surf, grid)
    out = {
        "case": case,
        "dim": D,
        "n_points": int(surf.points.shape[0]),
        "n_elements": int(surf.elements.shape[0]),
        "sub": int(sub or cut.DEFAULT_SUB[D]),
        "build_seconds": t_build,
        "counts": counts(ct),
        "counts_sparta": SPARTA_COUNTS[case],
        "counts_match": counts(ct) == SPARTA_COUNTS[case],
        "cut_set_matches_sparta_partial": bool(
            partial[ct == CUT].sum() == partial.sum() and not partial[ct != CUT].any()
        ),
        "n_partial_sparta": int(partial.sum()),
        "n_touched_full": int(((ct == CUT) & ~partial).sum()),
        "flow_volume_exact": exact,
        "flow_volume_sparta_dump": float(ref.sum()),
        "flow_volume_sparta_log": SPARTA_FLOW_VOLUME[case],
        "flow_volume_ours": float(ours.sum()),
        "total_error": float(err.sum()),
        "max_cell_error": float(np.abs(err).max()),
        "max_cell_error_uncut": float(np.abs(err[ct != CUT]).max()),
        "cell_volume": full,
        "S_cell": int(mesh.csurfs.shape[1]),
        "S_max": int(mesh.candidates.shape[1]),
        "body_volume_polygon": float(S.body_volume(surf)),
        "body_volume_analytic": math.pi * 9.0 if D == 2 else 4.0 / 3.0 * math.pi,
    }
    # the analytic body of the same radius: counts and volume
    centre = jnp.array([5.0, 5.0]) if D == 2 else jnp.zeros(3)
    r = 3.0 if D == 2 else 1.0
    body = AnalyticBody(centre=centre, radius=jnp.float64(r))
    mb = cut.build_mesh(body, grid, sub=sub)
    out["analytic_counts"] = counts(mb.cell_type)
    out["analytic_flow_volume"] = float(mb.volume.sum())
    out["analytic_flow_volume_exact"] = cut.exact_flow_volume(body, grid)
    return out, mesh, ref


def candidate_check(case: str, mesh, grid: GridSpec) -> dict:
    """Brute-force clip vs `csurfs` (2-D exact; 3-D bounding-box superset only)."""
    cs = np.asarray(mesh.csurfs)
    lo, hi = cut.cell_boxes(grid)
    p = np.asarray(mesh.body.points)[np.asarray(mesh.body.elements)]
    mismatched = 0
    if case == "circle":
        for c in range(lo.shape[0]):
            listed = {int(e) for e in cs[c][cs[c] >= 0]}
            brute = {
                e for e in range(p.shape[0]) if _segment_meets_box(p[e, 0], p[e, 1], lo[c], hi[c])
            }
            mismatched += listed != brute
    st = np.asarray(mesh.candidates)
    superset = all(
        {int(e) for e in cs[c][cs[c] >= 0]} <= {int(e) for e in st[c][st[c] >= 0]}
        for c in range(lo.shape[0])
    )
    return {"brute_force_mismatched_cells": mismatched, "stencil_is_superset": bool(superset)}


def _segment_meets_box(a, b, lo, hi) -> bool:
    t0, t1 = 0.0, 1.0
    d = b - a
    for k in range(a.size):
        for sign, bound in ((-1.0, lo[k]), (1.0, hi[k])):
            p = sign * d[k]
            q = sign * (bound - a[k])
            if p == 0.0:
                if q < 0.0:
                    return False
            else:
                t = q / p
                t0, t1 = (max(t0, t), t1) if p < 0.0 else (t0, min(t1, t))
    return t0 <= t1


def watertight_check(surf, grid: GridSpec, n: int = 4000, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    lo = np.asarray(grid.lo)
    hi = np.asarray(grid.hi)
    p = lo + rng.random((n, lo.size)) * (hi - lo)
    q = lo + rng.random((n, lo.size)) * (hi - lo)
    out = cut._global_outside(surf, p) & cut._global_outside(surf, q)
    k = cut.crossings(surf, p[out], q[out])
    return {
        "segments": int(out.sum()),
        "even_crossings": bool((k % 2 == 0).all()),
        "max": int(k.max()),
    }


def convergence(case: str, grid: GridSpec, ref: np.ndarray, subs) -> dict:
    surf = S.read_sparta_surf(os.path.join(SPARTA, f"data.{case}"), len(grid.n))
    cs = cut.overlaps(surf, grid)
    ct = cut.classify(surf, grid, cs)
    oc = cut.outside_corner(surf, grid, ct, cs)
    full = float(cell_volume(grid)[0])
    rows = []
    for sub in subs:
        fr = cut.volume_fraction(surf, grid, ct, cs, oc, sub=sub)
        err = fr * full - ref
        rows.append(
            {
                "sub": sub,
                "max_cell_error": float(np.abs(err).max()),
                "total_error": float(err.sum()),
                "rms_cell_error": float(np.sqrt(np.mean(err[ct == CUT] ** 2))),
            }
        )
    return rows


def figure_cells_2d(mesh, grid: GridSpec, ref: np.ndarray, path: str) -> None:
    fig, (ax0, ax1) = viz.make_figure(1, 2, figsize=(11.2, 5.0))
    n = tuple(int(x) for x in grid.n)
    im = viz.draw_field(
        ax0,
        np.asarray(mesh.cell_type),
        n,
        grid.lo,
        grid.hi,
        label="cell type (0 outside, 1 inside, 2 cut)",
        cmap=TYPE_CMAP,
        vmin=-0.5,
        vmax=2.5,
    )
    fig.colorbar(im, ax=ax0, ticks=[0, 1, 2])
    viz.draw_surface(ax0, mesh.body.points, mesh.body.elements)
    full = float(cell_volume(grid)[0])
    im = viz.draw_field(
        ax1,
        np.asarray(mesh.volume) / full,
        n,
        grid.lo,
        grid.hi,
        label="flow volume fraction",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(im, ax=ax1)
    viz.draw_surface(ax1, mesh.body.points, mesh.body.elements)
    viz.savefig(fig, path)
    # the same fraction with the body carved out at sub-cell resolution: what the tracker
    # and the flow region actually look like (`cut.flow_mask`)
    fig, (ax,) = viz.make_figure(1, 1, figsize=(5.6, 5.0))
    mask = cut.flow_mask(mesh, grid, sub=16)
    im = viz.draw_field(
        ax,
        np.asarray(mesh.volume) / full,
        n,
        grid.lo,
        grid.hi,
        label="flow volume fraction, body carved out (sub 16)",
        vmin=0.0,
        vmax=1.0,
        mask=mask,
    )
    fig.colorbar(im, ax=ax)
    viz.draw_surface(ax, mesh.body.points, mesh.body.elements)
    viz.savefig(fig, path.replace(".png", "_clipped.png"))


def figure_cells_3d_slice(mesh, grid: GridSpec, path: str) -> None:
    fig, (ax0, ax1) = viz.make_figure(1, 2, figsize=(11.2, 5.0))
    n = tuple(int(x) for x in grid.n)
    full = float(cell_volume(grid)[0])
    pl = viz.Plane(2, fraction=0.5)  # z = 0 (cells [10] start at z = 0)
    im = viz.draw_field(
        ax0,
        np.asarray(mesh.cell_type),
        n,
        grid.lo,
        grid.hi,
        label="cell type",
        cmap=TYPE_CMAP,
        vmin=-0.5,
        vmax=2.5,
        plane=pl,
    )
    fig.colorbar(im, ax=ax0, ticks=[0, 1, 2])
    pl_draw = viz.Plane(2, fraction=0.525)  # the slice's cell centres are at z = 0.1
    viz.draw_surface(
        ax0, mesh.body.points, mesh.body.elements, plane=pl_draw, lo=grid.lo, hi=grid.hi
    )
    im = viz.draw_field(
        ax1,
        np.asarray(mesh.volume) / full,
        n,
        grid.lo,
        grid.hi,
        label="flow volume fraction",
        vmin=0.0,
        vmax=1.0,
        plane=pl,
    )
    fig.colorbar(im, ax=ax1)
    viz.draw_surface(
        ax1, mesh.body.points, mesh.body.elements, plane=pl_draw, lo=grid.lo, hi=grid.hi
    )
    viz.savefig(fig, path)


def figure_volume_error(meshes, grids, refs, path: str) -> None:
    fig, axes = viz.make_figure(1, 2, figsize=(11.2, 5.0))
    for ax, case in zip(axes, ("circle", "sphere"), strict=True):
        mesh, grid, ref = meshes[case], grids[case], refs[case]
        n = tuple(int(x) for x in grid.n)
        full = float(cell_volume(grid)[0])
        err = (np.asarray(mesh.volume) - ref) / full
        vmax = float(np.abs(err).max()) or 1e-3
        plane = viz.Plane(2, fraction=0.5) if len(n) == 3 else None
        im = viz.draw_field(
            ax,
            err,
            n,
            grid.lo,
            grid.hi,
            label=f"{case}: (ours - SPARTA) / cell volume",
            cmap=viz.FIELD_CMAP_DIVERGING,
            vmin=-vmax,
            vmax=vmax,
            plane=plane,
        )
        fig.colorbar(im, ax=ax)
    viz.savefig(fig, path)


def figure_convergence(conv: dict, path: str) -> None:
    fig, (ax,) = viz.make_figure(1, 1)
    for case, color in (("circle", viz.SERIES_ANALYTIC), ("sphere", viz.SERIES_MEASURED)):
        rows = conv[case]
        subs = [r["sub"] for r in rows]
        ax.loglog(
            subs,
            [r["max_cell_error"] for r in rows],
            "o-",
            color=color,
            label=f"{case}: max cell error",
        )
        ax.loglog(
            subs,
            [r["rms_cell_error"] for r in rows],
            "s--",
            color=color,
            label=f"{case}: rms cut-cell error",
        )
    s = np.array([4.0, 64.0])
    ax.loglog(s, 0.05 * (s / 4.0) ** -1.5, color=viz.REFERENCE, lw=1, label="sub^-1.5")
    ax.set_xlabel("sub-grid resolution per axis")
    ax.set_ylabel("flow-volume error [cell volume units]")
    ax.legend(fontsize=8)
    viz.savefig(fig, path)


ANALYTIC = {
    "circle": AnalyticBody(centre=jnp.array([5.0, 5.0]), radius=jnp.float64(3.0)),
    "sphere": AnalyticBody(centre=jnp.zeros(3), radius=jnp.float64(1.0)),
}


def levelset_check(case: str, grid: GridSpec, hs, seed: int = 7, n: int = 40000) -> dict:
    """LS0 (plan phase E): the level set interpolating the circle / sphere at knot spacing
    ``h`` vs the analytic body -- surface error (``|phi|`` at analytic surface points), hit
    position error on random rays, and the tunnelling rate of a grazing family (impact
    parameter ``r - delta``) as a function of the chord inside the body over the sample
    spacing ``h / 2``.  All numbers are documented, the surface error and hit error must
    fall as ``h^4``."""
    body = ANALYTIC[case]
    c, r = np.asarray(body.centre), float(body.radius)
    dim = c.size
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    x = lo + rng.uniform(size=(n, dim)) * (hi - lo)
    x = x[np.linalg.norm(x - c, axis=1) > r + 0.05]
    v = rng.normal(size=(x.shape[0], 3)) * 400.0
    if dim == 2:
        v[:, 2] = 0.0
    xj, vj = jnp.asarray(x), jnp.asarray(v)
    t_an = np.asarray(analytic.hit_body(xj, vj, body, jnp.zeros(x.shape[0], bool)))
    speed = np.linalg.norm(v[:, :dim], axis=1)
    ang = rng.uniform(size=(2000, dim - 1)) * np.pi * np.array([2.0, 1.0][: dim - 1])
    if dim == 2:
        on = c + r * np.stack([np.cos(ang[:, 0]), np.sin(ang[:, 0])], axis=1)
    else:
        on = c + r * np.stack(
            [
                np.sin(ang[:, 1]) * np.cos(ang[:, 0]),
                np.sin(ang[:, 1]) * np.sin(ang[:, 0]),
                np.cos(ang[:, 1]),
            ],
            axis=1,
        )
    out = {"h": [], "surface_error": [], "hit_error": [], "tunnelling": []}
    deltas = np.geomspace(1e-6, 1e-1, 300)
    for h in hs:
        origin, shape = L.grid_for_box(grid.lo, grid.hi, h)
        ls = L.from_analytic(body.centre, r, origin, h, shape, n_sample=16)
        surf = float(np.abs(np.asarray(L.value(ls, jnp.asarray(on)))).max())
        t_ls = np.asarray(L.hit_body(xj, vj, ls))
        both = np.isfinite(t_an) & np.isfinite(t_ls) & (t_an * speed <= ls.reach)
        hit_err = float((np.abs(t_ls[both] - t_an[both]) * speed[both]).max())
        # grazing family along +x at height c_y + r - delta (z = c_z), each ray started
        # 0.05 before its analytic entry point so the search reach covers the chord
        xg = np.tile(c, (deltas.size, 1))
        xg[:, 0] = c[0] - np.sqrt(2.0 * r * deltas - deltas * deltas) - 0.05
        xg[:, 1] = c[1] + r - deltas
        vg = np.zeros((deltas.size, 3))
        vg[:, 0] = 500.0
        tg = np.asarray(L.hit_body(jnp.asarray(xg), jnp.asarray(vg), ls))
        chord = 2.0 * np.sqrt(2.0 * r * deltas)
        out["h"].append(float(h))
        out["surface_error"].append(surf)
        out["hit_error"].append(hit_err)
        out["tunnelling"].append(
            {
                "chord_over_spacing": (chord / (h / 2.0)).tolist(),
                "found": np.isfinite(tg).astype(int).tolist(),
            }
        )
    return out


def figure_levelset(checks: dict, path: str) -> None:
    fig, axes = viz.make_figure(1, 2, figsize=(8.4, 3.4))
    ax = axes[0]
    for case, chk in checks.items():
        h = np.asarray(chk["h"])
        ax.loglog(h, chk["surface_error"], "o-", label=f"{case}: surface |phi| error")
        ax.loglog(h, chk["hit_error"], "s--", label=f"{case}: hit position error")
    h = np.asarray(next(iter(checks.values()))["h"])
    ax.loglog(h, 1e-3 * (h / h[0]) ** 4, color=viz.REFERENCE, lw=1, label="h^4")
    ax.set_xlabel("knot spacing h")
    ax.set_ylabel("error")
    ax.legend(fontsize=7)
    ax = axes[1]
    for case, chk in checks.items():
        for hh, tun in zip(chk["h"], chk["tunnelling"], strict=True):
            cs = np.asarray(tun["chord_over_spacing"])
            found = np.asarray(tun["found"], dtype=float)
            edges = np.geomspace(cs.min(), cs.max(), 25)
            idx = np.clip(np.digitize(cs, edges) - 1, 0, edges.size - 2)
            rate = np.array(
                [
                    1.0 - found[idx == b].mean() if (idx == b).any() else np.nan
                    for b in range(edges.size - 1)
                ]
            )
            ax.semilogx(np.sqrt(edges[:-1] * edges[1:]), rate, "-", label=f"{case}, h = {hh:g}")
    ax.axvline(1.0, color=viz.REFERENCE, lw=1)
    ax.set_xlabel("chord inside the body / sample spacing (h / 2)")
    ax.set_ylabel("tunnelling rate of the grazing family")
    ax.legend(fontsize=7)
    viz.savefig(fig, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sub-2d", type=int, default=None)
    ap.add_argument("--sub-3d", type=int, default=None)
    args = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    report = {}
    meshes, grids, refs = {}, {}, {}
    for case, sub in (("circle", args.sub_2d), ("sphere", args.sub_3d)):
        out, mesh, ref = check_case(case, sub)
        grid = grid_for(case)
        out.update(candidate_check(case, mesh, grid))
        out["watertight"] = watertight_check(mesh.body, grid)
        report[case] = out
        meshes[case], grids[case], refs[case] = mesh, grid, ref
        print(
            f"{case}: counts {out['counts']} (SPARTA {out['counts_sparta']}), "
            f"flow volume {out['flow_volume_ours']:.5f} (dump {out['flow_volume_sparta_dump']:.5f}, "
            f"exact {out['flow_volume_exact']:.5f}), max cell error {out['max_cell_error']:.2e}, "
            f"build {out['build_seconds']:.1f} s, brute-force mismatches "
            f"{out['brute_force_mismatched_cells']}, watertight {out['watertight']['even_crossings']}"
        )
    report["convergence"] = {
        "circle": convergence("circle", grids["circle"], refs["circle"], (4, 8, 16, 32, 64)),
        "sphere": convergence("sphere", grids["sphere"], refs["sphere"], (4, 8, 16)),
    }
    figure_cells_2d(
        meshes["circle"],
        grids["circle"],
        refs["circle"],
        os.path.join(FIGURES, "mesh_cells_circle.png"),
    )
    figure_cells_3d_slice(
        meshes["sphere"], grids["sphere"], os.path.join(FIGURES, "mesh_cells_sphere_slice.png")
    )
    figure_volume_error(meshes, grids, refs, os.path.join(FIGURES, "mesh_volume_error.png"))
    figure_convergence(report["convergence"], os.path.join(FIGURES, "mesh_volume_convergence.png"))
    report["levelset"] = {
        "circle": levelset_check("circle", grids["circle"], (0.5, 0.25, 0.125, 0.0625)),
        "sphere": levelset_check("sphere", grids["sphere"], (0.2, 0.1, 0.05)),
    }
    for case, chk in report["levelset"].items():
        surf = [f"{e:.2e}" for e in chk["surface_error"]]
        hit = [f"{e:.2e}" for e in chk["hit_error"]]
        print(f"{case} level set: h {chk['h']}, surface error {surf}, hit error {hit}")
    figure_levelset(report["levelset"], os.path.join(FIGURES, "mesh_levelset_checks.png"))
    with open(os.path.join(RESULTS, "geometry_checks.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", os.path.join(RESULTS, "geometry_checks.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
