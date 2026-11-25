"""Seconds-scale checks of the 3-D benchmark port (sparta_jax.geometry3d / sim3d / driver)."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sparta_jax  # noqa: F401  (x64)
from sparta_jax import decks as D
from sparta_jax import driver
from sparta_jax import geometry3d as G3
from sparta_jax import sim3d as S


def _line_tri_ref(start, stop, v0, v1, v2, norm):
    """Geometry::line_tri_intersect, transcribed (numpy, one pair)."""
    dotstart = norm @ (start - v0)
    dotstop = norm @ (stop - v0)
    if dotstart < 0 and dotstop < 0 or dotstart > 0 and dotstop > 0 or dotstart == 0 == dotstop:
        return False, 0.0, False
    param = (norm @ (v0 - start)) / (norm @ (stop - start))
    param = min(max(param, 0.0), 1.0)
    point = start + param * (stop - start)
    for a, b in ((v0, v1), (v1, v2), (v2, v0)):
        if np.cross(b - a, point - a) @ norm < -1.0e-16:
            return False, 0.0, False
    return True, param, dotstart > 0


def test_line_tri_matches_sparta():
    rng = np.random.default_rng(0)
    pts, tris = G3.read_surf_tris(D.sphere().surf_file)
    s = G3.make_surf(pts, tris)
    M = 400
    t = rng.integers(0, len(tris), M)
    start = rng.normal(size=(M, 3)) * 1.2
    stop = rng.normal(size=(M, 3)) * 1.2
    # half the segments aimed through a point near the triangle (inside or just outside it)
    w = rng.dirichlet([1.0, 1.0, 1.0], M // 2) * 1.1 - 0.033
    aim = w[:, :1] * s.p1[t[: M // 2]] + w[:, 1:2] * s.p2[t[: M // 2]] + w[:, 2:] * s.p3[t[: M // 2]]
    nrm = s.norm[t[: M // 2]]
    start[: M // 2] = aim + nrm * rng.uniform(-0.05, 0.1, (M // 2, 1))
    stop[: M // 2] = aim - nrm * rng.uniform(-0.05, 0.1, (M // 2, 1)) + rng.normal(size=(M // 2, 3)) * 0.01
    tri = np.concatenate([s.p1[t], s.p2[t], s.p3[t], s.norm[t]], axis=1)[:, None, :]
    hit, param, out = S._line_tri(
        [jnp.asarray(start[:, a : a + 1]) for a in range(3)],
        [jnp.asarray(stop[:, a : a + 1]) for a in range(3)],
        jnp.asarray(tri),
    )
    hit, param, out = (np.asarray(v)[:, 0] for v in (hit, param, out))
    for m in range(M):
        h, p, o = _line_tri_ref(start[m], stop[m], s.p1[t[m]], s.p2[t[m]], s.p3[t[m]], s.norm[t[m]])
        assert h == hit[m]
        if h:
            assert p == pytest.approx(param[m], abs=1e-14)
            assert o == out[m]
    assert 50 < hit.sum() < M  # both outcomes exercised
    assert out[hit].any() and (~out[hit]).any()


@pytest.mark.parametrize("n", [(1, 1, 1), (2, 1, 1), (3, 3, 3), (10, 10, 10), (20, 20, 20)])
def test_cut_volumes_add_up_to_the_body(n):
    pts, tris = G3.read_surf_tris(D.sphere().surf_file)
    s = G3.make_surf(pts, tris)
    grid = G3.Grid(lo=(-5.0, -5.0, -5.0), hi=(5.0, 5.0, 5.0), n=n)
    cut = G3.cut_cells(grid, s)
    inside = G3.classify_uncut(grid, s, cut.gid)
    chk = G3.check_volumes(grid, s, cut, inside)
    assert abs(chk["rel_err"]) < 1e-8
    assert (cut.volume > -1e-12).all()


def test_sphere_census_matches_sparta_log():
    """read_surf in bench/log.*.sphere.*.10K.1: '32 = cells with surfs', '1860 = total surfs'."""
    g = G3.build(D.sphere(10, 10, 10))
    assert len(g.cut.gid) == 32
    assert int(g.cut.nsurf.sum()) == 1860


def _small(deck, steps):
    runs = tuple(dataclasses.replace(r, nsteps=steps) for r in deck.runs)
    return dataclasses.replace(deck, runs=runs)


def test_collide_box_conserves_count_and_energy():
    d = _small(D.collide(4, 4, 4, steps=20), 20)
    geo, cfg, consts, state, _ = driver.setup(d, seed=3)
    alive = np.asarray(state.cell) < cfg.ncell_local

    def energy(st):
        v2 = np.asarray(st.vx) ** 2 + np.asarray(st.vy) ** 2 + np.asarray(st.vz) ** 2
        return float(np.sum(v2[np.asarray(st.cell) < cfg.ncell_local]))

    e0, n0 = energy(state), int(alive.sum())
    state = S.reset_vremax(state, cfg)
    state, st = S.run(state, 0, d.runs[0].dt, 3, consts, cfg, 10)
    state, st2 = S.run(state, 10, d.runs[1].dt, 3, consts, cfg, 10)
    assert int(np.asarray(st.ncoll).sum()) > 0  # the collision path is exercised
    cell = np.asarray(state.cell)
    assert int((cell < cfg.ncell_local).sum()) == n0
    assert energy(state) == pytest.approx(e0, rel=1e-12)
    # every particle is inside the box and in the cell that holds it
    x = np.stack([np.asarray(state.x), np.asarray(state.y), np.asarray(state.z)], 1)[cell < cfg.ncell_local]
    lo, hi, h = np.array(cfg.lo), np.array(cfg.hi), np.array(cfg.h)
    assert ((x >= lo) & (x <= hi)).all()
    ijk = np.minimum(np.floor((x - lo) / h).astype(int), np.array(cfg.n) - 1)
    lc = ijk[:, 0] + cfg.n[0] * (ijk[:, 1] + cfg.n[1] * ijk[:, 2])
    assert (lc == cell[cell < cfg.ncell_local]).all()


def test_sphere_runs_clean_and_inflow_matches_flux():
    d = _small(D.sphere(10, 10, 10), 100)
    res = driver.run_deck(d, seed=5)
    tot = res["runs"][-1]["totals"]
    assert tot["overflow"] == 0 and tot["emit_fail"] == 0 and tot["nstuck"] == 0
    assert tot["nscoll"] > 0 and tot["ncoll"] > 0
    g = G3.build(d)
    expect = float(g.task_rate.sum() * d.runs[-1].dt * 100)
    assert abs(tot["ninsert"] - expect) < 5 * np.sqrt(expect)


_DECOMP_SCRIPT = r"""
import dataclasses, json, sys
sys.path.insert(0, {root!r})
import sparta_jax
from sparta_jax import decks as D, driver
out = {{}}
for name, n in (("collide", (12, 8, 8)), ("sphere", (12, 12, 12))):
    d = D.make(name, *n)
    d = dataclasses.replace(d, runs=tuple(dataclasses.replace(r, nsteps=100) for r in d.runs))
    out[name] = {{}}
    for nrank in (1, 8):
        res = driver.run_deck_decomposed(d, nrank, seed=7)
        r = res["runs"][-1]
        out[name][nrank] = {{"np": r["np_end"], "moves": r["particle_moves"], **r["totals"]}}
print("RESULT" + json.dumps(out))
"""


def test_decomposition_8_ranks_matches_1():
    """Bricks 2x2x2 on 8 forced host devices against one device: the box conserves the count
    exactly, nothing overflows, and the per-move rates agree within noise."""
    import json
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=8", JAX_PLATFORMS="cpu")
    p = subprocess.run(
        [sys.executable, "-c", _DECOMP_SCRIPT.format(root=root)], env=env, capture_output=True, text=True, timeout=900
    )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
    assert line, p.stderr[-3000:]
    out = json.loads(line[0][6:])
    box = out["collide"]
    assert box["1"]["np"] == box["8"]["np"] == 7680
    for name in out:
        for nr in ("1", "8"):
            assert out[name][nr]["overflow"] == 0
        a, b = out[name]["1"], out[name]["8"]
        for k in ("ncoll", "nscheck"):
            ra, rb = a[k] / a["moves"], b[k] / b["moves"]
            if a[k] > 1000:
                assert abs(ra - rb) / ra < 0.1, (name, k, ra, rb)


def test_collide_variants_are_bitwise_identical():
    """The tuning knobs of the collision stage change speed only: 'prefix' (attempts on the
    cells that have them) and unrolled attempts give the same state as the default."""
    d = _small(D.collide(6, 6, 6, steps=10), 10)
    geo, cfg, consts, state, _ = driver.setup(d, seed=11)
    state = S.reset_vremax(state, cfg)
    state, _ = S.run(state, 0, d.runs[0].dt, 11, consts, cfg, 10)  # large dt: many attempts per cell
    outs = []
    for kw in ({}, {"collide_mode": "prefix"}, {"collide_unroll": 3}, {"sort": "full"}):
        c2 = dataclasses.replace(cfg, **kw)
        st, stats = S.run(jax.tree.map(jnp.copy, state), 10, d.runs[0].dt, 11, consts, c2, 5)
        outs.append((np.asarray(st.vx), np.asarray(st.vremax), np.asarray(stats.ncoll)))
    assert outs[0][2].sum() > 100
    for o in outs[1:]:
        for a, b in zip(outs[0], o, strict=True):
            assert np.array_equal(a, b)
