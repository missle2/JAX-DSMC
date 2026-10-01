"""Fast checks of sparta_jax (the SPARTA-faithful forward for the circle deck)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparta_jax import geometry as G
from sparta_jax import sim


@functools.cache
def _setup(**kw):
    return sim.make(**kw)


def test_geometry_matches_sparta_read_surf():
    geo = G.build()
    # read_surf in log.circle_ar: "264 88 48 = cells outside/inside/overlapping surfs",
    # "71.8 71.8 = cell-wise and global flow volume"
    assert np.sum(geo.cell_type == G.OUTSIDE) == 264
    assert np.sum(geo.cell_type == G.INSIDE) == 88
    assert np.sum(geo.cell_type == G.OVERLAP) == 48
    assert abs(geo.volume.sum() - 71.8) < 1e-3
    # outward normals: the body centre is on the INSIDE side of every line
    mid = 0.5 * (geo.p1 + geo.p2)
    assert np.all(np.sum((mid - 5.0) * geo.norm, axis=1) > 0)
    assert np.allclose(geo.task_ntarget, 7.6695157, atol=1e-6)


def test_prefix_sum_and_floor_helpers():
    r = np.random.default_rng(1)
    for n in (40, 128, 5120, 53248):
        m = (r.random(n) < 0.3).astype(np.int32)
        assert np.array_equal(np.asarray(jax.jit(sim.cumsum)(jnp.asarray(m))), np.cumsum(m))
    v = r.random(1000) * 40.0 - 5.0
    v[:5] = [0.0, 1.0, -1.0, 3.0, 19.0]
    got = jax.jit(lambda a: sim.floor_int(a * 1.0))(jnp.asarray(v))  # jitted: fused
    assert np.array_equal(np.asarray(got), np.floor(v).astype(np.int32))


def test_line_line_intersect():
    f = sim._line_line_intersect
    # line from (0,-1) to (0,1), normal +x; move from (1,0) to (-1,0): hit at param 0.5
    hit, param, px, py, outside = f(1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0)
    assert bool(hit) and bool(outside) and float(param) == 0.5 and float(px) == 0.0
    # starting inside (behind the line): hit but side INSIDE -> not a collision in Update::move
    hit, _p, _x, _y, outside = f(-1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0)
    assert bool(hit) and not bool(outside)
    # passes beyond the segment end
    hit, *_ = f(1.0, 2.0, -1.0, 2.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0)
    assert not bool(hit)
    # both ends on the same side
    hit, *_ = f(1.0, 0.0, 0.5, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0)
    assert not bool(hit)


def _one_particle(cfg, x, y, vx, vy):
    st = sim.empty_state(cfg)
    c = int(np.floor(y / cfg.dy)) * cfg.nx + int(np.floor(x / cfg.dx))
    return st._replace(
        x=st.x.at[0].set(x),
        y=st.y.at[0].set(y),
        vx=st.vx.at[0].set(vx),
        vy=st.vy.at[0].set(vy),
        cell=st.cell.at[0].set(c),
    )


def test_specular_reflection_off_the_circle():
    cfg, consts, geo = _setup(capacity=1024, lane=64, lane2=64)
    # head-on at the leftmost vertex region of the circle (x = 2 at y = 5)
    st = _one_particle(cfg, 1.99, 4.9, 300.0, 0.0)
    new, _surf, counts = sim.move(st, jnp.uint64(1), jnp.int32(0), consts, cfg)
    assert int(counts["nscoll"]) == 1
    assert float(new.vx[0]) < 0.0  # bounced back
    speed0 = 300.0
    speed1 = float(jnp.hypot(new.vx[0], new.vy[0]))
    assert abs(speed1 - speed0) < 1e-9
    # the particle ends outside the body and in a flow cell
    assert not G._inside_polygon((float(new.x[0]), float(new.y[0])), geo.body_polygon)
    assert geo.cell_type[int(new.cell[0])] != G.INSIDE


def test_outflow_and_reflect_boundaries():
    cfg, consts, _geo = _setup(capacity=1024, lane=64, lane2=64)
    st = _one_particle(cfg, 9.99, 0.005, 400.0, -200.0)  # exits through xhi
    new, _s, counts = sim.move(st, jnp.uint64(1), jnp.int32(0), consts, cfg)
    assert int(new.cell[0]) == cfg.ncell  # a dense exit (counted by conservation in step)
    st = _one_particle(cfg, 5.0, 0.005, 10.0, -200.0)  # mirrors off ylo
    new, _s, counts = sim.move(st, jnp.uint64(1), jnp.int32(0), consts, cfg)
    assert float(new.vy[0]) == 200.0 and abs(float(new.y[0]) - 0.015) < 1e-12
    assert int(counts["nbound"]) == 1


def test_partition_sort_equals_full_sort_and_run_is_sane():
    cfg, consts, _geo = _setup(capacity=16384, lane=2048, lane2=512, mover_sizes=(512, 2048))
    st = sim.empty_state(cfg)
    st, stats = sim.run(st, 5, jnp.int32(0), consts, cfg, 60)
    s = sim.as_numpy_stats(stats)
    assert s["overflow"].sum() == 0 and s["emit_fail"].sum() == 0
    # fill rate: ~153.4 inserts per step, almost no exits in 60 steps
    assert abs(s["ninsert"].mean() - 153.4) < 5.0
    assert int(s["np"][-1]) == int(np.sum(s["ninsert"]) - np.sum(s["nexit"]))
    full = sim.sort_full(st, consts, cfg)
    assert np.array_equal(np.asarray(full.start), np.asarray(st.start))
    assert np.array_equal(np.asarray(sim.skey(full)), np.asarray(sim.skey(st)))
    # same cell groups (order inside a cell is free)
    cell = np.asarray(st.cell)
    assert np.array_equal(cell[np.asarray(sim.plist(st))], np.asarray(sim.skey(st)))
    assert np.array_equal(np.sort(np.asarray(sim.plist(st))), np.arange(cfg.capacity))
    # the cached cut-free range of every live slot is its cell's
    alive = cell < cfg.ncell
    assert np.array_equal(np.asarray(st.scode)[alive], np.asarray(consts.safe)[cell[alive]])


def test_lane_sizes_do_not_change_the_run():
    """Lane overflow (several lane blocks, later stages spilling over their lanes) and the
    mover-sort buckets are bookkeeping: the run is bit-identical."""
    a = _setup(capacity=16384, lane=2048, lane2=512, mover_sizes=(512, 2048))
    b = _setup(capacity=16384, lane=256, lane2=64, lane3=8, mover_sizes=(64, 128))
    out = []
    for cfg, consts, _geo in (a, b):
        st, stats = sim.run(sim.empty_state(cfg), 5, jnp.int32(0), consts, cfg, 40)
        out.append((st, sim.as_numpy_stats(stats), cfg))
    (sa, ta, ca), (sb, tb, _cb) = out
    assert int(tb["nlane"].max()) > 256  # several lane blocks were needed
    for k in ("np", "nscoll", "nscheck", "nexit", "nbound"):
        assert np.array_equal(ta[k], tb[k]), k
    for f in ("x", "y", "vx", "vy", "vz", "cell"):
        assert np.array_equal(np.asarray(getattr(sa, f)), np.asarray(getattr(sb, f))), f


@pytest.mark.parametrize("seed", [3])
def test_collisions_conserve_momentum_and_energy(seed):
    # 1e20 x denser gas with 1e20 x heavier simulators: same particle count, ~40 attempts/cell/step
    deck = G.Deck(nrho=1.0e20, fnum=1.0e17)
    cfg, consts, _geo = sim.make(
        deck, capacity=16384, lane=2048, lane2=512, mover_sizes=(512, 2048)
    )
    st = sim.empty_state(cfg)
    st, stats = sim.run(st, seed, jnp.int32(0), consts, cfg, 40)
    assert int(np.sum(stats.ncoll)) > 0
    alive = np.asarray(st.cell) < cfg.ncell
    v0 = np.stack([np.asarray(a)[alive] for a in (st.vx, st.vy, st.vz)])
    after, nattempt, ncoll = jax.jit(sim.collide, static_argnames="cfg")(
        st, jnp.uint64(seed), jnp.int32(41), consts, cfg
    )
    v1 = np.stack([np.asarray(a)[alive] for a in (after.vx, after.vy, after.vz)])
    assert int(ncoll) > 0 and int(nattempt) >= int(ncoll)
    assert np.allclose(v0.sum(1), v1.sum(1), rtol=0, atol=1e-6 * np.abs(v0).sum())
    e0, e1 = np.sum(v0**2), np.sum(v1**2)
    assert abs(e1 / e0 - 1.0) < 1e-12
    assert np.sum(np.any(v0 != v1, axis=0)) >= int(ncoll)
