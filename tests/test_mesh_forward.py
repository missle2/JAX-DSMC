"""Forward physics with an embedded body: FG6 (diffuse-body equilibrium), FG8 (NTC on cut
volumes), the run-level plumbing (`simulate.run(mesh=...)`, tallies, diagnostics) and the
box-only invariance of the step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_dsmc import initialize, simulate
from jax_dsmc.boundary import BC, Faces, WallParams, apply_faces
from jax_dsmc.collide import ntc, species_params
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import cell_volume, num_cells
from jax_dsmc.mesh import cut
from jax_dsmc.mesh import surface as S
from jax_dsmc.mesh.types import CUT, INSIDE, OUTSIDE, AnalyticBody
from jax_dsmc.state import FlowParams, GridSpec, SimParams, SimStatic

MASS = 66.3e-27
T_WALL = 300.0


def _vth(temp):
    return float(np.sqrt(K_BOLTZ * temp / MASS))


def _grid2():
    return GridSpec(lo=jnp.array([0.0, 0.0]), hi=jnp.array([10.0, 10.0]), n=(20, 20))


def _sp():
    return species_params(
        jnp.array([MASS]),
        jnp.array([[4.11e-10]]),
        jnp.array([[0.81]]),
        jnp.array([[273.15]]),
        jnp.array([[1.4]]),
    )


def test_fg6_gas_equilibrates_to_a_diffuse_body(tracker="stencil"):
    """Closed specular box, diffuse circle at T_w: the body alone must set T (FG6)."""
    grid = _grid2()
    body = AnalyticBody(centre=jnp.array([5.0, 5.0]), radius=jnp.float64(3.0))
    mesh = cut.build_mesh(body, grid, radius=2, sub=4)  # radius 2: 1 m covers the 600 K tail
    faces = Faces.uniform(BC.SPECULAR, 2)
    wall = WallParams.uniform(2, T_WALL, acc=0.0, n_models=1).with_model(2, 0, temp=T_WALL, acc=1.0)
    n = 6000
    key = jax.random.PRNGKey(3)
    x = jax.random.uniform(key, (n, 2), dtype=jnp.float64) * 10.0
    v = _vth(2 * T_WALL) * jax.random.normal(jax.random.fold_in(key, 1), (n, 3), dtype=jnp.float64)
    alive = jnp.sum((x - 5.0) ** 2, axis=1) > 9.0
    # 0.18 m per thermal flight (< one cell).  The slow tail is the particles with a tiny
    # x-velocity: specular box faces preserve |v_x|, so one with |v_x| = 0.01 v_th needs
    # 2 / |v_x| ~ 0.6 s to reach the body -- hence 2500 steps of 5e-4 s.
    dt = 5.0e-4
    step = jax.jit(
        lambda x, v, a, k: apply_faces(
            x,
            v,
            a,
            grid,
            faces,
            wall,
            jnp.float64(MASS),
            jnp.float64(dt),
            k,
            mesh=mesh,
            tracker=tracker,
        )
    )
    temps = []
    for s in range(2500):
        x, v, alive, counts, _ = step(x, v, alive, jax.random.fold_in(key, 100 + s))
        if s % 250 == 249:
            temps.append(float(MASS * jnp.mean(jnp.sum(v[alive] ** 2, axis=1)) / (3 * K_BOLTZ)))
    assert bool(counts.inside_body == 0) and bool(counts.stencil_overrun == 0)
    assert int(alive.sum()) == int(jnp.sum(jnp.sum((x - 5.0) ** 2, axis=1) > 9.0))
    assert temps[0] < 2 * T_WALL and temps[-1] == pytest.approx(T_WALL, rel=0.015), temps


def test_fg8_ntc_attempts_scale_with_the_cut_volume():
    """`attempt_count` on a cut grid: `n ~ N_c^2 / V_c`, zero attempts in body cells (FG8)."""
    grid = _grid2()
    poly = S.read_sparta_surf("examples/mesh/sparta/data.circle", 2)
    mesh = cut.build_mesh(poly, grid)
    cut_grid = cut.with_volume(grid, mesh)
    C = num_cells(grid)
    counts = jnp.full((C,), 40, jnp.int32)
    vremax = jnp.full((C,), 1e-2)  # sigma * g [m^3 s^-1], exaggerated for a large count
    # a large expected count so stochastic rounding is a small relative effect
    dt, fnum = jnp.float64(1.0), jnp.full((C,), 1e6)
    n_uni, _ = ntc.attempt_count(counts, vremax, cell_volume(grid), dt, fnum, jax.random.PRNGKey(0))
    n_cut, logp = ntc.attempt_count(
        counts, vremax, cell_volume(cut_grid), dt, fnum, jax.random.PRNGKey(0)
    )
    ct = np.asarray(mesh.cell_type)
    vol = np.asarray(mesh.volume)
    full = float(cell_volume(grid)[0])
    assert (np.asarray(n_uni) > 1000).all()
    np.testing.assert_array_equal(
        np.asarray(n_cut)[ct == OUTSIDE], np.asarray(n_uni)[ct == OUTSIDE]
    )
    assert (np.asarray(n_cut)[ct == INSIDE] == 0).all()
    assert np.isfinite(np.asarray(logp)).all()
    partial = (ct == CUT) & (vol < full)
    ratio = np.asarray(n_cut)[partial] / np.asarray(n_uni)[partial]
    np.testing.assert_allclose(ratio, full / vol[partial], rtol=2e-3)


@pytest.mark.parametrize("tracker", ["stencil", "walk"])
def test_run_with_mesh_reports_hits_tallies_and_zero_failures(tracker):
    grid = _grid2()
    poly = S.read_sparta_surf("examples/mesh/sparta/data.circle", 2)
    mesh = cut.build_mesh(poly, grid)
    grid = cut.with_volume(grid, mesh)
    sp = _sp()
    C = num_cells(grid)
    N = 4000
    params = SimParams(dt=jnp.float64(1e-4), fnum=jnp.full((C,), 1e-3), epsilon=jnp.float64(0.0))
    static = SimStatic(
        N=N, D=2, S=1, G=1, A_max=2, K_bounce=4, n_steps=5, tracker=tracker, K_walk=8
    )
    faces = Faces.uniform(BC.SPECULAR, 2)
    wall = WallParams.uniform(2, T_WALL, acc=0.0, n_models=1)
    flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.zeros(3))
    key = jax.random.PRNGKey(0)
    state = initialize.initial_state(key, flow, grid, sp, static)
    inside = np.asarray(jnp.sum((state.x - 5.0) ** 2, axis=1) < 9.0)
    state = state.__class__(
        x=state.x,
        v=state.v,
        species=state.species,
        alive=state.alive & ~jnp.asarray(inside),
        logh_acc=state.logh_acc,
    )
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    (state2, _), (out, diag, tallies) = simulate.run(
        state,
        cstate,
        grid,
        sp,
        params,
        static,
        key,
        faces=faces,
        wall=wall,
        mesh=mesh,
        emit_surface=True,
    )
    assert int(diag.surface_hits.sum()) > 0
    assert int(diag.inside_body.sum()) == 0 and int(diag.stencil_overrun.sum()) == 0
    assert int(diag.walk_exhausted.sum()) == 0 and int(diag.wall_recrossed.sum()) == 0
    assert tallies.hits.shape == (5, 4 + poly.n_elements)
    assert int(tallies.hits[:, 4:].sum()) == int(diag.surface_hits.sum())
    # the cut volumes feed the density: body cells report zero, cut cells the flow density
    nrho = np.asarray(out.nrho[-1])
    ct = np.asarray(mesh.cell_type)
    assert (nrho[ct == INSIDE] == 0).all() and nrho[ct == OUTSIDE].mean() > 0
    assert int(state2.alive.sum()) == int(state.alive.sum())  # closed box, nothing lost


def test_step_is_bitwise_box_only_when_the_body_is_unreachable():
    """A tiny body in a corner no particle visits: `run(mesh=)` equals the tracker run
    without it -- the alive set, velocities and collisions bitwise, the positions to
    round-off (FG0 at the run level).  The box faces
    are deterministic (an outflow face for the diffuse one): with an embedded surface a diffuse face's re-emission draws are
    per slot (``wall_adjoint._slot_uniforms``), not the box-only tracker's dense draws, so
    a DIFFUSE face would differ in its random stream (not in its law)."""
    grid = _grid2()
    body = AnalyticBody(centre=jnp.array([0.25, 0.25]), radius=jnp.float64(0.1))
    mesh = cut.build_mesh(body, grid, radius=1, sub=4)
    sp = _sp()
    C = num_cells(grid)
    N = 3000
    params = SimParams(dt=jnp.float64(1e-4), fnum=jnp.full((C,), 1e18), epsilon=jnp.float64(0.0))
    static = SimStatic(N=N, D=2, S=1, G=1, A_max=8, K_bounce=4, n_steps=4)
    faces = Faces((BC.OUTFLOW, BC.SPECULAR, BC.PERIODIC, BC.PERIODIC))
    wall = WallParams.uniform(2, T_WALL, n_models=1)
    flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.zeros(3))
    key = jax.random.PRNGKey(1)
    state = initialize.initial_state(key, flow, grid, sp, static)
    far = np.asarray(jnp.sum(state.x**2, axis=1) > 1.0)
    state = state.__class__(
        x=state.x,
        v=state.v,
        species=state.species,
        alive=state.alive & jnp.asarray(far),
        logh_acc=state.logh_acc,
    )
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    ref = simulate.run(state, cstate, grid, sp, params, static, key, faces=faces, wall=wall)
    out = simulate.run(
        state, cstate, grid, sp, params, static, key, faces=faces, wall=wall, mesh=mesh
    )
    (st_ref, cs_ref), (st_out, cs_out) = ref[0], out[0]
    alive = np.asarray(st_ref.alive)
    assert np.array_equal(alive, np.asarray(st_out.alive))
    assert np.array_equal(np.asarray(st_ref.v), np.asarray(st_out.v))
    # the mesh path forms a periodic wrap's position in its own order: round-off only
    np.testing.assert_allclose(
        np.asarray(st_out.x)[alive], np.asarray(st_ref.x)[alive], rtol=0, atol=1e-14
    )
    for a, b in zip(jax.tree.leaves(cs_ref), jax.tree.leaves(cs_out), strict=True):
        assert bool((a == b).all())
    assert int(out[1][1].surface_hits.sum()) == 0


def test_field_frames_leave_the_mesh_forward_bitwise_unchanged():
    """``ensemble_mesh.forward(frames_every=k)`` runs the warm-up in blocks of ``k`` steps
    and snapshots the fields at every block end (the optimiser's movie); the RNG contract
    (keys from the absolute step index) makes the run identical to the one-block warm-up."""
    from examples import ensemble_mesh as EM  # noqa: PLC0415

    cfg = EM.default_config()
    cfg.update({"warmup": 6, "steps": 4, "window": 2, "fnum_scale": 64.0})
    key = jax.random.key(3)
    value = EM.nominal(cfg, "radius")
    state_a, final_a, diag_a, _rec, tallies_a, s_a = EM.forward(cfg, "radius", value, key, False)
    state_b, final_b, diag_b, _rec, tallies_b, s_b = EM.forward(
        cfg, "radius", value, key, False, frames_every=3
    )
    assert s_a["frames"] is None
    frames = s_b["frames"]
    assert list(frames["step"]) == [0, 3, 6, 10]
    assert frames["nrho"].dtype == np.float32
    assert frames["nrho"].shape == (4, num_cells(s_b["grid"]))
    assert frames["u"].shape == (4, num_cells(s_b["grid"]), 3)
    assert float(frames["nrho"][0].max()) == 0.0  # the empty box
    assert float(frames["nrho"][-1].max()) > 0.0
    for a, b in ((state_a, state_b), (final_a, final_b)):
        np.testing.assert_array_equal(np.asarray(a.x), np.asarray(b.x))
        np.testing.assert_array_equal(np.asarray(a.v), np.asarray(b.v))
        np.testing.assert_array_equal(np.asarray(a.alive), np.asarray(b.alive))
    np.testing.assert_array_equal(np.asarray(tallies_a.momentum), np.asarray(tallies_b.momentum))
    assert int(diag_a.surface_hits.sum()) == int(diag_b.surface_hits.sum())
    with pytest.raises(ValueError, match="must divide"):
        EM.forward(cfg, "radius", value, key, False, frames_every=4)


def test_split_cells_run_with_their_own_density_and_collision_cells():
    """A thin plate (0.1 thick) inside a cell row: the 12 cells it spans are split cells.
    A collisional run in a closed box conserves particles, reports a density for every
    region (both sides of the plate at the freestream value) and never puts particles
    from opposite sides of the plate into one collision cell."""
    from jax_dsmc.grid import collision_cell  # noqa: PLC0415

    grid = _grid2()
    pts = np.array([[2.0, 4.7], [8.0, 4.7], [8.0, 4.8], [2.0, 4.8]])
    plate = S.make_surface(pts, np.array([[1, 0], [2, 1], [3, 2], [0, 3]]))
    mesh = cut.build_mesh(plate, grid)
    assert mesh.n_sub == 12
    grid = cut.with_volume(grid, mesh)
    sp = _sp()
    C = num_cells(grid)
    assert C == 412
    params = SimParams(dt=jnp.float64(1e-4), fnum=jnp.full((C,), 2e-3), epsilon=jnp.float64(0.0))
    static = SimStatic(N=60000, D=2, S=1, G=1, A_max=8, K_bounce=4, n_steps=10)
    faces = Faces.uniform(BC.SPECULAR, 2)
    wall = WallParams.uniform(2, T_WALL, acc=0.0, n_models=1)
    flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.zeros(3))
    key = jax.random.PRNGKey(3)
    state = initialize.initial_state(key, flow, grid, sp, static)
    x = np.asarray(state.x)
    in_plate = (x[:, 0] > 2.0) & (x[:, 0] < 8.0) & (x[:, 1] > 4.7) & (x[:, 1] < 4.8)
    state = state.__class__(
        x=state.x,
        v=state.v,
        species=state.species,
        alive=state.alive & ~jnp.asarray(in_plate),
        logh_acc=state.logh_acc,
    )
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    (state2, _), (out, diag) = simulate.run(
        state, cstate, grid, sp, params, static, key, faces=faces, wall=wall, mesh=mesh
    )
    assert int(diag.inside_body.sum()) == 0 and int(diag.stencil_overrun.sum()) == 0
    assert int(diag.attempts_clipped.sum()) == 0
    assert int(state2.alive.sum()) == int(state.alive.sum())
    nrho = np.asarray(out.nrho).mean(axis=0)
    assert nrho.shape == (412,)
    split = np.nonzero(np.asarray(mesh.split_of) >= 0)[0]
    free = np.asarray(mesh.cell_type) == OUTSIDE
    ref = nrho[:400][free].mean()
    # every region of every split cell: ~ the freestream density (~40 particles each)
    np.testing.assert_allclose(nrho[split].mean(), ref, rtol=0.1)
    np.testing.assert_allclose(nrho[400:].mean(), ref, rtol=0.1)
    # collision cells never mix the two sides of the plate
    xs = np.asarray(state2.x)[np.asarray(state2.alive)]
    cc = np.asarray(collision_cell(jnp.asarray(xs), grid))
    span = (xs[:, 0] > 2.0) & (xs[:, 0] < 8.0)  # the split cells; beyond it no plate
    below = cc[span & (xs[:, 1] < 4.7) & (xs[:, 1] >= 4.5)]
    above = cc[span & (xs[:, 1] > 4.8) & (xs[:, 1] < 5.0)]
    assert below.size > 100 and above.size > 100
    assert not (set(below) & set(above))
