"""Phase C of the mesh plan: ``grad_x`` from the fit, value-form rewards in the device
backward, the traced replay of the wall map and its VJP with respect to the geometry.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

from jax_dsmc import ensemble_backward, ensemble_device, initialize, shape, simulate
from jax_dsmc.boundary import BC, Faces, WallParams
from jax_dsmc.collide import species_params
from jax_dsmc.collide.ntc import PairRecord
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import cell_neighbourhood as grid_neighbourhood
from jax_dsmc.grid import num_cells
from jax_dsmc.mesh import cut, levelset
from jax_dsmc.mesh import geometry as G
from jax_dsmc.mesh import surface as S
from jax_dsmc.mesh.types import AnalyticBody
from jax_dsmc.sampling import (
    diffuse_velocity_general,
    log_diffuse_density,
    standard_diffuse_draws,
    thermal_speed,
)
from jax_dsmc.simulate import EventRecord
from jax_dsmc.state import FlowParams, GridSpec, SimParams, SimStatic
from jax_dsmc.wall_adjoint import WallEvents, no_wall_events

MASS = 66.3e-27
T_WALL = 300.0


def _sp():
    return species_params(
        jnp.array([MASS]),
        jnp.array([[4.11e-10]]),
        jnp.array([[0.81]]),
        jnp.array([[273.15]]),
        jnp.array([[1.4]]),
    )


def _grid2():
    return GridSpec(lo=jnp.array([0.0, 0.0]), hi=jnp.array([10.0, 10.0]), n=(20, 20))


# --- field_query: grad_x -------------------------------------------------------------


def test_field_query_returns_exact_position_slopes_of_a_quadratic_field():
    key = jax.random.PRNGKey(0)
    q, p, dim = 3, 40, 2
    node_x = jax.random.uniform(jax.random.fold_in(key, 1), (q, p, dim))
    node_v = jax.random.normal(jax.random.fold_in(key, 2), (q, p, 3))
    b = jnp.array([0.7, -1.3])
    c = jnp.array([0.2, 0.5, -0.4])
    a = jnp.array([0.3, -0.6])

    def field(x, v):
        return 1.5 + x @ b + v @ c + (x**2) @ a + 0.1 * v[..., 0] * v[..., 1]

    node_val = field(node_x, node_v)
    q_x = node_x[:, 0, :]
    q_v = node_v[:, 0, :]
    scale = jnp.array([0.5, 0.5, 1.0, 1.0, 1.0])
    value, grad_x, grad_v = ensemble_device.field_query(
        node_x, node_v, jnp.ones((q, p), bool), node_val, q_x, q_v, scale, 32
    )
    # the relative ridge (1e-8) is what limits the fit, not the arithmetic
    np.testing.assert_allclose(np.asarray(value), np.asarray(field(q_x, q_v)), rtol=1e-5)
    expect_x = b + 2.0 * a * q_x
    np.testing.assert_allclose(np.asarray(grad_x), np.asarray(expect_x), rtol=1e-4, atol=1e-6)
    expect_v = c + 0.1 * jnp.stack((q_v[:, 1], q_v[:, 0], jnp.zeros(q)), axis=1)
    np.testing.assert_allclose(np.asarray(grad_v), np.asarray(expect_v), rtol=1e-4, atol=1e-6)


# --- value-form rewards ----------------------------------------------------------------


def _collisionless_record(n, m, k_pass, dim, key):
    """``m`` steps of ``n`` free-streaming particles in one cell: no accepted attempt, no
    newborn, one recorded hit per step on particle 0 at pass 1."""
    a_max, c = 2, 1
    x = jax.random.uniform(key, (m, n, dim))
    v = jax.random.normal(jax.random.fold_in(key, 1), (m, n, 3))
    walls = no_wall_events(n, k_pass)
    walls = WallEvents(
        walls.face.at[1, 0].set(2 * dim),
        walls.v_out.at[1, 0].set(v[0, 0]),
        walls.diffuse.at[1, 0].set(True),
    )
    walls = jax.tree.map(lambda a: jnp.broadcast_to(a, (m, *a.shape)), walls)
    pairs = PairRecord(
        jnp.full((m, a_max, c), n, jnp.int32),
        jnp.full((m, a_max, c), n, jnp.int32),
        jnp.zeros((m, a_max, c), bool),
        jnp.zeros((m, a_max, c, 3)),
        jnp.zeros((m, a_max, c, 3)),
        jnp.zeros((m, a_max, c, 3)),
        jnp.zeros((m, a_max, c, 3)),
    )
    return EventRecord(
        x=x,
        v=v,
        alive=jnp.ones((m, n), bool),
        born_face=jnp.full((m, n), -1, jnp.int32),
        born_v=jnp.zeros((m, n, 3)),
        pairs=pairs,
        v_post=v,
        walls=walls,
        sigma=jnp.ones((m, c)),
    )


@pytest.mark.parametrize("virtual", [False, True])
def test_value_form_rewards_accumulate_backward_without_collisions(virtual):
    """No collisions: the carried value is the exact suffix sum of the step rewards, the
    hit rewards join on the move pull-back, and the per-hit weights are the rewards-to-go."""
    n, m, k_pass, dim = 8, 5, 3, 2
    key = jax.random.PRNGKey(4)
    rec = _collisionless_record(n, m, k_pass, dim, key)
    grid = GridSpec(jnp.zeros(dim), jnp.ones(dim), (1,) * dim)
    phi_step = jax.random.normal(jax.random.fold_in(key, 7), (m, n))
    hit_reward = jnp.zeros((m, k_pass, n)).at[:, 1, 0].set(jnp.arange(m, dtype=jnp.float64) + 1.0)
    hit_reward = hit_reward.at[2, 2, 0].set(10.0)
    phi_final = jnp.full(n, 0.25)
    final = simulate.ParticleState(
        rec.x[-1], rec.v_post[-1], jnp.zeros(n, jnp.int32), rec.alive[-1], jnp.zeros(n)
    )
    scale = jnp.ones(dim + 3)
    if virtual:
        dev = ensemble_device.backward_virtual(
            rec,
            jnp.ones(n, bool),
            final,
            phi_final,
            grid,
            scale,
            _sp(),
            jax.random.PRNGKey(1),
            npc_max=n,
            k=8,
            substeps=2,
            partners=2,
            phi_step=phi_step,
            hit_reward=hit_reward,
        )
    else:
        dev = ensemble_device.backward(
            rec,
            jnp.ones(n, bool),
            final,
            phi_final,
            grid,
            scale,
            npc_max=n,
            k=8,
            phi_step=phi_step,
            hit_reward=hit_reward,
        )
    lam = np.asarray(dev.lam)
    phi = np.asarray(phi_step)
    r = np.asarray(hit_reward).sum(axis=1)
    # lam[k] = value at step k's post-move state = phi_k + (rewards of steps k+1..) + phi_final
    expect = np.zeros_like(lam)
    carry = np.asarray(phi_final)
    for k in reversed(range(m)):
        carry = carry + phi[k]
        expect[k] = carry
        carry = carry + r[k]
    np.testing.assert_allclose(lam, expect, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(dev.lambda0), carry, rtol=1e-12)
    adj = ensemble_backward.from_device(rec, dev, 1.0, hit_reward=hit_reward)
    hits = {(h.step, int(h.slots[0])): h for h in adj.hits}
    # the recorded hit is at pass 1: its weight is lam plus the rewards from pass 1 on
    for k in range(m):
        h = hits[(k, 0)]
        from_here = float(np.asarray(hit_reward)[k, 1:, 0].sum())
        np.testing.assert_allclose(h.values[0], lam[k, 0] + from_here, rtol=1e-12)


# --- the diffuse density and its normal score ------------------------------------------


def test_log_diffuse_density_is_frame_independent_and_normalised():
    key = jax.random.PRNGKey(2)
    n = 5
    normal = jax.random.normal(key, (n, 3))
    normal = normal / jnp.linalg.norm(normal, axis=1, keepdims=True)
    v_out = jax.random.normal(jax.random.fold_in(key, 1), (n, 3)) * 300.0
    v_out = v_out + 500.0 * normal  # on the flow side
    temp = 300.0
    lp = log_diffuse_density(v_out, normal, temp, MASS, jnp.zeros((n, 3)))
    # normalisation: Monte Carlo of exp(lp) over the half-space in the frame of normal 0
    s = float(jnp.sqrt(K_BOLTZ * temp / MASS))
    m = 200_000
    kz = jax.random.fold_in(key, 3)
    z = jax.random.normal(kz, (m, 3)) * s * 4.0
    lpz = log_diffuse_density(z, jnp.broadcast_to(normal[:1], (m, 3)), temp, MASS, jnp.zeros(3))
    dens = jnp.exp(jnp.where(jnp.isfinite(lpz), lpz, -jnp.inf))
    weight = (2.0 * jnp.pi * (4.0 * s) ** 2) ** 1.5 * jnp.exp(
        jnp.sum(z**2, axis=1) / (2 * (4 * s) ** 2)
    )
    integral = float(jnp.mean(dens * weight))
    assert abs(integral - 1.0) < 0.03
    # a rotation about the normal leaves the density unchanged
    axis = normal[0]
    ang = 0.7
    rot = (
        jnp.cos(ang) * jnp.eye(3)
        + jnp.sin(ang)
        * jnp.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        + (1 - jnp.cos(ang)) * jnp.outer(axis, axis)
    )
    lp_rot = log_diffuse_density(v_out[:1] @ rot.T, normal[:1], temp, MASS, jnp.zeros((1, 3)))
    np.testing.assert_allclose(float(lp_rot[0]), float(lp[0]), rtol=1e-10)


def test_normal_score_matches_finite_differences_of_the_density():
    dim = 2
    grid = _grid2()
    body = AnalyticBody(centre=jnp.array([5.0, 5.0]), radius=jnp.float64(3.0))
    mesh = cut.build_mesh(body, grid, radius=1, sub=4)
    wall = WallParams.uniform(2, T_WALL, acc=0.0, n_models=1).with_model(
        2, 0, temp=350.0, acc=1.0, vwall=jnp.array([0.0, 0.0, 40.0])
    )
    k_pass, n = 2, 3
    ev = no_wall_events(n, k_pass, trace_dim=dim)
    normal = jnp.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 1.0, 0.0]])
    v_out = jnp.array([[300.0, 50.0, -20.0], [200.0, 300.0, 10.0], [-40.0, 250.0, 60.0]])
    ev = ev._replace(
        face=ev.face.at[0].set(jnp.array([4, 4, 1], jnp.int32)),
        diffuse=ev.diffuse.at[0].set(jnp.array([True, False, True])),
        v_out=ev.v_out.at[0].set(v_out),
        normal=ev.normal.at[0].set(normal),
    )
    weight = jnp.array([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]])
    cot = np.asarray(shape.normal_score(ev, weight, wall, mesh, MASS, dim))
    # only the diffuse-branch element hit (slot 0) is scored
    assert np.all(cot[1] == 0) and np.all(cot[0, 1:] == 0)
    h = 1e-6
    for axis in range(3):
        d = jnp.zeros(3).at[axis].set(h)
        lp = functools.partial(
            log_diffuse_density,
            v_out[:1],
            temp=350.0,
            mass=MASS,
            vwall=jnp.array([[0.0, 0.0, 40.0]]),
        )
        fd = (float(lp(normal[:1] + d)[0]) - float(lp(normal[:1] - d)[0])) / (2 * h)
        np.testing.assert_allclose(cot[0, 0, axis], 2.0 * fd, rtol=1e-5, atol=1e-8)


def test_normal_score_grazing_floor_keeps_the_primal_and_bounds_the_tail():
    """A grazing re-emission (``v_out . n = 1e-3 |v_out|``) scores ``v_t / v_n`` = 1000 |v_t|
    per unit weight; with ``grazing = 0.05`` the ``1 / v_n`` is floored at ``1 / (0.05 |v|)``
    and a well-inclined emission is unchanged."""
    dim = 2
    grid = _grid2()
    centre, radius = jnp.array([5.0, 5.0]), jnp.float64(3.0)
    wall = WallParams.uniform(2, T_WALL, acc=0.0, n_models=1).with_model(2, 0, temp=350.0, acc=1.0)
    normal = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    v_out = jnp.array([[0.3, 300.0, 0.0], [200.0, 300.0, 10.0]])
    ev = no_wall_events(2, 1, trace_dim=dim)
    ev = ev._replace(
        face=ev.face.at[0].set(jnp.array([4, 4], jnp.int32)),
        diffuse=ev.diffuse.at[0].set(jnp.array([True, True])),
        v_out=ev.v_out.at[0].set(v_out),
        normal=ev.normal.at[0].set(normal),
    )
    weight = jnp.ones((1, 2))
    cots = {}
    for eps in (0.0, 0.05):
        body = AnalyticBody(centre=centre, radius=radius, grazing=eps)
        mesh = cut.build_mesh(body, grid, radius=1, sub=4)
        cots[eps] = np.asarray(shape.normal_score(ev, weight, wall, mesh, MASS, dim))[0]
        # the density itself never changes
        lp = log_diffuse_density(v_out, normal, 350.0, MASS, jnp.zeros((2, 3)), eps * 300.0)
        np.testing.assert_allclose(
            np.asarray(lp),
            np.asarray(log_diffuse_density(v_out, normal, 350.0, MASS, jnp.zeros((2, 3)))),
        )
    # exact: the tangential component of the score is v_t / v_n = 300 / 0.3 = 1000
    np.testing.assert_allclose(cots[0.0][0, 1], 1000.0, rtol=1e-6)
    # floored: v_t / max(v_n, 0.05 |v|) = 300 / 15 = 20
    np.testing.assert_allclose(cots[0.05][0, 1], 20.0, rtol=1e-6)
    # a well-inclined emission (v_n = 200 > 0.05 |v|) is untouched
    np.testing.assert_allclose(cots[0.05][1], cots[0.0][1], rtol=1e-12)


def _emitted_x(normal, z, r, v_in):
    return diffuse_velocity_general(z, r, normal, 350.0, MASS, jnp.zeros((z.shape[0], 3)), v_in)[
        :, 0
    ]


def test_diffuse_reparameterisation_in_the_normal_is_bounded_and_unbiased():
    """``diffuse_velocity_general`` differentiated in the normal at fixed draws: the sample
    is the SPARTA one, the tangent is ``sigma (r dn - (z1 t1 . dn + z2 t2 . dn) n)``
    (parallel-transported frame, bounded even at normal incidence where the incident
    frame's derivative is ``1 / |v_t,in|``), and its draw average is the derivative of the
    mean emitted velocity ``sigma sqrt(pi/2) dn``."""
    n = 200_000
    z, r = standard_diffuse_draws(jax.random.PRNGKey(5), n)
    normal = jnp.tile(jnp.array([[1.0, 0.0, 0.0]]), (n, 1))
    # near-normal incidence: |v_t,in| / |v| = 1e-9
    v_in = jnp.tile(jnp.array([[-500.0, 5e-7, 0.0]]), (n, 1))
    dn = jnp.tile(jnp.array([[0.0, 1.0, 0.0]]), (n, 1))
    sigma = float(thermal_speed(350.0, MASS))
    f = functools.partial(
        diffuse_velocity_general, z, r, temp=350.0, mass=MASS, vwall=jnp.zeros((n, 3)), v_in=v_in
    )
    v_out, dv = jax.jvp(f, (normal,), (dn,))
    # the primal is the incident-frame sample
    ref = diffuse_velocity_general(z, r, normal, 350.0, MASS, jnp.zeros((n, 3)), v_in)
    np.testing.assert_array_equal(np.asarray(v_out), np.asarray(ref))
    # t1 is along the incident tangential direction (+y): the frame rotates as -(t . dn) n
    t1 = np.array([0.0, 1.0, 0.0])
    expected = sigma * (
        np.asarray(r)[:, None] * np.asarray(dn)
        - (np.asarray(z)[:, :1] * (t1 @ np.array([0.0, 1.0, 0.0]))) * np.asarray(normal)
    )
    np.testing.assert_allclose(np.asarray(dv), expected, rtol=1e-9, atol=1e-9 * sigma)
    assert np.abs(np.asarray(dv)).max() < 10.0 * sigma
    # unbiased: mean tangent = d/dn E[v_out] = sigma sqrt(pi / 2) dn
    mean = np.asarray(dv).mean(axis=0)
    se = np.asarray(dv).std(axis=0, ddof=1) / np.sqrt(n)
    target = sigma * np.sqrt(np.pi / 2) * np.array([0.0, 1.0, 0.0])
    assert np.all(np.abs(mean - target) < 4.0 * se + 1e-12)


# --- replay and VJP -----------------------------------------------------------------------


def _circle_setup(kind, n_elem=50, radius=1):
    grid = _grid2()
    if kind == "analytic":
        body = AnalyticBody(centre=jnp.array([5.0, 5.0]), radius=jnp.float64(3.0))
    elif kind == "levelset":
        # reach = n_sample h / 2 = 3.0 covers the 2.25 m flights of the VJP test
        lo, m = levelset.grid_for_box(grid.lo, grid.hi, 0.125)
        body = levelset.from_analytic(jnp.array([5.0, 5.0]), 3.0, lo, 0.125, m, n_sample=48)
    else:
        body = S.circle(3.0, n_elem, jnp.array([5.0, 5.0]))
    mesh = cut.build_mesh(body, grid, radius=radius, sub=4)
    return grid, mesh


@pytest.mark.parametrize("kind", ["analytic", "polygon", "levelset"])
def test_wall_map_vjp_matches_finite_differences_on_specular_hits(kind):
    """One deterministic move per particle, several hitting the specular body: the VJP of
    the post-move state with respect to the geometry equals central differences."""
    grid, mesh = _circle_setup(kind, radius=6)  # 2.25 m flights: the stencil must cover them
    faces = Faces.uniform(BC.SPECULAR, 2)
    wall = WallParams.uniform(2, T_WALL, acc=0.0, n_models=1)
    n = 64
    key = jax.random.PRNGKey(5)
    ang = jax.random.uniform(key, (n,)) * 2 * jnp.pi
    # start 1.0 outside the circle, aim roughly at the centre with some spread
    x = jnp.array([5.0, 5.0]) + 4.0 * jnp.stack((jnp.cos(ang), jnp.sin(ang)), axis=1)
    aim = jnp.array([5.0, 5.0]) + jax.random.normal(jax.random.fold_in(key, 1), (n, 2)) * 1.5
    v = 1500.0 * (aim - x) / jnp.linalg.norm(aim - x, axis=1, keepdims=True)
    v = jnp.concatenate((v, jnp.zeros((n, 1))), axis=1)
    alive = jnp.ones(n, bool)
    dt = 1.5e-3
    static = SimStatic(N=n, D=2, S=1, G=1, A_max=2, K_bounce=4, n_steps=1)
    cot_x = jax.random.normal(jax.random.fold_in(key, 2), (n, 2))
    cot_v = jax.random.normal(jax.random.fold_in(key, 3), (n, 3))

    def scalar(geom):
        x1, v1, *_ = shape.wall_map(
            geom, x, v, alive, key, grid, faces, wall, MASS, dt, mesh, 0.0, 0.0, 4
        )
        return jnp.sum(cot_x * x1) + jnp.sum(cot_v * v1)

    geom = shape.geometry(mesh)
    _, _, _, _, _, ev = shape.wall_map(
        geom, x, v, alive, key, grid, faces, wall, MASS, dt, mesh, 0.0, 0.0, 4
    )
    assert int(jnp.sum(ev.face[0] >= 4)) > 20  # most of them hit
    grad = jax.grad(scalar)(geom)
    h = 1e-6
    flat, unravel = jax.flatten_util.ravel_pytree(geom)
    gflat, _ = jax.flatten_util.ravel_pytree(grad)
    rng = np.random.default_rng(0)
    # a level set has thousands of coefficients, most far from the surface: test the live ones
    live = np.nonzero(np.abs(np.asarray(gflat)) > 1e-3 * np.abs(np.asarray(gflat)).max())[0]
    for i in rng.choice(live, size=min(6, live.size), replace=False):
        d = jnp.zeros_like(flat).at[i].set(h)
        fd = (float(scalar(unravel(flat + d))) - float(scalar(unravel(flat - d)))) / (2 * h)
        np.testing.assert_allclose(float(gflat[i]), fd, rtol=2e-5, atol=1e-6)
    del static


def test_traced_events_are_consistent_with_the_geometry():
    grid, mesh = _circle_setup("analytic")
    faces = Faces.uniform(BC.SPECULAR, 2)
    wall = WallParams.uniform(2, T_WALL, acc=0.0, n_models=1).with_model(2, 0, temp=T_WALL, acc=1.0)
    n = 200
    key = jax.random.PRNGKey(6)
    x = jax.random.uniform(key, (n, 2)) * 10.0
    v = 600.0 * jax.random.normal(jax.random.fold_in(key, 1), (n, 3))
    alive = jnp.sum((x - 5.0) ** 2, axis=1) > 9.0
    _, _, _, _, _, ev = shape.wall_map(
        shape.geometry(mesh), x, v, alive, key, grid, faces, wall, MASS, 2e-3, mesh, 0.0, 0.0, 4
    )
    hit = np.asarray(ev.face >= 4)
    assert hit.sum() > 10
    xh = np.asarray(ev.x_hit)[hit]
    np.testing.assert_allclose(np.linalg.norm(xh - 5.0, axis=1), 3.0, rtol=1e-12)
    nrm = np.asarray(ev.normal)[hit]
    np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(nrm[:, :2], (xh - 5.0) / 3.0, atol=1e-12)
    assert (np.sum(np.asarray(ev.v_in)[hit] * nrm, axis=1) < 0).all()
    assert (np.sum(np.asarray(ev.v_out)[hit] * nrm, axis=1) > 0).all()
    assert np.asarray(ev.diffuse)[hit].all()


def _drag_reward(events, tallies, rec_k, step, mass, dim, weight=None):
    del tallies, rec_k, step
    dv = events.v_in - events.v_out
    r = mass * (dv[..., 0] if weight is None else jnp.sum(dv[..., :dim] * weight, axis=-1))
    return jnp.where(events.face >= 2 * dim, r, 0.0)


def _replay_chain(tracker, record="full", weight=None, **gradient_kw):
    """Forward -> traced replay -> value-form backward -> shape VJP on a short specular
    circle run: ``(terms, replay, diag, tallies, record)``; ``weight`` (``f64[2]``) the
    traced momentum weights of the reward and the tally cotangent (``reward_weight``)."""
    grid, mesh = _circle_setup("analytic")
    grid = cut.with_volume(grid, mesh)
    sp = _sp()
    C = num_cells(grid)
    N = 3000
    params = SimParams(dt=jnp.float64(2e-4), fnum=jnp.full((C,), 1e-3), epsilon=jnp.float64(0.0))
    static = SimStatic(
        N=N, D=2, S=1, G=1, A_max=2, K_bounce=4, n_steps=6, tracker=tracker, K_walk=8,
        record=record,
    )  # fmt: skip
    faces = Faces.uniform(BC.SPECULAR, 2)
    wall = WallParams.uniform(2, T_WALL, acc=0.0, n_models=1)
    flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.zeros(3))
    key = jax.random.PRNGKey(0)
    state = initialize.initial_state(key, flow, grid, sp, static)
    inside = jnp.sum((state.x - 5.0) ** 2, axis=1) < 9.0
    state = state.__class__(
        x=state.x,
        v=3.0 * state.v,
        species=state.species,
        alive=state.alive & ~inside,
        logh_acc=state.logh_acc,
    )
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    (final, _), (_, diag, rec, tallies) = simulate.run(
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
        emit_moments="none",
        capture=True,
        emit_surface=True,
    )
    assert int(diag.surface_hits.sum()) > 50
    steps = jnp.arange(static.n_steps, dtype=jnp.int32)
    reward = functools.partial(_drag_reward, mass=MASS, dim=2)
    rep = shape.replay(
        rec, steps, state, key, grid, faces, wall, MASS, params.dt, mesh, 0.0, 0.0, static, reward,
        reward_weight=weight,
    )  # fmt: skip
    # a backward with the drag as hit reward, then the shape terms
    scale = jnp.array([0.5, 0.5, 300.0, 300.0, 300.0])
    dev = ensemble_device.backward_virtual(
        rec,
        state.alive,
        final,
        jnp.zeros(N),
        grid,
        scale,
        sp,
        jax.random.PRNGKey(9),
        npc_max=64,
        k=16,
        substeps=1,
        partners=1,
        hit_reward=rep.hit_total[:, None],
    )
    assert np.isfinite(np.asarray(dev.lam)).all()
    w = jnp.array([1.0, 0.0]) if weight is None else weight
    d_tally = tallies.__class__(
        None, jnp.zeros((4 + 1, 3)).at[4, :2].set(w), jnp.zeros(5), jnp.zeros(5)
    )
    terms = shape.gradient(
        rec,
        dev.lam,
        steps,
        state,
        key,
        grid,
        faces,
        wall,
        MASS,
        params.dt,
        mesh,
        0.0,
        0.0,
        static,
        scale,
        64,
        hit_reward=rep.hit_reward,
        tally_cotangent=d_tally,
        k=16,
        reward_weight=weight,
        **gradient_kw,
    )
    return terms, rep, diag, tallies, rec


def test_weighted_rewards_give_the_combination_of_the_component_gradients():
    """``reward_weight`` (2026-10-25, the lift-to-drag objective): the per-hit momentum
    weighted by ``(a, b)`` through the replay, backward and shape VJP is ``a`` times the drag
    chain plus ``b`` times the lift chain (everything downstream is linear in the rewards),
    and the weight ``(1, 0)`` is the unweighted drag."""
    drag = _replay_chain("stencil")
    x_only = _replay_chain("stencil", weight=jnp.array([1.0, 0.0]))
    lift = _replay_chain("stencil", weight=jnp.array([0.0, 1.0]))
    mixed = _replay_chain("stencil", weight=jnp.array([0.3, -2.0]))
    np.testing.assert_allclose(
        np.asarray(lift[1].hit_total).sum(axis=1),
        np.asarray(lift[3].momentum[:, 4:, 1]).sum(axis=1),
        rtol=1e-12,
    )
    for part in ("pathwise_x", "pathwise_v", "pathwise_tally", "score"):
        leaves = [
            [np.asarray(a) for a in jax.tree.leaves(getattr(t[0], part))]
            for t in (drag, x_only, lift, mixed)
        ]
        for d, x, y, m in zip(*leaves, strict=True):
            np.testing.assert_allclose(x, d, rtol=1e-13, atol=1e-15 * np.abs(d).max())
            ref = 0.3 * x - 2.0 * y
            np.testing.assert_allclose(m, ref, rtol=1e-9, atol=1e-10 * np.abs(ref).max())


@pytest.mark.parametrize("tracker", ["stencil", "lane"])
def test_lean_record_gives_the_full_records_chain_bitwise(tracker):
    """``SimStatic.record = "lean"`` (``simulate.lean_record``) keeps only what the replay,
    the device backward and the shape VJP read: the whole chain -- replay rewards, the
    stratified classes (from the ``i8`` surface ids) and every shape term -- is bitwise the
    full record's."""
    full = _replay_chain(tracker, value_class="hit_ahead")
    lean = _replay_chain(tracker, "lean", value_class="hit_ahead")
    rec_f, rec_l = full[4], lean[4]
    assert rec_l.v_post is None and rec_l.born_v is None and rec_l.pairs.vi is None
    assert rec_l.born_face.dtype == jnp.int8
    np.testing.assert_array_equal(
        np.asarray(shape.hit_ahead(rec_l, None)), np.asarray(shape.hit_ahead(rec_f, None))
    )
    np.testing.assert_array_equal(np.asarray(lean[1].hit_total), np.asarray(full[1].hit_total))
    for part in ("pathwise", "pathwise_x", "pathwise_v", "pathwise_tally", "score"):
        a, b = getattr(full[0], part), getattr(lean[0], part)
        for leaf_a, leaf_b in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_array_equal(np.asarray(leaf_b), np.asarray(leaf_a))


@pytest.mark.parametrize("tracker", ["stencil", "lane"])
def test_batched_steps_give_the_sequential_shape_terms(tracker):
    """``shape.gradient(batch=3)``: the captured steps three at a time through ``vmap``, each
    runtime branch of the move taken once per batch (:mod:`jax_dsmc.step_axis`) -- the
    sequential scan's terms up to summation order, the same replay check."""
    seq = _replay_chain(tracker)[0]
    bat = _replay_chain(tracker, batch=3)[0]
    assert float(bat.replay_error) < 1e-12
    np.testing.assert_array_equal(np.asarray(bat.surface_hits), np.asarray(seq.surface_hits))
    for part in ("pathwise_x", "pathwise_v", "pathwise_tally", "score"):
        a, b = getattr(seq, part), getattr(bat, part)
        for leaf_a, leaf_b in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_allclose(np.asarray(leaf_b), np.asarray(leaf_a), rtol=1e-12,
                                       atol=1e-14 * float(np.max(np.abs(leaf_a))))  # fmt: skip


@pytest.mark.parametrize("tracker", ["stencil", "lane", "walk"])
def test_replay_reproduces_the_captured_forward_bitwise_and_the_vjp_runs(tracker):
    """A short run with the specular circle: the traced replay of every step matches the
    record to roundoff (the RNG contract), the hit rewards agree with the tallies, and the
    shape gradient is finite with a nonzero pathwise radius term."""
    terms, rep, diag, tallies, _rec = _replay_chain(tracker)
    assert float(rep.replay_error) < 1e-12
    np.testing.assert_array_equal(np.asarray(rep.surface_hits), np.asarray(diag.surface_hits))
    np.testing.assert_allclose(
        np.asarray(rep.hit_total).sum(axis=1),
        np.asarray(tallies.momentum[:, 4:, 0]).sum(axis=1),
        rtol=1e-12,
    )
    assert float(terms.replay_error) < 1e-12
    np.testing.assert_array_equal(np.asarray(terms.surface_hits), np.asarray(diag.surface_hits))
    assert np.isfinite(float(terms.pathwise.radius)) and float(terms.pathwise.radius) != 0.0
    assert np.isfinite(np.asarray(terms.pathwise.centre)).all()
    # specular body, epsilon 0: nothing is scored
    assert float(terms.score.radius) == 0.0 and np.all(np.asarray(terms.score.centre) == 0.0)


def test_lane_and_stencil_shape_gradients_agree():
    """The lane tracker's split events through replay, backward and shape VJP give the
    dense stencil's shape gradient (the same events, folded per slot) to roundoff."""
    ts, rs, ds, _, rec_s = _replay_chain("stencil")
    tl, rl, dl, _, rec_l = _replay_chain("lane")
    # the lane's compact record: the same stratified-fit classes and, step by step, the
    # same events as the stencil's dense record
    np.testing.assert_array_equal(
        np.asarray(shape.hit_ahead(rec_l, None)), np.asarray(shape.hit_ahead(rec_s, None))
    )
    for k in range(rec_s.x.shape[0]):
        dense = ensemble_backward._step_walls(rec_s.walls, k)
        compact = ensemble_backward._step_walls(rec_l.walls, k)
        rows = compact[0].shape[0]
        np.testing.assert_array_equal(compact[0], dense[0][:rows])
        assert (dense[0][rows:] < 0).all()
    np.testing.assert_array_equal(np.asarray(ds.surface_hits), np.asarray(dl.surface_hits))
    np.testing.assert_allclose(np.asarray(rl.hit_total), np.asarray(rs.hit_total), rtol=1e-12)
    for part in ("pathwise", "pathwise_x", "pathwise_v", "pathwise_tally"):
        a, b = getattr(ts, part), getattr(tl, part)
        np.testing.assert_allclose(float(b.radius), float(a.radius), rtol=1e-9)
        np.testing.assert_allclose(
            np.asarray(b.centre), np.asarray(a.centre), rtol=1e-9, atol=1e-30
        )


def test_timing_cotangent_selects_the_band_with_the_lost_jump():
    dt, band = 0.05, 0.2
    ev = no_wall_events(4, 2, trace_dim=2)
    ev = ev._replace(
        face=ev.face.at[0].set(jnp.array([4, 4, -1, 1], jnp.int32)),
        time=ev.time.at[0].set(jnp.array([0.045, 0.02, 0.049, 0.041])),
    )
    jump = jnp.ones((2, 4)) * 3.0
    cot = np.asarray(shape.timing_cotangent(ev, jump, dt, band))
    # in the band and recorded: slots 0 (element) and 3 (diffuse face); slot 1 is early,
    # slot 2 has no recorded hit; pass 1 has no events at all
    np.testing.assert_allclose(cot[0], [-3.0 / (band * dt), 0.0, 0.0, -3.0 / (band * dt)])
    assert np.all(cot[1] == 0.0)


def test_traced_times_are_the_hit_times():
    grid, mesh = _circle_setup("analytic")
    faces = Faces.uniform(BC.SPECULAR, 2)
    wall = WallParams.uniform(2, T_WALL, acc=0.0, n_models=1)
    n = 200
    key = jax.random.PRNGKey(8)
    x = jax.random.uniform(key, (n, 2)) * 10.0
    v = 600.0 * jax.random.normal(jax.random.fold_in(key, 1), (n, 3))
    alive = jnp.sum((x - 5.0) ** 2, axis=1) > 9.0
    dt = 2e-3
    _, _, _, _, _, ev = shape.wall_map(
        shape.geometry(mesh), x, v, alive, key, grid, faces, wall, MASS, dt, mesh, 0.0, 0.0, 4
    )
    hit = np.asarray(ev.face[0] >= 4)
    t = np.asarray(ev.time[0])[hit]
    flight = np.linalg.norm(np.asarray(ev.x_hit[0])[hit] - np.asarray(x)[hit], axis=1)
    speed = np.linalg.norm(np.asarray(v)[hit][:, :2], axis=1)
    np.testing.assert_allclose(t, flight / speed, rtol=1e-10)
    assert (t > 0).all() and (t <= dt).all()


def test_nearest_normal_picks_the_closest_surface_within_one_cell():
    grid = _grid2()
    # analytic circle: radial normal within one cell (0.5) of the surface, zero further away
    body = AnalyticBody(centre=jnp.array([5.0, 5.0]), radius=jnp.float64(3.0))
    mesh = cut.build_mesh(body, grid, radius=1, sub=4)
    x = jnp.array([[8.2, 5.0], [5.0, 8.4], [9.0, 5.0], [1.0, 1.0], [0.3, 5.0]])
    n = np.asarray(G.nearest_normal(x, grid, mesh, diffuse_faces=(0,)))
    np.testing.assert_allclose(n[0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(n[1], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.all(n[2] == 0.0) and np.all(n[3] == 0.0)
    np.testing.assert_allclose(n[4], [1.0, 0.0, 0.0], atol=1e-12)  # the diffuse face x = 0
    # polygon: the nearest ELEMENT, not the nearest vertex
    poly = S.circle(3.0, 8, jnp.array([5.0, 5.0]))
    mesh8 = cut.build_mesh(poly, grid, radius=1, sub=4)
    x = jnp.array([[8.2, 5.2], [5.0 + 3.1 * np.cos(0.3), 5.0 + 3.1 * np.sin(0.3)]])
    n = np.asarray(G.nearest_normal(x, grid, mesh8))
    ang = np.arctan2(n[:, 1], n[:, 0])
    # the octagon's element normals are at (2j+1) pi/8; a point at polar angle 0 or 0.3
    # lies nearest to the elements whose normals are at +/- pi/8 and pi/8
    assert np.isclose(abs(ang[0]), np.pi / 8, atol=1e-9)
    assert np.isclose(ang[1], np.pi / 8, atol=1e-9)
    assert np.allclose(np.linalg.norm(n[:, :2], axis=1), 1.0)


def test_one_sided_backward_is_a_no_op_with_zero_normals_and_runs_with_real_ones():
    n, m, k_pass, dim = 12, 4, 2, 2
    key = jax.random.PRNGKey(11)
    rec = _collisionless_record(n, m, k_pass, dim, key)
    grid = GridSpec(jnp.zeros(dim), jnp.ones(dim), (1,) * dim)
    final = simulate.ParticleState(
        rec.x[-1], rec.v_post[-1], jnp.zeros(n, jnp.int32), rec.alive[-1], jnp.zeros(n)
    )
    args = (rec, jnp.ones(n, bool), final, jnp.full(n, 0.25), grid, jnp.ones(dim + 3), _sp())
    kw = {"npc_max": n, "k": 8, "substeps": 2, "partners": 2}
    ref = ensemble_device.backward_virtual(*args, jax.random.PRNGKey(1), **kw)
    zero = ensemble_device.backward_virtual(
        *args, jax.random.PRNGKey(1), normals=jnp.zeros((m, n, 3)), **kw
    )
    np.testing.assert_array_equal(np.asarray(ref.lam), np.asarray(zero.lam))
    normals = jnp.zeros((m, n, 3)).at[:, :, 0].set(1.0)
    one = ensemble_device.backward_virtual(*args, jax.random.PRNGKey(1), normals=normals, **kw)
    assert np.isfinite(np.asarray(one.lam)).all()
    # local_normals of a real mesh has the record's shape and only unit or zero rows
    grid2, mesh = _circle_setup("analytic")
    x = jax.random.uniform(key, (m, n, 2)) * 10.0
    rec2 = rec.__class__(**{**rec.__dict__, "x": x})
    nrm = np.asarray(shape.local_normals(rec2, grid2, mesh, Faces.uniform(BC.SPECULAR, 2)))
    assert nrm.shape == (m, n, 3)
    lengths = np.linalg.norm(nrm, axis=-1)
    assert np.all((np.abs(lengths - 1.0) < 1e-12) | (lengths == 0.0))


def test_pooled_one_sided_node_gradients_are_exact_on_a_quadratic_field():
    """Nodes of the 3^D block: a quadratic field in (x, v) is fitted exactly, one- or
    two-sided, and the pooled table holds exactly the neighbours' valid nodes."""
    key = jax.random.PRNGKey(3)
    n, dim = 600, 2
    grid = GridSpec(jnp.zeros(dim), jnp.ones(dim), (4, 3))
    x = jax.random.uniform(key, (n, dim))
    v = jax.random.normal(jax.random.fold_in(key, 1), (n, 3))
    b, a, c = jnp.array([0.7, -1.3]), jnp.array([0.3, -0.6]), jnp.array([0.2, 0.5, -0.4])
    lam = 1.5 + x @ b + (x**2) @ a + v @ c + 0.1 * v[:, 0] * v[:, 1]
    alive = jnp.ones(n, bool)
    scale = jnp.array([0.25, 1 / 3, 1.0, 1.0, 1.0])
    direction = jnp.zeros((n, 3)).at[:, 0].set(jnp.sign(v[:, 0]))
    for pool, k in ((0, 24), (1, 96)):
        gx, gv = ensemble_device.node_gradients(
            x, v, alive, lam, grid, 128, scale, k, pool=pool, direction=direction
        )
        np.testing.assert_allclose(np.asarray(gx), np.asarray(b + 2 * a * x), rtol=1e-3, atol=1e-4)
        expect_v = c + 0.1 * jnp.stack((v[:, 1], v[:, 0], jnp.zeros(n)), axis=1)
        np.testing.assert_allclose(np.asarray(gv), np.asarray(expect_v), rtol=1e-3, atol=1e-4)
    ids, valid, cell, _, count = ensemble_device.cell_nodes(x, alive, grid, 128)
    tables = ensemble_device.pooled_tables((x[ids], v[ids], valid, lam[ids]), grid, 1)
    nbr, exists = grid_neighbourhood(grid, 1)
    expect = np.array(
        [int(np.asarray(count)[nbr[c]][np.asarray(exists)[c]].sum()) for c in range(12)]
    )
    np.testing.assert_array_equal(np.asarray(tables[2]).sum(axis=1), expect)


def test_stratified_node_gradients_ignore_the_other_class():
    """Two populations in every cell -- "about to hit" with ``lam = 2 m v_x`` and
    "reflected" with ``lam = 0`` (the phase-space split next to a wall): the mixed fit of
    a reflected particle reads a ``d lam / d v_x`` of order ``2 m P(hit)`` (the specular
    capsule's -27 %), the class-stratified fit reads zero, and the hitting class keeps
    its exact slope."""
    key = jax.random.PRNGKey(5)
    n, dim = 800, 2
    grid = GridSpec(jnp.zeros(dim), jnp.ones(dim), (4, 3))
    x = jax.random.uniform(key, (n, dim))
    v = jax.random.normal(jax.random.fold_in(key, 1), (n, 3))
    klass = jax.random.bernoulli(jax.random.fold_in(key, 2), 0.5, (n,))
    lam = jnp.where(klass, 2.0 * v[:, 0], 0.0)
    alive = jnp.ones(n, bool)
    scale = jnp.array([0.25, 1 / 3, 1.0, 1.0, 1.0])
    for pool, k in ((0, 24), (1, 96)):
        _gx, gv_mixed = ensemble_device.node_gradients(
            x, v, alive, lam, grid, 128, scale, k, pool=pool
        )
        _gx, gv = ensemble_device.node_gradients(
            x, v, alive, lam, grid, 128, scale, k, pool=pool, klass=klass
        )
        mixed = np.asarray(gv_mixed[:, 0])[~np.asarray(klass)]
        assert abs(mixed.mean()) > 0.5  # the mixture's slope, order 2 P(hit)
        np.testing.assert_allclose(np.asarray(gv[:, 0])[~np.asarray(klass)], 0.0, atol=1e-9)
        np.testing.assert_allclose(np.asarray(gv[:, 0])[np.asarray(klass)], 2.0, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(gv[:, 1:])[np.asarray(klass)], 0.0, atol=1e-4)


def test_node_gradients_at_queried_slots_match_the_full_fit():
    """``queries``: the fit at a subset of slots (with unused ``-1`` entries, a dead
    queried slot, one-sided directions and the class stratification) equals the full
    call's rows there and is zero elsewhere -- what ``shape.gradient(hit_fits=True)``
    relies on."""
    key = jax.random.PRNGKey(7)
    n, dim = 700, 2
    grid = GridSpec(jnp.zeros(dim), jnp.ones(dim), (4, 3))
    x = jax.random.uniform(key, (n, dim))
    v = jax.random.normal(jax.random.fold_in(key, 1), (n, 3))
    lam = jnp.sin(3 * x[:, 0]) + x[:, 1] * v[:, 0] + 0.2 * v[:, 1] ** 2
    alive = jnp.arange(n) % 17 != 0
    klass = jax.random.bernoulli(jax.random.fold_in(key, 2), 0.5, (n,))
    direction = jnp.zeros((n, 3)).at[:, 1].set(jnp.sign(v[:, 1]))
    scale = jnp.array([0.25, 1 / 3, 1.0, 1.0, 1.0])
    picked = np.array([3, 17, 40, 41, 200, 699, 512])  # 17: a dead slot
    queries = jnp.asarray(np.concatenate([picked, -np.ones(9, int)]), dtype=jnp.int32)
    for pool, k in ((0, 24), (1, 96)):
        kw = {"pool": pool, "direction": direction, "klass": klass}
        gx, gv = ensemble_device.node_gradients(x, v, alive, lam, grid, 128, scale, k, **kw)
        qx, qv = ensemble_device.node_gradients(
            x, v, alive, lam, grid, 128, scale, k, queries=queries, **kw
        )
        mask = np.zeros(n, bool)
        mask[picked] = True
        np.testing.assert_allclose(
            np.asarray(qx)[mask], np.asarray(gx)[mask], rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            np.asarray(qv)[mask], np.asarray(gv)[mask], rtol=1e-12, atol=1e-14
        )
        assert not np.asarray(qx)[~mask].any() and not np.asarray(qv)[~mask].any()
        assert not np.asarray(qx)[17].any()  # dead


def test_hit_ahead_marks_later_body_hits_only():
    """``shape.hit_ahead``: true at the steps strictly before a slot's body hit (surface
    id ``>= 2D``); box-face hits (``< 2D``) and the hit's own step do not count."""
    from types import SimpleNamespace  # noqa: PLC0415

    m, k, n, dim = 5, 2, 3, 2
    face = np.full((m, k, n), -1, np.int32)
    face[3, 0, 0] = 2 * dim + 4  # slot 0: body hit at step 3
    face[1, 1, 1] = 1  # slot 1: a box face at step 1 only
    face[0, 0, 2] = 2 * dim  # slot 2: body hit at step 0 (nothing ahead of it)
    face[4, 1, 2] = 2 * dim  # ... and again at step 4
    rec = SimpleNamespace(x=jnp.zeros((m, n, dim)), walls=SimpleNamespace(face=jnp.asarray(face)))
    ahead = np.asarray(shape.hit_ahead(rec, None))
    np.testing.assert_array_equal(ahead[:, 0], [True, True, True, False, False])
    np.testing.assert_array_equal(ahead[:, 1], [False] * m)
    np.testing.assert_array_equal(ahead[:, 2], [True, True, True, True, False])


def test_along_flight_replaces_only_the_component_along_the_velocity():
    gx = jnp.array([[1.0, 2.0], [0.3, -0.4]])
    v = jnp.array([[3.0, 0.0, 5.0], [1.0, 1.0, 0.0]])
    t = jnp.array([0.6, -0.2])
    out = np.asarray(shape.along_flight(gx, v, t))
    vd = np.asarray(v[:, :2])
    # along the flight: v . grad = -T ; across it: unchanged
    np.testing.assert_allclose(np.sum(out * vd, axis=1), -np.asarray(t), atol=1e-12)
    perp = np.stack((-vd[:, 1], vd[:, 0]), axis=1)
    np.testing.assert_allclose(np.sum(out * perp, axis=1), np.sum(np.asarray(gx) * perp, axis=1))


# --- silhouette band term -------------------------------------------------------------------


class _Rec(NamedTuple):
    x: jax.Array
    v: jax.Array
    alive: jax.Array


def _drag(events, tallies, rec_k, step, mass, scale):
    del tallies, rec_k, step
    return jnp.where(
        events.face >= 4, mass * (events.v_in[..., 0] - events.v_out[..., 0]) * scale, 0.0
    )


@pytest.mark.parametrize("acc", [0.0, 1.0])
def test_silhouette_term_counts_the_hit_set_change(acc):
    """Rays along +x at uniformly spaced impact parameters past the level-set circle, the
    drag reward and a zero adjoint (``lam = 0``): the silhouette term contracted with
    ``d coef / d r = -1`` must equal (rate of new hits per unit radius = 2 / db) x (mean
    reward of a grazing hit).  Specular: a grazing hit gives no reward -> ~0.  Diffuse:
    the incident x-momentum is absorbed, the re-emission has no mean x-component at the
    silhouette (normal along y) -> ``m v_x scale`` per new hit."""
    grid = _grid2()
    lo, m = levelset.grid_for_box(grid.lo, grid.hi, 0.125)
    ls = levelset.from_analytic(jnp.array([5.0, 5.0]), 3.0, lo, 0.125, m, n_sample=16)
    mesh = cut.build_mesh(ls, grid, radius=1, sub=4)
    faces = Faces.uniform(BC.SPECULAR, 2)
    wall = WallParams.uniform(2, T_WALL, acc=0.0, n_models=1).with_model(2, 0, temp=T_WALL, acc=acc)
    db = 0.002
    b = jnp.arange(-4.0, 4.0, db)
    n = b.shape[0]
    # start 0.4 before the closest-approach abscissa of the silhouette rays, 1 m/step flights
    x = jnp.stack((jnp.full(n, 5.0 - 0.4), 5.0 + b), axis=1)
    v = jnp.zeros((n, 3)).at[:, 0].set(500.0)
    dt = 1.0 / 500.0
    alive = jnp.ones(n, bool)
    key = jax.random.PRNGKey(3)
    static = SimStatic(N=n, D=2, S=1, G=1, A_max=2, K_bounce=4, n_steps=1)
    setup = shape._setup(key, grid, faces, wall, MASS, dt, mesh, 0.0, 0.0, static)["map"]
    scale = 1e-3
    reward = functools.partial(_drag, mass=MASS, scale=scale)
    geom = shape.geometry(mesh)
    x1, v1, _a, _lp, _tal, events = shape.wall_map(geom, x, v, alive, key, **setup)
    r_k = reward(events, None, None, 0)
    to_go = shape._to_go(
        jnp.zeros(n), shape._as_event_array(r_k, events), shape.as_split(events).slot
    )
    rec = _Rec(x1, v1, alive)
    fit = {
        "npc_max": 512,
        "scale": jnp.array([0.5, 0.5, 300.0, 300.0, 300.0]),
        "k": 32,
        "basis": "quadratic_full",
        "chunk": 4096,
        "pool": 0,
    }
    band = 0.05
    c, x_star = shape.silhouette_cotangent(
        mesh.body,
        x,
        v,
        alive,
        events,
        to_go,
        rec,
        jnp.zeros(n),
        0,
        jax.random.PRNGKey(9),
        setup,
        fit,
        band,
        reward,
    )
    n_band = int((c != 0).sum())
    assert 60 < n_band < 140  # 4 sides x band / db = 100 expected
    # contract the cotangent with d coef / d r = -1 through phi(x*) -> d(term)/d r
    _phi, vjp = jax.vjp(
        functools.partial(shape._phi_at, ls=mesh.body, x_star=x_star), mesh.body.coef
    )
    (g,) = vjp(c)
    d_dr = -float(jnp.sum(g))
    expected = 2.0 / db * MASS * 500.0 * scale  # new hits per unit radius x absorbed x-momentum
    if acc == 0.0:
        assert abs(d_dr) < 0.05 * expected
    else:
        np.testing.assert_allclose(d_dr, expected, rtol=0.15)


def test_replay_mismatch_separates_discrete_divergence_from_roundoff():
    """``shape._mismatch``: a particle whose outcome differs (alive in the replay, dead in
    the record, or the reverse) reports 1 however close the positions are; round-off
    reports its size; a slot the inflow refilled (the record's newborn) is not a flip."""
    grid = GridSpec(lo=jnp.array([0.0, 0.0]), hi=jnp.array([10.0, 10.0]), n=(4, 4))
    x = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    v = jnp.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [50.0, 50.0, 0.0]])
    alive = jnp.array([True, True, True])

    class _Rec(NamedTuple):
        x: jax.Array
        v: jax.Array
        alive: jax.Array
        born_face: jax.Array

    rec = _Rec(x, v, jnp.array([True, True, True]), jnp.array([-1, -1, -1]))
    surv = alive & rec.alive
    noise = x.at[0, 0].add(3e-12 * 10.0)
    err = float(shape._mismatch(noise, v, alive, alive, rec, surv, grid))
    np.testing.assert_allclose(err, 3e-12, rtol=1e-3)
    # the replay loses particle 1 (an outflow the record does not have): discrete
    flip = jnp.array([True, False, True])
    assert float(shape._mismatch(x, v, alive, flip, rec, surv, grid)) == 1.0
    # the record's slot 1 died and was refilled by the inflow: the replay's death is right
    reborn = _Rec(x, v, jnp.array([True, True, True]), jnp.array([-1, 0, -1]))
    surv2 = alive & reborn.alive & (reborn.born_face < 0)
    assert float(shape._mismatch(x, v, alive, flip, reborn, surv2, grid)) == 0.0


# --- cut-volume term -----------------------------------------------------------------------


def test_volume_weights_sum_the_cell_collision_terms_over_the_volume():
    grid = GridSpec(
        lo=jnp.array([0.0, 0.0]), hi=jnp.array([3.0, 1.0]), n=(3, 1),
        volume=jnp.array([0.5, 2.0, 0.0]),
    )  # fmt: skip
    cells = jnp.array([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    w = np.asarray(shape.volume_weights(cells, grid))
    np.testing.assert_allclose(w, [-8.0, -1.0, 0.0])


def test_volume_cotangent_of_unit_weights_is_the_flow_area_derivative():
    """Unit weights sum the flow volumes: a circle of radius r grown by dr (every
    coefficient lowered by dr) loses ``2 pi r dr`` of flow area."""
    grid = _grid2()
    lo, shp = levelset.grid_for_box(grid.lo, grid.hi, 0.125)
    ls = levelset.from_analytic(jnp.array([5.0, 5.0]), 3.0, lo, 0.125, shp, n_sample=8)
    mesh = cut.build_mesh(ls, grid, radius=1)
    grid = cut.with_volume(grid, mesh)
    w = jnp.ones(grid.volume.shape[0]) / float(grid.transverse)
    cot = shape.volume_cotangent(ls.coef, mesh, grid, w)
    np.testing.assert_allclose(-float(jnp.sum(cot)), -2.0 * np.pi * 3.0, rtol=2e-3)


def test_halfspace_drops_the_nodes_behind_the_query_plane():
    """A thin body's far side: nodes with y < 0.5 carry an unrelated field.  Queries above
    the plane y = 0.5 with normal +y fit the near field exactly once the half-space drops
    the far nodes; without it the pooled fit mixes the two."""
    key = jax.random.PRNGKey(5)
    n, dim = 1200, 2
    grid = GridSpec(jnp.zeros(dim), jnp.ones(dim), (4, 4))
    x = jax.random.uniform(key, (n, dim))
    v = jax.random.normal(jax.random.fold_in(key, 1), (n, 3))
    near = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * v[:, 1]
    lam = jnp.where(x[:, 1] >= 0.5, near, 5.0 - 7.0 * x[:, 1] + 2.0 * v[:, 0])
    alive = jnp.ones(n, bool)
    scale = jnp.array([0.25, 0.25, 1.0, 1.0, 1.0])
    above = (x[:, 1] > 0.5) & (x[:, 1] < 0.6) & (v[:, 1] > 0.0)
    queries = jnp.where(above, jnp.arange(n), -1)
    direction = jnp.zeros((n, 3)).at[:, 1].set(1.0)
    # halfspace 0: only the nodes at or above each query's own y, all of them near-side
    gx, gv = ensemble_device.node_gradients(
        x, v, alive, lam, grid, 256, scale, 48, pool=1, direction=direction,
        queries=queries, halfspace=0.0,
    )  # fmt: skip
    rows = np.asarray(above)
    np.testing.assert_allclose(np.asarray(gx)[rows], np.tile([2.0, -0.5], (rows.sum(), 1)),
                               atol=1e-6)  # fmt: skip
    np.testing.assert_allclose(np.asarray(gv)[rows], np.tile([0.0, 0.3, 0.0], (rows.sum(), 1)),
                               atol=1e-6)  # fmt: skip
    gx0, _ = ensemble_device.node_gradients(
        x, v, alive, lam, grid, 256, scale, 48, pool=1, direction=direction, queries=queries
    )
    assert np.max(np.abs(np.asarray(gx0)[rows] - [2.0, -0.5])) > 0.1
