"""Which parameters reach J pathwise, and which only through the score?

Answering this decides how the shock verification has to be run: a fixed-key
finite difference can only validate a parameter that has a pathwise route.
"""
import sys
sys.path.insert(0, ".")
import functools
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from jax_dsmc import adjoint
from jax_dsmc.boundary import BC
from jax_dsmc.collide import species_params, vremax_init
from jax_dsmc.grid import num_cells
from jax_dsmc.initialize import initial_state
from jax_dsmc.state import (CollideState, FlowParams, GridSpec, SimParams,
                            SimStatic, SpeciesParams)
from examples.shock.normal import AR_MASS, AR_DIAM, AR_OMEGA, AR_TREF, AR_ALPHA, mfp_vss

NRHO, TEMP, MACH = 1e21, AR_TREF, 2.0
K = 1.380649e-23


def build(theta, dim, n_x, npc, n_steps, a_max):
    lam = mfp_vss(NRHO)
    length = 40.0 * lam
    n = (n_x,) + (2,) * (dim - 1)
    dx = length / n_x
    trans = dx * 2
    hi = jnp.array([length] + [trans] * (dim - 1))
    grid = GridSpec(lo=jnp.zeros(dim), hi=hi, n=n,
                    transverse=jnp.float64(trans ** (3 - dim)))
    C = num_cells(grid)
    mass = jnp.array([AR_MASS])
    sp = species_params(mass,
                        jnp.array([[theta["diam"]]]), jnp.array([[theta["omega"]]]),
                        jnp.array([[theta["tref"]]]), jnp.array([[theta["alpha"]]]))
    npart = C * npc
    volume = length * trans ** 2
    fnum = jnp.full((C,), theta["nrho"] * volume / npart)
    vth = jnp.sqrt(K * TEMP / AR_MASS)
    dt = 0.15 * dx / jnp.sqrt(K * TEMP / AR_MASS)
    static = SimStatic(N=npart, D=dim, S=1, G=1, A_max=a_max, K_bounce=1, n_steps=n_steps)
    params = SimParams(dt=jnp.float64(dt), fnum=fnum, epsilon=jnp.float64(0.0))
    flow = FlowParams(nrho=theta["nrho"], temp=theta["temp"],
                      vstream=jnp.array([theta["ux"], 0.0, 0.0]))
    st = initial_state(jax.random.PRNGKey(0), flow, grid, sp, static)
    cs = CollideState(vremax=vremax_init(jnp.full((C,), TEMP), sp.omega[0, 0], sp.tref[0, 0],
                                         sp.mr[0, 0], sp.prefactor[0, 0], C, 1))
    return st, cs, grid, sp, params, static, length


def J(theta, key, dim, scored, n_x=16, npc=60, n_steps=12, a_max=200):
    st, cs, grid, sp, params, static, length = build(theta, dim, n_x, npc, n_steps, a_max)
    phi = functools.partial(adjoint.phi_windowed, inner=adjoint.phi_kinetic,
                            axis=0, center=0.25 * length, width=0.15 * length)
    j, _ = adjoint.terminal_objective(st, cs, grid, sp, params, static, key, phi,
                                      rho=1.0, bc=BC.SPECULAR, baseline="mean",
                                      scored=scored)
    return j


THETA = dict(nrho=jnp.float64(NRHO), temp=jnp.float64(TEMP), ux=jnp.float64(-600.0),
             diam=jnp.float64(AR_DIAM), omega=jnp.float64(AR_OMEGA),
             tref=jnp.float64(AR_TREF), alpha=jnp.float64(AR_ALPHA))

for dim in (1, 2, 3):
    key = jax.random.PRNGKey(7)
    g_path = jax.grad(J)(THETA, key, dim, False)
    g_full = jax.grad(J)(THETA, key, dim, True)
    val = J(THETA, key, dim, True)
    print(f"\n--- D = {dim}   J = {val:.6e}")
    print(f"{'param':>8} {'pathwise':>14} {'full (path+score)':>20} {'score share':>12}")
    for k in THETA:
        p, f = float(g_path[k]), float(g_full[k])
        share = "n/a" if f == 0 else f"{(f - p) / f:8.3f}"
        print(f"{k:>8} {p:14.6e} {f:20.6e} {share:>12}")
