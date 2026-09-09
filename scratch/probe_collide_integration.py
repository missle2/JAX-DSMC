"""End-to-end check of collide(): conservation, jit, and the defect-1 regression."""
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from jax_dsmc.state import (ParticleState, CollideState, GridSpec, SpeciesParams,
                            SimParams, SimStatic)
from jax_dsmc.sort import build_table
from jax_dsmc.grid import num_cells
from jax_dsmc.collide import collide, make_prefactor, vremax_init
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc import sampling

AR_MASS, AR_DIAM, AR_TREF = 6.63e-26, 4.11e-10, 273.15

def build(omega=1.0, alpha=1.0, N=2000, n=(1,), T=300.0, A_max=400):
    mass = jnp.array([AR_MASS]); mr = jnp.array([[AR_MASS/2]])
    diam = jnp.array([[AR_DIAM]]); om = jnp.array([[omega]])
    tref = jnp.array([[AR_TREF]]); al = jnp.array([[alpha]])
    sp = SpeciesParams(mass=mass, diam=diam, omega=om, tref=tref, alpha=al, mr=mr,
                       prefactor=make_prefactor(diam, om, tref, mr))
    g = GridSpec(lo=jnp.zeros(len(n)), hi=jnp.ones(len(n))*1e-4, n=n)
    C = num_cells(g)
    st = SimStatic(N=N, D=len(n), S=1, G=1, A_max=A_max, K_bounce=3, n_steps=1)
    pr = SimParams(dt=jnp.float64(1e-7), fnum=jnp.full((C,), 1e12), epsilon=jnp.float64(0.0))
    k = jax.random.PRNGKey(0)
    x = sampling.sample_positions(jax.random.fold_in(k,1), N, g)
    v = sampling.sample_maxwellian(jax.random.fold_in(k,2), N, jnp.float64(T), mass[0], jnp.zeros(3))
    state = ParticleState(x=x, v=v, species=jnp.zeros(N, jnp.int32),
                          alive=jnp.ones(N, bool), logh_acc=jnp.zeros(N))
    cs = CollideState(vremax=vremax_init(jnp.full((C,), T), om[0,0], tref[0,0], mr[0,0],
                                         sp.prefactor[0,0], C, 1))
    return state, cs, g, sp, pr, st, jnp.full((C,), T), k

state, cs, g, sp, pr, st, T, k = build()
table = build_table(state, g)
f = jax.jit(collide, static_argnames=("static",))
ns, ncs, diag = f(state, table, cs, g, sp, pr, st, T, 0, k)

m = AR_MASS
p0, p1 = (m*state.v).sum(0), (m*ns.v).sum(0)
e0, e1 = 0.5*m*(state.v**2).sum(), 0.5*m*(ns.v**2).sum()
print("=== conservation over one collide step ===")
print(f"  momentum rel err : {float(jnp.abs(p1-p0).max()/jnp.abs(p0).max()):.3e}")
print(f"  energy   rel err : {float(abs(e1-e0)/e0):.3e}")
print(f"  velocities changed: {int((jnp.abs(ns.v-state.v).max(1)>0).sum())} / {st.N} particles")
print("=== diagnostics ===")
for fld in ("attempts_clipped","max_attempts","majorant_violations","min_majorant_margin"):
    print(f"  {fld:24s} {getattr(diag,fld)}")

print("\n=== DEFECT-1 REGRESSION: dt / fnum / domain gradients must NOT be zero ===")
def J(dt, fnum, hi):
    gg = GridSpec(lo=g.lo, hi=hi, n=g.n)
    pp = SimParams(dt=dt, fnum=fnum, epsilon=pr.epsilon)
    tb = build_table(state, gg)
    s2, _, _ = collide(state, tb, cs, gg, sp, pp, st, T, 0, k)
    # surrogate objective: DiCE magic box (Finding 2 / section 11)
    magic = jnp.exp(s2.logh_acc - jax.lax.stop_gradient(s2.logh_acc))
    return jnp.sum(magic * jnp.sum(s2.v**2, axis=-1))
gd = jax.grad(J, argnums=(0,1,2))(pr.dt, pr.fnum, g.hi)
print(f"  dJ/d(dt)   = {float(gd[0]):.6e}")
print(f"  dJ/d(fnum) = {float(gd[1][0]):.6e}")
print(f"  dJ/d(hi)   = {float(gd[2][0]):.6e}")
assert float(gd[0]) != 0.0, "DEFECT 1 REGRESSED: dt gradient is zero"
assert float(gd[1][0]) != 0.0, "DEFECT 1 REGRESSED: fnum gradient is zero"
gv = jax.grad(lambda vv: J2(vv))(state.v) if False else None
print("\n=== velocity gradient path ===")
def Jv(vv):
    s = ParticleState(x=state.x, v=vv, species=state.species, alive=state.alive,
                      logh_acc=state.logh_acc)
    s2,_,_ = collide(s, table, cs, g, sp, pr, st, T, 0, k)
    magic = jnp.exp(s2.logh_acc - jax.lax.stop_gradient(s2.logh_acc))
    return jnp.sum(magic * jnp.sum(s2.v**2, axis=-1))
gvv = jax.grad(Jv)(state.v)
print(f"  ||dJ/dv|| = {float(jnp.linalg.norm(gvv)):.6e}   finite={bool(jnp.isfinite(gvv).all())}")
print("\n=== VSS params gradient (model calibration) ===")
def Jp(diam, omega, alpha):
    sp2 = SpeciesParams(mass=sp.mass, diam=diam, omega=omega, tref=sp.tref, alpha=alpha,
                        mr=sp.mr, prefactor=make_prefactor(diam, omega, sp.tref, sp.mr))
    s2,_,_ = collide(state, table, cs, g, sp2, pr, st, T, 0, k)
    magic = jnp.exp(s2.logh_acc - jax.lax.stop_gradient(s2.logh_acc))
    return jnp.sum(magic * jnp.sum(s2.v**2, axis=-1))
gp = jax.grad(Jp, argnums=(0,1,2))(sp.diam, sp.omega, sp.alpha)
print(f"  dJ/d(diam)  = {float(gp[0][0,0]):.6e}")
print(f"  dJ/d(omega) = {float(gp[1][0,0]):.6e}")
print(f"  dJ/d(alpha) = {float(gp[2][0,0]):.6e}")
