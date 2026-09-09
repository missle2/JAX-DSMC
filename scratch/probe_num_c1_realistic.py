"""CLAIM 1: is the huge cotangent reachable in REALISTIC DSMC use?"""
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
EPS=1e-30
def safe_pow(x,p,eps=EPS):
    ok = x > eps; return jnp.where(ok, jnp.exp(p*jnp.log(jnp.where(ok,x,1.0))), 0.0)
def safe_sqrt(x,eps=EPS):
    ok = x > eps; return jnp.where(ok, jnp.sqrt(jnp.where(ok,x,1.0)), 0.0)

print("="*78)
print("Scenario A: COLD BEAM / duplicated parcels -- v_i and v_j nominally equal,")
print("            differing only by f64 round-off from the advection arithmetic.")
print("="*78)
V = 340.0
for omega in [0.5, 0.75, 1.0, 1.2]:
    # jitter at the f64 relative level
    for jit in [0.0, 1e-16, 1e-13, 1e-10]:
        vi = jnp.array([V,0.,0.]); vj = jnp.array([V*(1+jit),0.,0.])
        f = lambda vi,vj: safe_pow(jnp.sum((vi-vj)**2), 1.0-omega)
        g = jax.grad(f)(vi,vj)
        print(f"  omega={omega:4.2f} rel-jitter={jit:8.1e}  |vr|={float(jnp.linalg.norm(vi-vj)):9.2e}"
              f"  q={float(f(vi,vj)):11.4e}  |dq/dvi|={float(jnp.linalg.norm(g)):11.4e}")
    print()

print("="*78)
print("Scenario B: how far into the tail does a 1e8-pair Maxwellian run actually go?")
print("="*78)
key = jax.random.PRNGKey(7); sig=340.0
mins=[]
for b in range(20):
    k=jax.random.fold_in(key,b)
    vr = jax.random.normal(k,(5_000_000,3))*sig*np.sqrt(2)
    mins.append(float(jnp.min(jnp.linalg.norm(vr,axis=1))))
mins=np.array(mins)
print(f"  100M sampled relative velocities, sigma_vr={sig*np.sqrt(2):.0f} m/s")
print(f"  min |vr| observed: {mins.min():.4e} m/s   (median over 20 batches {np.median(mins):.4e})")
print(f"  smallest vr2      : {mins.min()**2:.4e}  vs EPS=1e-30  -> ratio {mins.min()**2/1e-30:.3e}")
for omega in [0.75,1.2]:
    r=mins.min(); print(f"  worst-case |dq/dvr| at omega={omega}: {abs(2*(1-omega))*r**(1-2*omega):.4e}"
                        f"   (typical at |vr|=sigma: {abs(2*(1-omega))*(sig*1.414)**(1-2*omega):.4e})")

print()
print("="*78)
print("Physically-scaled eps proposal:  eps_v2 = (k T_ref/m) * 1e-12")
print("="*78)
kB=1.380649e-23
for name,m,T in [("Ar",6.63e-26,300.0),("N2",4.65e-26,300.0),("N2 cold",4.65e-26,10.0)]:
    s2 = kB*T/m
    for f in [1e-12, 1e-16]:
        e = s2*f
        print(f"  {name:8s} kT/m={s2:10.3e} m2/s2  eps={e:10.3e}  -> |vr|_cut={np.sqrt(e):9.3e} m/s"
              f" = {np.sqrt(f):8.1e} x thermal")
        for omega in [0.75,1.2]:
            print(f"       bound |dq/dvr| <= {abs(2*(1-omega))*e**(0.5-omega):.4e}  (omega={omega})")
    print()
print("For safe_sqrt itself the bound is |d/dx| <= 1/(2 sqrt(eps)):")
for e in [1e-30, 1e-20, 8.9e-8, 8.9e-12]:
    print(f"   eps={e:10.3e} -> 1/(2 sqrt eps) = {1/(2*np.sqrt(e)):.4e}")
