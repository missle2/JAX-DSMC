"""CLAIM 1 (b)/(c): is the huge cotangent harmful? and is EPS_D dimensional?"""
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
EPS, EPS_D = 1e-30, 1e-6

def safe_sqrt(x, eps=EPS):
    ok = x > eps; return jnp.where(ok, jnp.sqrt(jnp.where(ok, x, 1.0)), 0.0)
def safe_div(n,d,eps=EPS,fb=0.0):
    ok = jnp.abs(d) > eps; return jnp.where(ok, n/jnp.where(ok,d,1.0), fb)
def safe_norm(v,eps=EPS): return safe_sqrt(jnp.sum(v*v),eps)
def safe_pow(x,p,eps=EPS):
    ok = x > eps; return jnp.where(ok, jnp.exp(p*jnp.log(jnp.where(ok,x,1.0))), 0.0)

# ---------------------------------------------------------------- HARM DEMO
print("="*78); print("1B. HARM: one degenerate cotangent summed with N normal ones")
print("="*78)
# objective: sum_k q_k  with q = |vr|^(2-2omega); grad wrt every velocity component.
def make(omega, N, vr_deg):
    key = jax.random.PRNGKey(0)
    vr = jax.random.normal(key,(N,3))*340.0    # SI thermal
    vr = vr.at[0].set(jnp.array([vr_deg,0.,0.]))
    return vr
def J(vr, omega): return jnp.sum(safe_pow(jnp.sum(vr*vr,axis=1), 1.0-omega))

for omega in [0.75, 1.2]:
    print(f"\n omega = {omega}")
    print(f"{'|vr_0|':>10} {'|g_0|':>13} {'|g_rest| rms':>14} {'ratio':>11} {'sum(g) rel err':>16}")
    for e in [0, -2, -4, -6, -8, -10, -14]:
        vr = make(omega, 4096, 10.0**e)
        g = jax.grad(J)(vr, omega)
        g0 = float(jnp.linalg.norm(g[0])); grest = float(jnp.sqrt(jnp.mean(jnp.sum(g[1:]**2,axis=1))))
        tot = jnp.sum(g)
        # reference: exclude the degenerate particle, computed in f64 both ways
        tot_ex = jnp.sum(g[1:])
        # catastrophic cancellation check: recompute sum in higher-precision-ish order
        import numpy as _np
        exact = _np.longdouble(0.0)
        for val in _np.asarray(g, dtype=_np.float64).ravel(): exact += _np.longdouble(val)
        rel = abs(float(tot) - float(exact))/max(abs(float(exact)),1e-300)
        print(f"{10.0**e:10.1e} {g0:13.4e} {grest:14.4e} {g0/grest:11.3e} {rel:16.3e}")

print("\n --- how often does |vr| get that small? Maxwellian tail (sigma=340 m/s) ---")
key = jax.random.PRNGKey(1)
sig = 340.0
for r in [1e-1, 1e-2, 1e-4, 1e-6, 1e-8]:
    # P(|vr|<r) for 3D gaussian, small-r: (4/3)pi r^3 /(2 pi sig^2)^{3/2}
    p = (4/3)*np.pi*r**3/((2*np.pi*sig**2)**1.5)
    print(f"   P(|vr| < {r:8.1e} m/s) = {p:.3e}   -> 1 event per {1/p:.3e} pairs")

# ------------------------------------------------------------ EPS_D SCALING
print()
print("="*78); print("1C. EPS_D = 1e-6 ABSOLUTE: does the degenerate branch fire differently")
print("     under nondimensionalisation?")
print("="*78)

def scatter_vr(vr, cosX, sinX, eps_g, rel=False, EPSD=EPS_D):
    vrm = safe_sqrt(jnp.sum(vr*vr))
    d = safe_norm(vr[1:3])
    thr = EPSD*vrm if rel else EPSD
    ok = d > thr
    d_safe = jnp.where(ok, d, 1.0)
    ua_g = cosX*vr[0] + sinX*d*jnp.sin(eps_g)
    vb_g = cosX*vr[1] + sinX*(vrm*vr[2]*jnp.cos(eps_g) - vr[0]*vr[1]*jnp.sin(eps_g))/d_safe
    wc_g = cosX*vr[2] - sinX*(vrm*vr[1]*jnp.cos(eps_g) + vr[0]*vr[2]*jnp.sin(eps_g))/d_safe
    ua_d = cosX*vr[0]; vb_d = sinX*vr[0]*jnp.cos(eps_g); wc_d = sinX*vr[0]*jnp.sin(eps_g)
    return jnp.stack([jnp.where(ok,ua_g,ua_d), jnp.where(ok,vb_g,vb_d), jnp.where(ok,wc_g,wc_d)])

cosX, sinX, epsg = 0.3, np.sqrt(1-0.09), 1.1
print("Same PHYSICAL configuration, expressed in 3 unit systems.")
print("vr = (V, theta*V, 0) i.e. d/|vr| = theta exactly. Vary theta.")
print(f"{'d/|vr|':>10} | {'SI V=340':>26} | {'nondim V=1':>26} | {'CGS V=3.4e4':>26}")
print(f"{'':>10} | {'d':>12}{'branch':>8}{'':>6} | {'d':>12}{'branch':>8}{'':>6} | {'d':>12}{'branch':>8}{'':>6}")
for th in [1e-2, 1e-4, 1e-6, 1e-8, 1e-9, 1e-10, 1e-12]:
    row = f"{th:10.1e} |"
    for V,label in [(340.0,'SI'),(1.0,'nd'),(3.4e4,'cgs')]:
        vr = jnp.array([V, th*V, 0.0])
        d = float(safe_norm(vr[1:3]))
        br = "GENERAL" if d > EPS_D else "DEGEN  "
        row += f" {d:12.3e} {br:>8}      |"
    print(row)

print("\n Now the RELATIVE form d > EPS_D*vrm  (same table):")
print(f"{'d/|vr|':>10} | {'SI':>10} | {'nondim':>10} | {'CGS':>10}")
for th in [1e-2, 1e-4, 1e-6, 1e-8, 1e-9, 1e-10, 1e-12]:
    row = f"{th:10.1e} |"
    for V in [340.0, 1.0, 3.4e4]:
        vr = jnp.array([V, th*V, 0.0])
        vrm = float(safe_sqrt(jnp.sum(vr*vr))); d = float(safe_norm(vr[1:3]))
        row += f" {'GENERAL' if d > EPS_D*vrm else 'DEGEN':>10} |"
    print(row)

print("\n Scale-invariance of the OUTPUT (vr_new/V should be identical across V):")
for rel in [False, True]:
    print(f"  relative={rel}")
    for th in [1e-2, 1e-7, 1e-9]:
        outs=[]
        for V in [1.0, 340.0, 3.4e4]:
            vr = jnp.array([V, th*V, 0.0])
            o = np.asarray(scatter_vr(vr, cosX, sinX, epsg, rel=rel))/V
            outs.append(o)
        spread = max(float(np.max(np.abs(outs[i]-outs[0]))) for i in range(3))
        print(f"    d/|vr|={th:8.1e}  max |vr_new/V - ref| across unit systems = {spread:.4e}")

print()
print("="*78); print("1C-bis. Jacobian norm of scatter vs d (the real cost of an ABSOLUTE eps)")
print("="*78)
Jf = jax.jacobian(lambda vr: scatter_vr(vr, cosX, sinX, epsg, rel=False))
Jr = jax.jacobian(lambda vr: scatter_vr(vr, cosX, sinX, epsg, rel=True))
print(f"{'V':>10} {'d/|vr|':>10} {'d':>11} {'||J||_abs':>13} {'||J||_rel':>13}")
for V in [1.0, 340.0]:
    for th in [1e-1, 1e-3, 1e-5, 1e-7, 1e-8, 1e-9]:
        vr = jnp.array([V, th*V, 0.0])
        d = float(safe_norm(vr[1:3]))
        na = float(jnp.linalg.norm(Jf(vr))); nr = float(jnp.linalg.norm(Jr(vr)))
        print(f"{V:10.1f} {th:10.1e} {d:11.3e} {na:13.4e} {nr:13.4e}")
