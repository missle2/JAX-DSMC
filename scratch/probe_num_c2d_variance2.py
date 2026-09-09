"""CLAIM 2(d): bias and VARIANCE of the DiCE score-function gradient vs the majorant Sigma.

0-D single cell, VSS/VHS, pair_ntc, DiCE surrogate exactly as spec 11.1.
Parameter theta: log-scale on the cross-section prefactor  (q = exp(theta) * q0).
Objective  J = mean_i v_x^2  after M steps, started anisotropic so collisions matter.
n_attempts is set by the physical NTC formula, so n grows LINEARLY with Sigma.
"""
import jax, jax.numpy as jnp, numpy as np, functools, sys
jax.config.update("jax_enable_x64", True)
EPS = 1e-30

N      = 48
M      = 4
OMEGA  = 0.75
ALPHA  = 1.0
SIGMA_THERM = 300.0     # velocity scale

def safe_log(p): return jnp.log(jnp.maximum(p, EPS))
def safe_sqrt(x):
    ok = x > EPS; return jnp.where(ok, jnp.sqrt(jnp.where(ok,x,1.0)), 0.0)
def safe_pow(x,p):
    ok = x > EPS; return jnp.where(ok, jnp.exp(p*jnp.log(jnp.where(ok,x,1.0))), 0.0)
def safe_norm(v,ax=-1): return safe_sqrt(jnp.sum(v*v,axis=ax))

def q_of(vi, vj, theta):
    vr2 = jnp.sum((vi-vj)**2, axis=-1)
    return jnp.exp(theta) * safe_pow(vr2, 1.0-OMEGA)

def scatter(vi, vj, key):
    ku, ke = jax.random.split(key)
    u_chi = jax.random.uniform(ku); eps = 2*jnp.pi*jax.random.uniform(ke)
    cosX = 2.0*safe_pow(u_chi, 1.0/ALPHA) - 1.0
    sinX = safe_sqrt(1.0-cosX*cosX)
    vr = vi-vj; vrm = safe_norm(vr); d = safe_norm(vr[1:3])
    ok = d > 1e-6*vrm            # relative form (scale-free); irrelevant here
    ds = jnp.where(ok, d, 1.0)
    ua = cosX*vr[0] + sinX*d*jnp.sin(eps)
    vb = cosX*vr[1] + sinX*(vrm*vr[2]*jnp.cos(eps) - vr[0]*vr[1]*jnp.sin(eps))/ds
    wc = cosX*vr[2] - sinX*(vrm*vr[1]*jnp.cos(eps) + vr[0]*vr[2]*jnp.sin(eps))/ds
    ua_d = cosX*vr[0]; vb_d = sinX*vr[0]*jnp.cos(eps); wc_d = sinX*vr[0]*jnp.sin(eps)
    vrn = jnp.stack([jnp.where(ok,ua,ua_d), jnp.where(ok,vb,vb_d), jnp.where(ok,wc,wc_d)])
    vcm = 0.5*(vi+vj)
    return vcm + 0.5*vrn, vcm - 0.5*vrn

def init_v(key):
    k1,k2,k3 = jax.random.split(key,3)
    vx = jax.random.normal(k1,(N,))*SIGMA_THERM*jnp.sqrt(2.0)   # Tx = 2T
    vy = jax.random.normal(k2,(N,))*SIGMA_THERM*jnp.sqrt(0.5)
    vz = jax.random.normal(k3,(N,))*SIGMA_THERM*jnp.sqrt(0.5)
    return jnp.stack([vx,vy,vz],axis=1)

def trajectory(theta, key, Sigma, nslots, natt):
    """natt: attempts per step (static int). Sigma: fixed majorant."""
    v0 = init_v(jax.random.fold_in(key, 999))
    def step(carry, s):
        v, logh = carry
        def slot(c, a):
            v, logh = c
            ka = jax.random.fold_in(jax.random.fold_in(key, 1000+s), a)
            k1,k2,k3,k4 = jax.random.split(ka,4)
            i = jax.random.randint(k1,(),0,N)
            r = jax.random.randint(k2,(),0,N-1)
            j = (i+1+r) % N
            vi, vj = v[i], v[j]
            q = q_of(vi,vj,theta)
            p = jnp.clip(q/jax.lax.stop_gradient(Sigma), 0.0, 1.0)
            u = jax.random.uniform(k3)
            acc = (a < natt) & (u < p)
            vin, vjn = scatter(vi,vj,k4)
            v = v.at[i].set(jnp.where(acc, vin, vi))
            v = v.at[j].set(jnp.where(acc, vjn, vj))
            lh = jnp.where(a < natt, jnp.where(acc, safe_log(p), safe_log(1.0-p)), 0.0)
            logh = logh.at[i].add(lh).at[j].add(lh)
            return (v, logh), None
        (v,logh),_ = jax.lax.scan(slot,(v,logh),jnp.arange(nslots))
        return (v,logh), None
    (v,logh),_ = jax.lax.scan(step,(v0,jnp.zeros(N)),jnp.arange(M))
    return v, logh

def surrogate(theta, key, Sigma, nslots, natt):
    v, logh = trajectory(theta, key, Sigma, nslots, natt)
    phi = (v[:,0]/SIGMA_THERM)**2                        # per-particle reward
    magic = jnp.exp(logh - jax.lax.stop_gradient(logh))
    return jnp.mean(magic*phi)

# ---- calibrate: base attempt count from the NTC formula with Sigma = Sigma_ref
key0 = jax.random.PRNGKey(0)
v0 = init_v(jax.random.fold_in(key0,999))
i,j = np.triu_indices(N,1)
qq = np.asarray(q_of(v0[i,:], v0[j,:], 0.0))
Sig_ref = float(qq.max())
print(f"max q over all pairs at t=0 : {Sig_ref:.6e}   mean q {qq.mean():.4e}")
NATT_BASE = 16                       # n at Sigma = Sig_ref ; n scales linearly with Sigma

RESULTS={}
mults = [0.7, 0.9, 1.0, 1.001, 1.01, 1.1, 1.5, 2.0, 5.0, 10.0, 30.0, 100.0]
gfun = jax.jit(jax.vmap(jax.grad(surrogate), in_axes=(None,0,None,None,None)),
               static_argnums=(3,4))
vfun = jax.jit(jax.vmap(surrogate, in_axes=(None,0,None,None,None)), static_argnums=(3,4))

def sweep(m, K, seed=12345):
    S = m*Sig_ref
    natt = max(1,int(round(NATT_BASE*m))); nslots = natt
    keys = jax.random.split(jax.random.PRNGKey(seed), K)
    chunk = max(4, int(250000//(natt*M)))
    gs=[]; js=[]
    for a in range(0,K,chunk):
        kk = keys[a:a+chunk]
        gs.append(np.asarray(gfun(0.0, kk, S, nslots, natt)))
        js.append(np.asarray(vfun(0.0, kk, S, nslots, natt)))
    return natt, np.concatenate(gs), np.concatenate(js)

print("\nN=%d particles, M=%d steps, omega=%s"%(N,M,OMEGA))
print("n_attempts/step = %d x (Sigma/Sigma_ref) -- physical NTC scaling\n"%NATT_BASE)
hdr = "%10s %6s %6s %10s %13s %10s %11s %11s %8s"%('Sig/max q','n_att','K','mean J','mean dJ/dth','sem','std(g)','var','viol %')
print(hdr); print("-"*len(hdr))
ref=None; out={}
for m in mults:
    K = 8192 if m<=10 else 2048
    natt, g, Jv = sweep(m,K)
    var=g.var(ddof=1); sem=g.std(ddof=1)/np.sqrt(len(g))
    viol = float((qq> m*Sig_ref).mean())*100
    out[m]=(g.mean(),var,sem,natt)
    print("%10.3f %6d %6d %10.6f %13.6e %10.3e %11.5e %11.4e %8.2f"%(
        m,natt,len(g),Jv.mean(),g.mean(),sem,g.std(ddof=1),var,viol))
ref=out[10.0][1]
print()
print("variance normalised to the Sigma=10x value:")
for m in mults:
    mm,vv,ss,na = out[m]
    print("  Sigma=%8.3fx  var=%11.4e  var/var_10x=%8.3f   mean=%+.5e +- %.2e"%(m,vv,vv/ref,mm,ss))
print()
print("--- bias: z-score of each mean vs the Sigma=10x reference (genuine majorant) ---")
m0,v0_,s0,_ = out[10.0]
for m in mults:
    mm,vv,ss,_ = out[m]
    z=(mm-m0)/np.sqrt(ss**2+s0**2)
    flag = "  <-- MAJORANT VIOLATED" if m<1.0 else ""
    print("  Sigma=%8.3fx mean=%+.6e delta=%+.3e z=%+7.2f%s"%(m,mm,mm-m0,z,flag))
