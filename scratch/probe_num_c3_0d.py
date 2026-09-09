"""CLAIM 3(c): 0-D, fixed N, pair_perm -- do decisions flip? Is FD valid?
Two configs: Oracle 1 (Sigma == q const, p==1, no accept randomness)
             Oracle 2 (omega != 1, genuine majorant, accept bits can flip)."""
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
EPS=1e-30
N, M = 64, 8
SIG=300.0
BASE=jax.random.PRNGKey(77)
Z=jax.random.normal(jax.random.fold_in(BASE,1),(N,3))

def safe_pow(x,p):
    ok=x>EPS; return jnp.where(ok,jnp.exp(p*jnp.log(jnp.where(ok,x,1.0))),0.0)
def safe_sqrt(x):
    ok=x>EPS; return jnp.where(ok,jnp.sqrt(jnp.where(ok,x,1.0)),0.0)

def run0d(temp, omega, Sigma, prefac, record=False):
    v = Z*jnp.sqrt(temp)*SIG/np.sqrt(300.0)
    recs={'perm':[], 'acc':[]}
    for s in range(M):
        k = jax.random.fold_in(BASE, 500+s)
        kp,ka,ks = jax.random.split(k,3)
        perm = jax.random.permutation(kp, N)          # depends only on key + static N
        i,j = perm[0::2], perm[1::2]
        vi,vj = v[i],v[j]
        q = prefac*safe_pow(jnp.sum((vi-vj)**2,axis=-1),1.0-omega)
        p = jnp.clip(q/jax.lax.stop_gradient(Sigma),0.0,1.0)
        u = jax.random.uniform(ka,(N//2,))
        acc = u < p
        k1,k2 = jax.random.split(ks)
        cX=2*jax.random.uniform(k1,(N//2,))-1; sX=safe_sqrt(1-cX*cX)
        ep=2*jnp.pi*jax.random.uniform(k2,(N//2,))
        vr=vi-vj; vrm=safe_sqrt(jnp.sum(vr*vr,axis=-1))
        vrn=vrm[:,None]*jnp.stack([cX,sX*jnp.cos(ep),sX*jnp.sin(ep)],axis=1)
        vcm=0.5*(vi+vj)
        v=v.at[i].set(jnp.where(acc[:,None],vcm+0.5*vrn,vi))
        v=v.at[j].set(jnp.where(acc[:,None],vcm-0.5*vrn,vj))
        if record: recs['perm'].append(perm); recs['acc'].append(acc)
    J=jnp.mean(v[:,0]**2)/SIG**2
    if record:
        return float(J), {k:np.asarray(jnp.stack(vv)) for k,vv in recs.items()}
    return J

for name, omega, Smult in [("Oracle 1: omega=1, Sigma=q_const (p==1 always)", 1.0, None),
                           ("Oracle 2: omega=0.75, Sigma = 2 x max q",        0.75, 2.0),
                           ("Oracle 2': omega=0.75, Sigma = 1.05 x max q",    0.75, 1.05)]:
    T0=300.0
    v0=Z*SIG
    i0,j0=np.triu_indices(N,1)
    if omega==1.0:
        prefac=1.0; Sigma=1.0
    else:
        prefac=1.0
        qq=np.asarray(safe_pow(jnp.sum((v0[i0]-v0[j0])**2,axis=-1),1.0-omega))
        Sigma=float(qq.max())*Smult
    _,dref = run0d(T0,omega,Sigma,prefac,record=True)
    gAD=float(jax.grad(lambda T: run0d(T,omega,Sigma,prefac))(T0))
    print("="*88); print(name); print("="*88)
    print(f"  mean acceptance p = {float(np.mean(dref['acc'])):.4f}")
    print(f"{'h(rel)':>10} {'perm flips':>11} {'accept flips':>13} {'FD dJ/dT':>15} {'AD dJ/dT':>15} {'rel err':>11}")
    for e in range(1,15):
        h=10.0**(-e)
        Jp,dp=run0d(T0*(1+h),omega,Sigma,prefac,record=True)
        Jn,dn=run0d(T0*(1-h),omega,Sigma,prefac,record=True)
        pf=int(np.sum(dp['perm']!=dn['perm'])); af=int(np.sum(dp['acc']!=dn['acc']))
        fd=(Jp-Jn)/(2*h*T0)
        print(f"{h:10.1e} {pf:11d} {af:13d} {fd:15.7e} {gAD:15.7e} {abs(fd-gAD)/abs(gAD):11.3e}")
    print()

# statistics of accept-bit flips over many random configurations
print("="*88)
print("How often does an accept bit flip in 0-D, as a function of h? (200 random seeds)")
print("="*88)
omega=0.75
def _mk(sd):
    global BASE, Z
    BASE=jax.random.PRNGKey(1000+sd); Z=jax.random.normal(jax.random.fold_in(BASE,1),(N,3))
for e in [2,4,6,8,10,12]:
    h=10.0**(-e); nf=0; tot=0
    for sd in range(200):

        _mk(sd)
        v0=Z*SIG; i0,j0=np.triu_indices(N,1)
        qq=np.asarray(safe_pow(jnp.sum((v0[i0]-v0[j0])**2,axis=-1),1.0-omega))
        S=float(qq.max())*2.0
        _,dp=run0d(300.0*(1+h),omega,S,1.0,record=True)
        _,dn=run0d(300.0*(1-h),omega,S,1.0,record=True)
        nf+=int(np.sum(dp['acc']!=dn['acc'])); tot+=dp['acc'].size
    print(f"  h={h:8.1e}: {nf:5d} accept-bit flips out of {tot} bits  ({nf/tot:.3e} per bit)"
          f"  -> seeds with >=1 flip: {'yes' if nf else 'none'}")
