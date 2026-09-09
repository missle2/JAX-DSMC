"""CLAIM 3: do common random numbers give common DECISIONS?
Counts decision flips between theta*(1-h) and theta*(1+h) for a 1D-spatial DSMC step
sequence, and for the 0D pair_perm oracle configuration."""
import jax, jax.numpy as jnp, numpy as np, functools
jax.config.update("jax_enable_x64", True)
EPS=1e-30
C, NP, M = 8, 256, 6
OMEGA = 0.75
AMAX  = 24
LO, HI = 0.0, 1.0
DT = 2.0e-4
SIG = 300.0

def safe_pow(x,p):
    ok=x>EPS; return jnp.where(ok, jnp.exp(p*jnp.log(jnp.where(ok,x,1.0))),0.0)
def safe_sqrt(x):
    ok=x>EPS; return jnp.where(ok, jnp.sqrt(jnp.where(ok,x,1.0)),0.0)
def safe_log(p): return jnp.log(jnp.maximum(p,EPS))

BASE = jax.random.PRNGKey(2024)
Z  = jax.random.normal(jax.random.fold_in(BASE,1), (NP,3))     # frozen normal draws
X0 = jax.random.uniform(jax.random.fold_in(BASE,2), (NP,))      # frozen positions in [0,1)

def cell_of(x, lo, hi):
    dx=(hi-lo)/C
    return jnp.clip(jnp.floor((x-lo)/dx),0,C-1).astype(jnp.int32)

def run(temp, lo, hi, dt, Sigma, record=True):
    """returns (J, decisions dict)"""
    v = Z*jnp.sqrt(temp)*SIG/np.sqrt(300.0)
    x = lo + (hi-lo)*X0
    recs = {'cell':[], 'counts':[], 'natt':[], 'pi':[], 'pj':[], 'acc':[]}
    remain = jnp.full((C,), 0.5)
    for s in range(M):
        x = x + v[:,0]*dt
        x = lo + jnp.mod(x-lo, hi-lo)                       # periodic
        c = cell_of(x, lo, hi)
        order = jnp.argsort(c, stable=True)
        cs = jnp.sort(c)
        counts = jnp.sum(jax.nn.one_hot(c, C, dtype=jnp.float64), axis=0)
        starts = jnp.concatenate([jnp.zeros(1), jnp.cumsum(counts)[:-1]]).astype(jnp.int32)
        vol = (hi-lo)/C
        f = 0.5*counts*(counts-1.0)*Sigma*dt*1e18/vol + remain
        n = jnp.floor(f).astype(jnp.int32); remain = f-jnp.floor(f)
        cnt_i = counts.astype(jnp.int32)
        recs['cell'].append(c); recs['counts'].append(cnt_i); recs['natt'].append(n)
        pis=[];pjs=[];accs=[]
        for a in range(AMAX):
            ka = jax.random.fold_in(jax.random.fold_in(BASE,100+s), a)
            k1,k2,k3,k4 = jax.random.split(ka,4)
            npc = jnp.maximum(cnt_i,1)
            i = jax.random.randint(k1,(C,),0,npc)
            r = jax.random.randint(k2,(C,),0,jnp.maximum(npc-1,1))
            j = (i+1+r) % npc
            pi = order[jnp.clip(starts+i,0,NP-1)]
            pj = order[jnp.clip(starts+j,0,NP-1)]
            active = (a<n)&(cnt_i>=2)
            vi,vj = v[pi], v[pj]
            q = safe_pow(jnp.sum((vi-vj)**2,axis=-1), 1.0-OMEGA)*1e-16
            p = jnp.clip(q/jax.lax.stop_gradient(Sigma),0.0,1.0)
            u = jax.random.uniform(k3,(C,))
            acc = active & (u<p)
            # isotropic scatter (alpha=1)
            kk1,kk2 = jax.random.split(k4)
            cX = 2*jax.random.uniform(kk1,(C,))-1; sX=safe_sqrt(1-cX*cX)
            ep = 2*jnp.pi*jax.random.uniform(kk2,(C,))
            vr = vi-vj; vrm = safe_sqrt(jnp.sum(vr*vr,axis=-1))
            vrn = vrm[:,None]*jnp.stack([cX, sX*jnp.cos(ep), sX*jnp.sin(ep)],axis=1)
            vcm = 0.5*(vi+vj)
            vin = vcm+0.5*vrn; vjn = vcm-0.5*vrn
            v = v.at[pi].set(jnp.where(acc[:,None], vin, vi))
            v = v.at[pj].set(jnp.where(acc[:,None], vjn, vj))
            pis.append(jnp.where(active,pi,-1)); pjs.append(jnp.where(active,pj,-1)); accs.append(acc)
        recs['pi'].append(jnp.stack(pis)); recs['pj'].append(jnp.stack(pjs)); recs['acc'].append(jnp.stack(accs))
    J = jnp.mean(v[:,0]**2)/SIG**2
    d = {k: np.asarray(jnp.stack(vv)) for k,vv in recs.items()}
    return float(J), d

def flips(a,b):
    return {k: int(np.sum(a[k]!=b[k])) for k in a}

print("="*90)
print("CLAIM 3(a-b): 1-D spatial, C=%d cells, N=%d, M=%d steps. Perturb `temp` by relative h."%(C,NP,M))
print("   Same base_key for both evaluations (common random numbers).")
print("="*90)
T0=300.0
J0,d0 = run(T0, LO, HI, DT, 1.0)
Jm,dm = run(T0, LO, HI, DT, 1.0)  # sanity: identical
assert flips(d0,dm)=={k:0 for k in d0}
print(f"{'h(rel)':>10} {'cell':>7} {'counts':>7} {'n_att':>7} {'pair_i':>8} {'pair_j':>8} {'accept':>8} "
      f"{'FD dJ/dT':>14} {'AD dJ/dT':>14} {'rel err':>10}")
gAD = float(jax.grad(lambda T: run.__wrapped__(T) if False else 0.0)(0.0)) if False else None
# AD gradient
def Jonly(T): 
    v = Z*jnp.sqrt(T)*SIG/np.sqrt(300.0); x = LO+(HI-LO)*X0
    return None
# use jax.grad on a jitted differentiable version: reuse run but return J traced
def Jtrace(T):
    return _run_J(T)
def _run_J(temp):
    Sigma=1.0; lo,hi,dt = LO,HI,DT
    v = Z*jnp.sqrt(temp)*SIG/np.sqrt(300.0); x = lo+(hi-lo)*X0
    remain = jnp.full((C,),0.5)
    for s in range(M):
        x = x+v[:,0]*dt; x = lo+jnp.mod(x-lo,hi-lo)
        c = cell_of(x,lo,hi); order=jnp.argsort(c,stable=True)
        counts = jnp.sum(jax.nn.one_hot(c,C,dtype=jnp.float64),axis=0)
        starts = jnp.concatenate([jnp.zeros(1),jnp.cumsum(counts)[:-1]]).astype(jnp.int32)
        vol=(hi-lo)/C
        f = 0.5*counts*(counts-1.0)*Sigma*dt*1e18/vol+remain
        n = jnp.floor(f).astype(jnp.int32); remain=f-jnp.floor(f)
        cnt_i=counts.astype(jnp.int32)
        for a in range(AMAX):
            ka=jax.random.fold_in(jax.random.fold_in(BASE,100+s),a)
            k1,k2,k3,k4=jax.random.split(ka,4)
            npc=jnp.maximum(cnt_i,1)
            i=jax.random.randint(k1,(C,),0,npc); r=jax.random.randint(k2,(C,),0,jnp.maximum(npc-1,1))
            j=(i+1+r)%npc
            pi=order[jnp.clip(starts+i,0,NP-1)]; pj=order[jnp.clip(starts+j,0,NP-1)]
            active=(a<n)&(cnt_i>=2); vi,vj=v[pi],v[pj]
            q=safe_pow(jnp.sum((vi-vj)**2,axis=-1),1.0-OMEGA)*1e-16
            p=jnp.clip(q/jax.lax.stop_gradient(Sigma),0.0,1.0)
            u=jax.random.uniform(k3,(C,)); acc=active&(u<p)
            kk1,kk2=jax.random.split(k4)
            cX=2*jax.random.uniform(kk1,(C,))-1; sX=safe_sqrt(1-cX*cX); ep=2*jnp.pi*jax.random.uniform(kk2,(C,))
            vr=vi-vj; vrm=safe_sqrt(jnp.sum(vr*vr,axis=-1))
            vrn=vrm[:,None]*jnp.stack([cX,sX*jnp.cos(ep),sX*jnp.sin(ep)],axis=1)
            vcm=0.5*(vi+vj); vin=vcm+0.5*vrn; vjn=vcm-0.5*vrn
            v=v.at[pi].set(jnp.where(acc[:,None],vin,vi)); v=v.at[pj].set(jnp.where(acc[:,None],vjn,vj))
    return jnp.mean(v[:,0]**2)/SIG**2
gAD = float(jax.grad(_run_J)(T0))
for e in range(1,15):
    h=10.0**(-e)
    Jp,dp = run(T0*(1+h),LO,HI,DT,1.0)
    Jn,dn = run(T0*(1-h),LO,HI,DT,1.0)
    fl = flips(dp,dn)
    fd = (Jp-Jn)/(2*h*T0)
    print(f"{h:10.1e} {fl['cell']:7d} {fl['counts']:7d} {fl['natt']:7d} {fl['pi']:8d} {fl['pj']:8d} "
          f"{fl['acc']:8d} {fd:14.6e} {gAD:14.6e} {abs(fd-gAD)/abs(gAD):10.3e}")
