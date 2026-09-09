"""CLAIM 3: how does the zero-decision-flip FD window scale with N, C, M?
Bisect for h_crit = largest relative h with ZERO decision flips."""
import jax, jax.numpy as jnp, numpy as np, functools
jax.config.update("jax_enable_x64", True)
LO,HI=0.0,1.0; SIG=300.0; DT=2.0e-4

@functools.partial(jax.jit, static_argnums=(1,2,3,4))
def decisions(temp, NP, C, M, seed):
    B=jax.random.PRNGKey(seed)
    Z =jax.random.normal(jax.random.fold_in(B,1),(NP,3))
    X0=jax.random.uniform(jax.random.fold_in(B,2),(NP,))
    v = Z*jnp.sqrt(temp)*SIG/np.sqrt(300.0)
    x = LO+(HI-LO)*X0
    remain=jnp.full((C,),0.5); dx=(HI-LO)/C
    cells=[]; natts=[]
    for s in range(M):
        x=x+v[:,0]*DT; x=LO+jnp.mod(x-LO,HI-LO)
        c=jnp.clip(jnp.floor((x-LO)/dx),0,C-1).astype(jnp.int32)
        counts=jnp.sum(jax.nn.one_hot(c,C,dtype=jnp.float64),axis=0)
        f=0.5*counts*(counts-1.0)*1.0*DT*2.0/dx + remain   # calibrated so n = O(few)
        n=jnp.floor(f).astype(jnp.int32); remain=f-jnp.floor(f)
        cells.append(c); natts.append(n)
    return jnp.stack(cells), jnp.stack(natts)

def nflip(h, NP, C, M, seed, T0=300.0):
    cp,np_=decisions(T0*(1+h),NP,C,M,seed); cn,nn=decisions(T0*(1-h),NP,C,M,seed)
    return (int(jnp.sum(cp!=cn)), int(jnp.sum(np_!=nn)))

def hcrit(NP,C,M,seed):
    lo,hi=1e-16,1e-1
    if sum(nflip(hi,NP,C,M,seed))==0: return hi
    for _ in range(60):
        mid=np.sqrt(lo*hi)
        if sum(nflip(mid,NP,C,M,seed))==0: lo=mid
        else: hi=mid
        if hi/lo<1.02: break
    return lo

print("="*96)
print("Largest relative FD step h with ZERO decision flips (cell membership + n_attempts),")
print("perturbing `temp`, 1-D periodic, M steps of move+bin.  Median over 5 seeds.")
print("="*96)
print("%8s %6s %5s %14s %14s %14s"%("N","C","M","h_crit(median)","h_crit(min)","cell/natt split"))
for NP,C,M in [(64,8,6),(256,8,6),(1024,8,6),(4096,8,6),(16384,8,6),(65536,8,6),
               (1024,8,1),(1024,8,24),(1024,8,96),
               (1024,32,6),(1024,128,6),(4096,128,6)]:
    hs=[hcrit(NP,C,M,s) for s in range(5)]
    hm=float(np.median(hs))
    # which channel breaks first, just above h_crit
    cf,nf = nflip(hm*1.5,NP,C,M,0)
    print("%8d %6d %5d %14.3e %14.3e   cell=%d natt=%d"%(NP,C,M,hm,min(hs),cf,nf))

print()
print("="*96)
print("Practical FD window:  round-off floor gives central-FD relative error ~ 1e-16/(h*|J'|/|J|).")
print("With J = O(1) and dJ/dtheta * theta = O(1e-3..1), the FD estimate is round-off-dominated")
print("below h ~ 1e-10..1e-9 (measured in probe_num_c3_fd2.py: rel err 1.6e-6 at h=1e-10).")
print("="*96)
print("%8s %5s %14s %14s %10s"%("N","C","h_crit","h_roundoff","decades"))
for NP,C in [(64,8),(256,8),(1024,8),(4096,8),(16384,8),(65536,8),(1024,128),(4096,128)]:
    hs=[hcrit(NP,C,6,s) for s in range(5)]; hc=float(np.median(hs)); hr=1e-9
    print("%8d %5d %14.3e %14.3e %10.2f"%(NP,C,hc,hr,np.log10(hc/hr)))
