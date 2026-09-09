"""CLAIM 3(b): amplification -- one particle changing cell rebinds how much of the pairing?"""
import jax, jax.numpy as jnp, numpy as np, functools
jax.config.update("jax_enable_x64", True)
LO,HI=0.0,1.0; SIG=300.0; DT=2.0e-4; NP,C,M,AMAX=256,8,6,16

@jax.jit
def dec(temp):
    B=jax.random.PRNGKey(2024)
    Z=jax.random.normal(jax.random.fold_in(B,1),(NP,3)); X0=jax.random.uniform(jax.random.fold_in(B,2),(NP,))
    v=Z*jnp.sqrt(temp)*SIG/np.sqrt(300.0); x=LO+(HI-LO)*X0; dx=(HI-LO)/C
    cells=[];pis=[];pjs=[]
    for s in range(M):
        x=x+v[:,0]*DT; x=LO+jnp.mod(x-LO,HI-LO)
        c=jnp.clip(jnp.floor((x-LO)/dx),0,C-1).astype(jnp.int32)
        order=jnp.argsort(c,stable=True)
        counts=jnp.sum(jax.nn.one_hot(c,C,dtype=jnp.float64),axis=0)
        starts=jnp.concatenate([jnp.zeros(1),jnp.cumsum(counts)[:-1]]).astype(jnp.int32)
        ci=counts.astype(jnp.int32)
        P=[];Q=[]
        for a in range(AMAX):
            ka=jax.random.fold_in(jax.random.fold_in(B,100+s),a); k1,k2=jax.random.split(ka)
            npc=jnp.maximum(ci,1)
            i=jax.random.randint(k1,(C,),0,npc); r=jax.random.randint(k2,(C,),0,jnp.maximum(npc-1,1))
            j=(i+1+r)%npc
            P.append(order[jnp.clip(starts+i,0,NP-1)]); Q.append(order[jnp.clip(starts+j,0,NP-1)])
        cells.append(c); pis.append(jnp.stack(P)); pjs.append(jnp.stack(Q))
    return jnp.stack(cells), jnp.stack(pis), jnp.stack(pjs)

T0=300.0
print("Amplification of a single cell-membership flip into the pairing.")
print("N=%d particles, C=%d cells, M=%d steps, A_max=%d  ->  %d cell decisions, %d pair-index decisions"
      %(NP,C,M,AMAX,NP*M,2*M*AMAX*C))
print()
print("%10s %12s %14s %16s %14s"%("h(rel)","cell flips","particles moved","pair-idx flips","% of pairings"))
prev=None
for e in np.arange(3.0,6.01,0.125):
    h=10.0**(-e)
    cp,pip,pjp=dec(T0*(1+h)); cn,pin,pjn=dec(T0*(1-h))
    cf=int(jnp.sum(cp!=cn)); pf=int(jnp.sum(pip!=pin))+int(jnp.sum(pjp!=pjn))
    tot=2*M*AMAX*C
    if cf!=prev or cf<=6:
        print("%10.2e %12d %14d %16d %14.1f"%(h,cf,cf,pf,100.0*pf/tot))
    prev=cf
