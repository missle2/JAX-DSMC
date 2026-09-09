"""Probe J: the scratch-row fix must not leak spurious gradients."""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import random
import numpy as np

cell_of=[0,0,2,2,2,2,4,5,5,5]; N,C=10,6
c=jnp.asarray(cell_of,jnp.int32); order=jnp.argsort(c); cnt=jnp.bincount(c,length=C+1)
starts=(jnp.cumsum(cnt)-cnt)[:C]; counts=cnt[:C]; n=jnp.full((C,),3,jnp.int32)

def run(theta, key, fixed):
    v = theta*jnp.arange(1., N+1.)
    if fixed: v = jnp.concatenate([v, jnp.zeros(1)])
    def body(v, a):
        active=(a<n)&(counts>=2)
        npc = jnp.maximum(counts,2) if fixed else counts
        k1,k2=random.split(random.fold_in(key,a))
        i=random.randint(k1,npc.shape,0,npc); r=random.randint(k2,npc.shape,0,npc-1)
        j=(i+1+r)%npc
        pi = jnp.where(active, order[starts+i], N) if fixed else order[starts+i]
        pj = jnp.where(active, order[starts+j], N) if fixed else order[starts+j]
        vi,vj=v[pi],v[pj]
        vcm=0.5*(vi+vj); g=0.7*(vi-vj)                # smooth, differentiable "scatter"
        vin,vjn=vcm+0.5*g, vcm-0.5*g
        v=v.at[pi].set(jnp.where(active,vin,vi)); v=v.at[pj].set(jnp.where(active,vjn,vj))
        return v, None
    v,_=jax.lax.scan(body, v, jnp.arange(3))
    return jnp.sum((v[:N])**2)

for fixed in (False, True):
    val, g = jax.value_and_grad(run)(1.3, random.key(0), fixed)
    fd = (run(1.3+1e-6, random.key(0), fixed) - run(1.3-1e-6, random.key(0), fixed))/2e-6
    print(f"  fixed={fixed}: value={float(val):.10f}  grad={float(g):.8f}  fd={float(fd):.8f}  "
          f"relerr={abs(float(g)-float(fd))/abs(float(fd)):.2e}")
