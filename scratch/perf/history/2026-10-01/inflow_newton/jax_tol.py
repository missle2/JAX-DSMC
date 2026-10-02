import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax_dsmc import inflow
exec(open("scratch/inflow_newton/jax_accuracy.py").read().split("newton = jax.jit")[0].split("from jax_dsmc import inflow")[1])
eps=np.finfo(float).eps
s1=np.linspace(-10,10,81)
u1=np.concatenate([[eps,1e-15],np.logspace(-14,-1,27),np.linspace(0.05,0.95,19),1-np.logspace(-2,-15,27),[1-eps]])
U,S=np.meshgrid(u1,s1); U,S=U.ravel(),S.ravel()
yb=np.asarray(bisect(U,S)); yn=np.asarray(jax.jit(inflow.flux_quantile)(U,S)); rel=np.abs(yn/yb-1)
for smax in (3,5,10):
  for ulo in (1e-9,1e-6,1e-3):
    m=(np.abs(S)<=smax)&(U>=ulo)&(U<=1-ulo)
    print(smax,ulo,'%.1e'%rel[m].max())
# JVP vs FD
q=jax.jit(inflow.flux_quantile)
u=jnp.asarray(np.linspace(0.02,0.98,25)); 
for s in [-10.,-3.,-0.5,0.,0.5,3.,10.]:
  _,ts=jax.jvp(lambda d: inflow.flux_quantile(u,d),(jnp.full_like(u,s),),(jnp.ones_like(u),))
  _,tu=jax.jvp(lambda v: inflow.flux_quantile(v,jnp.full_like(u,s)),(u,),(jnp.ones_like(u),))
  h=1e-5
  fs=(q(u,jnp.full_like(u,s+h))-q(u,jnp.full_like(u,s-h)))/(2*h)
  hu=1e-6
  fu=(q(u+hu,jnp.full_like(u,s))-q(u-hu,jnp.full_like(u,s)))/(2*hu)
  print(s,'ds rel %.1e'%np.max(np.abs(ts/fs-1)),'du rel %.1e'%np.max(np.abs(tu/fu-1)))
