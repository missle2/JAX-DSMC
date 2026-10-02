import numpy as np
from proto import bisect
from truth import quantile_mp
eps=np.finfo(float).eps
s1=np.linspace(-10,10,81)
u1=np.concatenate([[eps,2*eps,1e-15],np.logspace(-14,-1,40),np.linspace(0.1,0.9,41)[1:-1],1-np.logspace(-1,-15,40),[1-2*eps,1-eps]])
U,S=np.meshgrid(u1,s1); U,S=U.ravel(),S.ravel()
ref=bisect(U,S)
T=np.array([quantile_mp(u,s,max(r,1e-12)) for u,s,r in zip(U,S,ref)])
np.savez('truth.npz',U=U,S=S,T=T,ref=ref)
