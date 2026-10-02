import numpy as np, sys, importlib
from proto import cdf, tail, bisect
from scipy.special import log_ndtr
mod=importlib.import_module(sys.argv[1]); solve=getattr(mod,sys.argv[2]); n=int(sys.argv[3])
rng=np.random.default_rng(1)
P=400000
S=rng.uniform(-10,10,P)
eps=np.finfo(float).eps
kind=rng.integers(0,3,P)
U=np.where(kind==0, rng.uniform(0,1,P), np.where(kind==1, 10**rng.uniform(np.log10(eps),0,P), 1-10**rng.uniform(np.log10(eps),0,P)))
U=np.clip(U,eps,1-eps)
T=bisect(U,S)
y=solve(U,S,n)
shift=np.minimum(S,0)**2; t0=tail(0*S,S)
pdf=T*np.exp(-(T-S)**2+shift)/t0
e0=np.exp(-S*S+shift); b0=S*np.sqrt(np.pi)*np.exp(shift+log_ndtr(np.sqrt(2)*S)); kap=(0.5*e0+np.abs(b0))/t0
dy=np.minimum(4*eps*np.maximum(kap,1)/np.maximum(pdf,1e-300)/16, 0.01*T)
dev=np.zeros_like(T)
for k in range(-64,65,4):
    yk=T+k*dy; dev=np.maximum(dev,np.abs(cdf(yk,S)-U-pdf*(yk-T)))
noise=np.maximum(dev/np.maximum(pdf,1e-300),4e-16*T)
z=np.abs(y-T)/noise; res=np.abs(cdf(y,S)-U)
print(n, 'z vs bisect max %.2f p99.99 %.2f; resid max %.1e; fails %d; nonfinite %d' % (z.max(), np.quantile(z,0.9999), res.max(), np.sum(~(res<1e-10)), np.sum(~np.isfinite(y))))
i=np.argsort(-z)[:3]; print(U[i],S[i],y[i],T[i])
from truth import quantile_mp
bad=np.where(z>10)[0]
print('flagged', len(bad))
rng2=np.random.default_rng(2); sel=bad if len(bad)<300 else rng2.choice(bad,300,replace=False)
worse=0; wmax=0
for j in sel:
    t=quantile_mp(U[j],S[j]); en=abs(y[j]/t-1); eb=abs(T[j]/t-1)
    if en>2*eb+4e-16:
        worse+=1; wmax=max(wmax,en)
        if worse<5: print('  worse', U[j],S[j],en,eb)
print('checked',len(sel),'new worse than bisect (vs mp truth):',worse,'max new relerr there',wmax)
