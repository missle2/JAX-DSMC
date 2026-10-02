import sys, importlib, numpy as np
from proto import cdf
d=np.load('truth.npz'); U,S,T,ref=d['U'],d['S'],d['T'],d['ref']
mod=importlib.import_module(sys.argv[1]); solve=getattr(mod,sys.argv[2])
er=np.abs(ref/T-1)
def report(y, tag):
    en=np.abs(y/T-1); res=np.abs(cdf(y,S)-U)
    worse = en > 2*er + 4e-16
    reg={'u<1e-9':U<1e-9,'tail':(U>=1e-9)&(U<1e-3),'mid':(U>=1e-3)&(U<=1-1e-3),'upper':U>1-1e-3}
    s=" ".join(f"{k}: new {en[m].max():.1e}/bis {er[m].max():.1e}" for k,m in reg.items())
    i=np.argmax(np.where(worse,en-er,0))
    print(tag, s, f"| worse {worse.sum()} worst(u={U[i]:.3g},s={S[i]:.3g},new {en[i]:.1e},bis {er[i]:.1e}) resid {res.max():.1e} nfail {np.sum(~(res<1e-10))}")
for n in range(3,9):
    report(solve(U,S,n), n)
report(ref,'bisect')
