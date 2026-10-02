import numpy as np, sys, importlib
from proto import cdf, tail
d=np.load('truth.npz'); U,S,T,ref=d['U'],d['S'],d['T'],d['ref']
shift=np.minimum(S,0)**2; t0=tail(0*S,S)
pdf=T*np.exp(-(T-S)**2+shift)/t0
# empirical F-noise around the truth: deviations of computed F from its linearisation
ks=np.arange(-64,65)
from scipy.special import log_ndtr
e0=np.exp(-S*S+shift); b0=S*np.sqrt(np.pi)*np.exp(shift+log_ndtr(np.sqrt(2)*S)); kap=(0.5*e0+np.abs(b0))/t0
dy=np.minimum(4*2.2e-16*np.maximum(kap,1)/np.maximum(pdf,1e-300)/16, 0.01*T)
dev=np.zeros_like(T)
for k in ks:
    yk=T+k*dy
    dev=np.maximum(dev,np.abs(cdf(yk,S)-U-pdf*(yk-T)))
noise_y=np.maximum(dev/np.maximum(pdf,1e-300), 4e-16*T)
np.save('noise_y.npy',noise_y)
def score(y):
    z=np.abs(y-T)/noise_y
    return z
if __name__=="__main__":
    mod=importlib.import_module(sys.argv[1]); solve=getattr(mod,sys.argv[2])
    zb=score(ref); print('bisect  z max %.2f p99.9 %.2f' % (zb.max(), np.quantile(zb,0.999)))
    for n in (3,4,5,6,8):
        z=score(solve(U,S,n)); i=np.argmax(z)
        print(n,'newton z max %.2f p99.9 %.2f  worst u=%.3g s=%.3g relerr %.1e noise %.1e' % (z.max(), np.quantile(z,0.999),U[i],S[i],abs(solve(U,S,n)[i]/T[i]-1),noise_y[i]/T[i]))
