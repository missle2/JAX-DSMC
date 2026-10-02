import numpy as np, sys, importlib
d=np.load('truth.npz'); U,S,T=d['U'],d['S'],d['T']
noise=np.load('noise_y.npy')
mod=importlib.import_module(sys.argv[1]); solve=getattr(mod,sys.argv[2])
N=12
_,hs=solve(U,S,N,hist=True)
ok=np.array([np.abs(h-T)/noise<3 for h in hs])  # N x P
# first k after which all later ok
need=np.full(len(U),N+1)
for k in range(N-1,-1,-1):
    good=ok[k:].all(0)
    need=np.where(good,k+1,need)
print('max iterations needed', need.max(), 'hist', np.bincount(need))
sreg=np.digitize(S,[-5,-1,0,1,3,6]); ureg=np.digitize(U,[1e-9,1e-3,0.5,1-1e-3,1-1e-9])
print('rows: s bins (-inf,-5,-1,0,1,3,6,inf); cols: u bins (<1e-9,<1e-3,<.5,<1-1e-3,<1-1e-9,rest)')
for a in range(7):
    print(a, [int(need[(sreg==a)&(ureg==b)].max()) if np.any((sreg==a)&(ureg==b)) else -1 for b in range(6)])
