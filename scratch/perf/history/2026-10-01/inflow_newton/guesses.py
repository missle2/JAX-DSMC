from proto import *
from scipy.special import ndtri
eps=np.finfo(float).eps
def parts(u,s):
    shift=np.minimum(s,0.0)**2; t0=tail(0*s,s); g=-np.log1p(-u)
    yU = s+np.sqrt(np.maximum(g+shift-np.log(2*t0),0))
    A=np.exp(-s*s+shift)/(2*t0); yL=np.sqrt(u/A)
    m=(s+np.sqrt(s*s+2))/2; sig=1/np.sqrt(2+1/m**2); yN=m+sig*ndtri(u)
    return yU,yL,yN
for s in [-10,-3,-1,-0.5,0,0.5,1,2,3,5,10]:
  row=[]
  for u in [1e-12,1e-6,1e-3,0.1,0.3,0.5,0.7,0.9,0.99,1-1e-6,1-eps]:
    U=np.array([u]);S=np.array([float(s)])
    r=bisect(U,S)[0]; yU,yL,yN=[x[0] for x in parts(U,S)]
    row.append(f"{yU/r-1:+.2f}/{yL/r-1:+.2f}/{yN/r-1:+.2f}")
  print(s," ".join(row))
