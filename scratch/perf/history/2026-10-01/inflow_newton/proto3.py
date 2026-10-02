from proto import tail, cdf, bisect, SQPI, np, log_ndtr
from scipy.special import ndtri
eps=np.finfo(float).eps

def pre(u, s):
    shift=np.minimum(s,0.0)**2
    e0=np.exp(-s*s+shift); b0=s*SQPI*np.exp(shift+log_ndtr(np.sqrt(2.0)*s))
    t0=0.5*e0+b0
    nf=8*eps*(0.5*e0+np.abs(b0))/t0
    return shift,t0,e0,nf

GIT=2
MODEL=4
GX=1.0
def guess3(u, s):
    shift,t0,e0,nf=pre(u,s)
    g=-np.log1p(-u)
    yU = s+np.sqrt(np.maximum(g+shift-np.log(2*t0),0))
    A=e0/(2*t0); y1=np.sqrt(u/A)
    yS=y1*(1-(2*s/3)*y1+(22*s*s+9)/36*y1*y1)
    m=(s+np.sqrt(s*s+2))/2; sig=1/np.sqrt(2+1/m**2); yN=m+sig*ndtri(u)
    small = y1*np.maximum(np.abs(s),1) < 0.1
    x=np.sqrt(np.maximum(-np.log(u*t0),0))
    for _ in range(GIT):
        yy=np.maximum(s-x,1e-3)
        x=np.sqrt(np.maximum(np.log(yy/(2*x))-np.log(u*t0),0))
    yG=s-x
    cand = np.minimum(y1, np.where(2*x*yG > GX, yG, np.inf))
    cand = np.minimum(cand, np.where(yN > 0.5*m, yN, np.inf))
    if MODEL:
        target=np.log(u*t0)+np.log(2)
        t=np.log(np.minimum(y1, m))
        for _ in range(MODEL):
            yy=np.exp(t); xx=s-yy; w=np.maximum(1+xx*yy, 1e-3)
            t=t-(2*t - xx*xx - np.log(w) - target)/(2 + 2*xx*yy - yy*(s-2*yy)/w)
        cand=np.exp(t)
    low = np.where(small, yS, np.where(s <= 0, y1, cand))
    up = np.where(s <= 0, yU, np.maximum(yU, yN))
    return np.where(u < 0.5, low, up)

def solve3(u, s, n, hist=False, halley=False):
    shift,t0,e0,nf=pre(u,s)
    lo=np.zeros_like(u); hi=np.maximum(s,0.0)+12.0
    y=np.clip(guess3(u,s), 0, hi)
    logu=np.log(u); logq=np.log1p(-u); lower = u < 0.5
    A=e0/(2*t0); y1=np.sqrt(u/A)
    frozen = (u <= 64*nf) & (y1*np.maximum(np.abs(s),1) < 0.1)
    out=[]
    for _ in range(n):
        e=np.exp(-((y-s)**2)+shift)
        tl=0.5*e+s*SQPI*np.exp(shift+log_ndtr(np.sqrt(2.0)*(s-y)))
        F=1-tl/t0
        left=F<u
        lo,hi=np.where(left,y,lo),np.where(left,hi,y)
        with np.errstate(all='ignore'):
            p = y*y*e/t0/F
            hl = np.log(F)-logu
            dl = hl/p
            r = y*e/tl
            hu = np.log(tl/t0)-logq
            du = -hu/r
            if halley:
                fl = 1 - 0.5*dl*(p*(2-2*y*(y-s))-p*p)/p
                fu = 1 - 0.5*du*(-r*(1/y-2*(y-s))-r*r)/(-r)
                dl = dl/np.where(fl > 0.5, fl, 1.0)
                du = du/np.where(fu > 0.5, fu, 1.0)
            yl = np.where(F > 64*nf, y*np.exp(-dl), y + (u - F)/(y*e/t0))
            yu = y - du
        yn=np.where(lower,yl,yu)
        over = ~np.isfinite(yn) | np.where(left, yn >= hi, yn <= lo)
        y=np.where(frozen, y, np.where(over,(lo+hi)/2,np.clip(yn,lo,hi)))
        out.append(y.copy())
    return (y,out) if hist else y

def grid():
    s1=np.linspace(-10,10,401)
    u1=np.concatenate([np.logspace(np.log10(eps),-1,80),np.linspace(0.1,0.9,161),1-np.logspace(-1,np.log10(eps),80)])
    U,S=np.meshgrid(u1,s1); return U.ravel(),S.ravel()

if __name__=="__main__":
    U,S=grid()
    ref=bisect(U,S)
    y,hs=solve3(U,S,10,hist=True)
    good = U>1e-9
    for k,h in enumerate(hs):
        rel=np.abs(h/ref-1); res=np.abs(cdf(h,S)-U)
        print(k+1, f"rel max {rel.max():.1e} (u>1e-9: {rel[good].max():.1e}, mid: {rel[(U>1e-6)&(U<1-1e-6)].max():.1e}) resid max {res.max():.1e} n>1e-10 {np.sum(res>=1e-10)}")
