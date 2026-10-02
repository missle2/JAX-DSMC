from proto import tail, cdf, bisect, SQPI, np, log_ndtr
from scipy.special import ndtri
eps=np.finfo(float).eps

def guess2(u, s):
    shift=np.minimum(s,0.0)**2; t0=tail(0*s,s); g=-np.log1p(-u)
    yU = s+np.sqrt(np.maximum(g+shift-np.log(2*t0),0))
    A=np.exp(-s*s+shift)/(2*t0); yL=np.sqrt(u/A)
    m=(s+np.sqrt(s*s+2))/2; sig=1/np.sqrt(2+1/m**2); yN=m+sig*ndtri(u)
    low = np.where(s <= 0, yL, np.where(yN > 0, np.minimum(yL, yN), yL))
    up = np.where(s <= 0, yU, np.maximum(yU, yN))
    return np.where(u < 0.5, low, up)

def solve2(u, s, n, hist=False, safeguard=True):
    shift=np.minimum(s,0.0)**2; t0=tail(0*s,s)
    lo=np.zeros_like(u); hi=np.maximum(s,0.0)+12.0
    y=np.clip(guess2(u,s), 0, hi)
    logu=np.log(u); logq=np.log1p(-u); lower = u < 0.5
    out=[]
    for _ in range(n):
        e=np.exp(-((y-s)**2)+shift)
        tl=0.5*e+s*SQPI*np.exp(shift+log_ndtr(np.sqrt(2.0)*(s-y)))
        F=1-tl/t0
        left=F<u
        lo,hi=np.where(left,y,lo),np.where(left,hi,y)
        with np.errstate(all='ignore'):
            # lower: newton in log y on log F
            yl = np.where(F > 0, y*np.exp(-(np.log(F)-logu)*F/(y*y*e/t0)), y + (u - F)/(y*e/t0))
            # upper: newton in y on log q
            yu = y + (np.log(tl/t0)-logq)*tl/(y*e)
        yn=np.where(lower,yl,yu)
        if safeguard:
            over = ~np.isfinite(yn) | np.where(left, yn >= hi, yn <= lo)
            y=np.where(over,(lo+hi)/2,np.clip(yn,lo,hi))
        else:
            y=yn
        out.append(y.copy())
    return (y,out) if hist else y

if __name__=="__main__":
    s1=np.linspace(-10,10,401)
    u1=np.concatenate([np.logspace(np.log10(eps),-1,80),np.linspace(0.1,0.9,161),1-np.logspace(-1,np.log10(eps),80)])
    U,S=np.meshgrid(u1,s1); U,S=U.ravel(),S.ravel()
    ref=bisect(U,S)
    y,hs=solve2(U,S,10,hist=True)
    for k,h in enumerate(hs):
        rel=np.abs(h/ref-1); res=np.abs(cdf(h,S)-U)
        good = U>1e-6  # outside the lower noise band
        print(k+1, f"rel max {rel.max():.1e} (u>1e-6: {rel[good].max():.1e}) resid max {res.max():.1e} n>1e-10 {np.sum(res>=1e-10)}")
    i=np.argmax(np.abs(hs[5]/ref-1)[U>1e-6]); print(U[U>1e-6][i],S[U>1e-6][i])
