import mpmath as mp
mp.mp.dps = 50
def tail_mp(z, s):
    z=mp.mpf(z); s=mp.mpf(s)
    return mp.mpf(0.5)*mp.exp(-(z-s)**2) + s*mp.sqrt(mp.pi)/2*mp.erfc(z-s)
def quantile_mp(u, s, y0=None):
    """bracketed Newton in mp: log-form residuals, bisection fallback."""
    u=mp.mpf(u); s=mp.mpf(s)
    t0=tail_mp(0,s); q=1-u
    lo=mp.mpf(0); hi=max(s,0)+14
    y=mp.mpf(y0) if y0 else (lo+hi)/2
    if not (lo < y < hi): y=(lo+hi)/2
    for _ in range(400):
        tl=tail_mp(y,s); F=(t0-tl)/t0
        if F<u: lo=y
        else: hi=y
        pdf=y*mp.exp(-(y-s)**2)/t0
        if u>0.5: step=(mp.log(tl/t0)-mp.log(q))*(tl/t0)/pdf
        else: step=-(mp.log(F)-mp.log(u))*F/pdf if F>0 else (u-F)/pdf
        yn=y+step
        if not (lo < yn < hi): yn=(lo+hi)/2
        if abs(yn-y) < mp.mpf(10)**-40*abs(y): return float(yn)
        y=yn
    raise RuntimeError((u,s))
