"""numpy prototype: iteration counts of safeguarded Newton/Halley for the flux quantile."""

import numpy as np
from scipy.special import log_ndtr

SQPI = np.sqrt(np.pi)


def tail(z, s):
    shift = np.minimum(s, 0.0) ** 2
    return 0.5 * np.exp(-((z - s) ** 2) + shift) + s * SQPI * np.exp(
        shift + log_ndtr(np.sqrt(2.0) * (s - z))
    )


def cdf(y, s):
    return 1.0 - tail(y, s) / tail(np.zeros_like(y), s)


def bisect(u, s, n=64):
    lo = np.zeros_like(u)
    hi = np.maximum(s, 0.0) + 12.0
    for _ in range(n):
        mid = (lo + hi) / 2
        left = cdf(mid, s) < u
        lo, hi = np.where(left, mid, lo), np.where(left, hi, mid)
    return (lo + hi) / 2


def guess(u, s):
    shift = np.minimum(s, 0.0) ** 2
    t0 = tail(0.0 * s, s)
    g = -np.log1p(-u)
    a = np.abs(s)
    root = np.sqrt(s * s + g)
    yq = np.where(s > 0, s + root, g / (root + a))
    A = np.exp(-s * s + shift) / (2 * t0)
    yl = np.sqrt(u / A)
    return np.where(s > 0, np.minimum(yq, yl), np.maximum(yq, yl))


def solve(u, s, n, method="halley", form="log", y0=None, hist=False):
    shift = np.minimum(s, 0.0) ** 2
    t0 = tail(0.0 * s, s)
    lo = np.zeros_like(u)
    hi = np.maximum(s, 0.0) + 12.0
    y = guess(u, s) if y0 is None else y0
    y = np.clip(y, lo, hi)
    logq = np.log1p(-u)
    out = []
    for _ in range(n):
        e = np.exp(-((y - s) ** 2) + shift)
        tl = 0.5 * e + s * SQPI * np.exp(shift + log_ndtr(np.sqrt(2.0) * (s - y)))
        F = 1 - tl / t0
        left = F < u
        lo, hi = np.where(left, y, lo), np.where(left, hi, y)
        if form == "log":
            # r(y) = log tl - log t0 - logq ; decreasing. r' = -y e / tl ; r'' = ...
            r = np.log(tl / t0) - logq
            d1 = -y * e / tl
            # d/dy (-y e / tl) = -(e + y e'(y))/tl + y e * tl'/tl^2, e' = -2(y-s)e, tl' = -y e
            d2 = -(e - 2 * y * (y - s) * e) / tl - (y * e) ** 2 / tl**2
        else:
            r = F - u
            d1 = y * e / t0
            d2 = (e - 2 * y * (y - s) * e) / t0
        step = r / d1
        if method == "halley":
            step = step / (1 - 0.5 * step * d2 / d1)
        yn = y - step
        ok = np.isfinite(yn) & (yn > lo) & (yn < hi)
        y = np.where(ok, yn, (lo + hi) / 2)
        if hist:
            out.append(y.copy())
    return (y, out) if hist else y


if __name__ == "__main__":
    eps = np.finfo(float).eps
    s1 = np.linspace(-10, 10, 201)
    u1 = np.concatenate(
        [np.logspace(np.log10(eps), -1, 60), np.linspace(0.1, 0.9, 81), 1 - np.logspace(-1, np.log10(eps), 60)]
    )
    U, S = np.meshgrid(u1, s1)
    U, S = U.ravel(), S.ravel()
    ref = bisect(U, S)
    y0 = guess(U, S)
    print("guess rel err: max", np.max(np.abs(y0 / ref - 1)), "p99", np.quantile(np.abs(y0 / ref - 1), 0.99))
    for method in ("newton", "halley"):
        for form in ("log", "cdf"):
            y, hs = solve(U, S, 10, method, form, hist=True)
            errs = [np.max(np.abs(h - ref) / np.maximum(ref, 1e-300)) for h in hs]
            res = np.max(np.abs(cdf(y, S) - U))
            print(method, form, " ".join(f"{e:.1e}" for e in errs), "resid", f"{res:.1e}")
