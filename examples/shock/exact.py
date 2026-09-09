"""Exact Riemann solution for the 1-D Euler equations — the analytic reference.

This is the standard iterative solver (Toro, *Riemann Solvers and Numerical Methods
for Fluid Dynamics*, ch. 4): solve for the star-region pressure by Newton iteration
on the pressure function, then sample the self-similar solution at ``xi = x/t``.

It is **inviscid**, so it is the right reference for a DSMC shock only in the
smooth regions and for the *positions* of the waves.  Three differences are
physical, expected, and must not be tuned away:

  * the shock has zero thickness here and ~a few mean free paths in DSMC;
  * the contact discontinuity is perfectly sharp here and diffuses in DSMC;
  * DSMC carries finite ``Kn``, so it approaches this solution only as ``Kn -> 0``.

The comparison is therefore made on wave positions, plateau values, and the
rarefaction profile, not on the resolution of the discontinuities themselves.
"""

from __future__ import annotations

import numpy as np


def _pressure_function(p, rho_k, p_k, a_k, gamma):
    """``f_K(p)`` and its derivative: shock branch for ``p > p_K``, else rarefaction."""
    if p > p_k:  # shock
        a = 2.0 / ((gamma + 1.0) * rho_k)
        b = (gamma - 1.0) / (gamma + 1.0) * p_k
        sq = np.sqrt(a / (p + b))
        return (p - p_k) * sq, sq * (1.0 - 0.5 * (p - p_k) / (p + b))
    # rarefaction
    pr = p / p_k
    f = 2.0 * a_k / (gamma - 1.0) * (pr ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
    df = pr ** (-(gamma + 1.0) / (2.0 * gamma)) / (rho_k * a_k)
    return f, df


def star_state(left, right, gamma=5.0 / 3.0, tol=1e-12, itmax=100):
    """Pressure and velocity in the star region.  ``left``/``right`` are ``(rho, u, p)``."""
    rl, ul, pl = left
    rr, ur, pr = right
    al, ar = np.sqrt(gamma * pl / rl), np.sqrt(gamma * pr / rr)

    # two-rarefaction approximation as the initial guess: robust for any jump
    z = (gamma - 1.0) / (2.0 * gamma)
    p = max(
        1e-12,
        ((al + ar - 0.5 * (gamma - 1.0) * (ur - ul)) / (al / pl**z + ar / pr**z)) ** (1.0 / z),
    )
    for _ in range(itmax):
        fl, dfl = _pressure_function(p, rl, pl, al, gamma)
        fr, dfr = _pressure_function(p, rr, pr, ar, gamma)
        dp = (fl + fr + (ur - ul)) / (dfl + dfr)
        p_new = max(1e-12, p - dp)
        if abs(p_new - p) / (0.5 * (p_new + p)) < tol:
            p = p_new
            break
        p = p_new
    fl, _ = _pressure_function(p, rl, pl, al, gamma)
    fr, _ = _pressure_function(p, rr, pr, ar, gamma)
    return p, 0.5 * (ul + ur) + 0.5 * (fr - fl)


def sample(xi, left, right, gamma=5.0 / 3.0):  # noqa: PLR0912, PLR0915
    """Sample the exact solution at similarity coordinate ``xi = (x - x0) / t``.

    The branch count is irreducible: the solution has a left and a right wave, each
    independently a shock or a rarefaction, and a rarefaction splits again into
    head / fan / tail.  Flattening it would only hide the case structure that makes
    it checkable against Toro's tables.

    Returns ``(rho, u, p)`` arrays for array ``xi``.
    """
    rl, ul, pl = left
    rr, ur, pr = right
    al, ar = np.sqrt(gamma * pl / rl), np.sqrt(gamma * pr / rr)
    ps, us = star_state(left, right, gamma)
    g1, g2 = (gamma - 1.0) / (2.0 * gamma), (gamma + 1.0) / (2.0 * gamma)

    xi = np.asarray(xi, dtype=float)
    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    p = np.empty_like(xi)

    for i, s in enumerate(xi.ravel()):
        if s <= us:  # left of the contact
            if ps > pl:  # left shock
                sl = ul - al * np.sqrt(g2 * ps / pl + g1)
                if s <= sl:
                    rho.ravel()[i], u.ravel()[i], p.ravel()[i] = rl, ul, pl
                else:
                    rho.ravel()[i] = (
                        rl
                        * (ps / pl + (gamma - 1) / (gamma + 1))
                        / ((gamma - 1) / (gamma + 1) * ps / pl + 1.0)
                    )
                    u.ravel()[i], p.ravel()[i] = us, ps
            else:  # left rarefaction
                shl = ul - al
                asl = al * (ps / pl) ** g1
                stl = us - asl
                if s <= shl:
                    rho.ravel()[i], u.ravel()[i], p.ravel()[i] = rl, ul, pl
                elif s >= stl:
                    rho.ravel()[i] = rl * (ps / pl) ** (1.0 / gamma)
                    u.ravel()[i], p.ravel()[i] = us, ps
                else:  # inside the fan
                    c = 2.0 / (gamma + 1.0) * (al + 0.5 * (gamma - 1.0) * (ul - s))
                    rho.ravel()[i] = rl * (c / al) ** (2.0 / (gamma - 1.0))
                    u.ravel()[i] = 2.0 / (gamma + 1.0) * (al + 0.5 * (gamma - 1) * ul + s)
                    p.ravel()[i] = pl * (c / al) ** (2.0 * gamma / (gamma - 1.0))
        elif ps > pr:  # right shock
            sr = ur + ar * np.sqrt(g2 * ps / pr + g1)
            if s >= sr:
                rho.ravel()[i], u.ravel()[i], p.ravel()[i] = rr, ur, pr
            else:
                rho.ravel()[i] = (
                    rr
                    * (ps / pr + (gamma - 1) / (gamma + 1))
                    / ((gamma - 1) / (gamma + 1) * ps / pr + 1.0)
                )
                u.ravel()[i], p.ravel()[i] = us, ps
        else:  # right rarefaction
            shr = ur + ar
            asr = ar * (ps / pr) ** g1
            str_ = us + asr
            if s >= shr:
                rho.ravel()[i], u.ravel()[i], p.ravel()[i] = rr, ur, pr
            elif s <= str_:
                rho.ravel()[i] = rr * (ps / pr) ** (1.0 / gamma)
                u.ravel()[i], p.ravel()[i] = us, ps
            else:
                c = 2.0 / (gamma + 1.0) * (ar - 0.5 * (gamma - 1.0) * (ur - s))
                rho.ravel()[i] = rr * (c / ar) ** (2.0 / (gamma - 1.0))
                u.ravel()[i] = 2.0 / (gamma + 1.0) * (-ar + 0.5 * (gamma - 1) * ur + s)
                p.ravel()[i] = pr * (c / ar) ** (2.0 * gamma / (gamma - 1.0))
    return rho, u, p


def wave_speeds(left, right, gamma=5.0 / 3.0):
    """``(head/tail of the left wave, contact, right wave)`` speeds, for reporting."""
    rl, ul, pl = left
    rr, ur, pr = right
    al, ar = np.sqrt(gamma * pl / rl), np.sqrt(gamma * pr / rr)
    ps, us = star_state(left, right, gamma)
    g1, g2 = (gamma - 1.0) / (2.0 * gamma), (gamma + 1.0) / (2.0 * gamma)
    sl = ul - al * np.sqrt(g2 * ps / pl + g1) if ps > pl else ul - al
    sr = ur + ar * np.sqrt(g2 * ps / pr + g1) if ps > pr else ur + ar
    return {"left": sl, "contact": us, "right": sr, "p_star": ps, "u_star": us}
