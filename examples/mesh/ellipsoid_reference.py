"""Free-molecular drag of a circle / sphere squashed in ``y``: the analytic reference of the
aspect-ratio optimisation (the ellipse ``(R, s R)`` per unit depth in 2-D, the ellipsoid
``(R, s R, R)`` in 3-D, stream along ``x``).

In 2-D the same Gauss-map change of variables gives ``ds = (ab)^2 / h(psi)^3 dpsi`` with
``h^2 = R^2 (cos^2 psi + s^2 sin^2 psi)`` (curvature ``h^3 / (ab)^2``; ``s = 1``: ``R dpsi``),
so ``D(s) = int_0^2pi p(cos psi) (-cos psi) R s^2 / (cos^2 psi + s^2 sin^2 psi)^(3/2) dpsi``
-- a periodic integrand, trapezoid rule.  The 2-D "volume" is the area ``pi R^2 s``.

    python3 examples/mesh/ellipsoid_reference.py            # checks + table

A specular convex body in a collisionless stream is hit at most once per molecule, so its
drag is the surface integral of the element law ``p = 2 p_i(n)`` (Bird ch. 7; no shear):

    D = -int_S p(n) n_x dA.

The integral is taken over the unit normals (the Gauss map), where the ellipsoid's area
element is ``dA = (abc)^2 / h(n)^4 dOmega`` with the support function
``h(n)^2 = a^2 n_x^2 + b^2 n_y^2 + c^2 n_z^2`` (Gaussian curvature ``K = h^4 / (abc)^2``).
With ``n = (cos t, sin t cos psi, sin t sin psi)`` and ``(a, b, c) = (R, sR, R)``,
``h^2 = R^2 (1 - k cos^2 psi)`` with ``k = (1 - s^2) sin^2 t``, and ``p`` depends on ``n_x``
only, so the azimuth integral is closed-form,

    int_0^2pi s^2 dpsi / (1 - k cos^2 psi)^2 = pi s^2 (2 - k) / (1 - k)^(3/2)  =: G(c, s),

and the drag is a 1-D integral over ``c = cos t``:

    D(s) = 2 R^2 int_{-1}^{1} p_i(c) (-c) G(c, s) dc        (s = 1: G = 2 pi, the sphere).

``p_i(c)`` is the incident normal momentum flux on an element with outward normal ``n``,
``n_x = c``.  For the stream the inlet actually produces -- the reservoir Maxwellian
restricted to ``v_x > 0`` (``examples/mesh/free_molecular.py``, FG5) -- write
``v_n = c v_x + q w`` (``q = sqrt(1 - c^2)``, ``w ~ N(0, sigma^2)`` the velocity component
along the transverse part of ``n``, independent of ``v_x``); the inner Gaussian moment is
analytic, ``E[X^2 1{X < 0}] = (mu^2 + tau^2) Phi(-mu/tau) - mu tau phi(mu/tau)`` for
``X ~ N(mu, tau^2)``, ``mu = c v_x``, ``tau = q sigma``, and the outer integral over
``v_x > 0`` is Gauss-Legendre.  ``halfrange=False`` integrates ``v_x`` over the whole line
and reproduces Bird's closed-form ``p_i`` (the check below).

Area measures of the ellipsoid (for the cost ``J = D + mu A``): volume ``4/3 pi R^3 s``,
frontal (``y-z``) projected area ``pi R^2 s``, and surface area (oblate spheroid with
symmetry axis ``y`` for ``s < 1``).  Every quantity has its ``s``-derivative.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.special import erf, ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from free_molecular import element_fluxes, reference_drag_halfrange  # noqa: E402

K_BOLTZ = 1.380649e-23
MASS_AR = 6.63e-26  # species_data.ARGON (SPARTA ar.species)
N_C = 400  # Gauss-Legendre nodes in c = cos t
N_V = 400  # Gauss-Legendre nodes in v_x (per unit of sigma, over [0, u + 12 sigma])


def _gauss(n, lo, hi):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


def _neg_second_moment(mu, tau):
    """``E[X^2 1{X < 0}]`` for ``X ~ N(mu, tau^2)`` (``tau >= 0``; ``tau = 0``: ``mu^2 1{mu<0}``)."""
    safe = np.where(tau > 0.0, tau, 1.0)
    a = mu / safe
    phi = np.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    val = (mu * mu + tau * tau) * ndtr(-a) - mu * tau * phi
    return np.where(tau > 0.0, val, mu * mu * (mu < 0.0))


def incident_pressure(c, n, mass, temp, u, halfrange=True, n_v=N_V):
    """Incident normal momentum flux ``p_i`` on elements with outward normal ``n_x = c``
    (array), stream ``u`` along ``x`` at density ``n`` and temperature ``temp``;
    ``halfrange``: the reservoir Maxwellian restricted to ``v_x > 0``."""
    sigma = math.sqrt(K_BOLTZ * temp / mass)
    lo = 0.0 if halfrange else u - 12.0 * sigma
    vx, wv = _gauss(n_v, lo, u + 12.0 * sigma)
    pdf = np.exp(-0.5 * ((vx - u) / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)
    c = np.asarray(c, dtype=float)[:, None]
    q = np.sqrt(np.maximum(1.0 - c * c, 0.0))
    inner = _neg_second_moment(c * vx[None, :], q * sigma)
    return mass * n * np.sum(inner * (pdf * wv)[None, :], axis=1)


def _g(c, s):
    """Azimuth integral ``G(c, s)`` and ``dG/ds`` (see the module docstring)."""
    k = (1.0 - s * s) * (1.0 - c * c)
    g = math.pi * s * s * (2.0 - k) / (1.0 - k) ** 1.5
    dk = -2.0 * s * (1.0 - c * c)
    # d/dk [(2 - k)(1 - k)^(-3/2)] = (1 - k)^(-5/2) (-(1 - k) + 1.5 (2 - k)) = (4 - k) / 2 / (1-k)^(5/2)
    dg = 2.0 * g / s + math.pi * s * s * 0.5 * (4.0 - k) / (1.0 - k) ** 2.5 * dk
    return g, dg


def drag(s, radius, n, mass, temp, u, halfrange=True, n_c=N_C):
    """``(D, dD/ds)`` of the specular ellipsoid with semi-axes ``(R, sR, R)``."""
    c, wc = _gauss(n_c, -1.0, 1.0)
    p = 2.0 * incident_pressure(c, n, mass, temp, u, halfrange)
    g, dg = _g(c, s)
    f = 2.0 * radius**2 * p * (-c) * wc  # the 2 R^2 prefactor and the measure dc
    return float(np.sum(f * g) / 2.0), float(np.sum(f * dg) / 2.0)


N_PSI = 8000  # trapezoid nodes over the ellipse's normal angle (2-D)


def drag_2d(s, radius, n, mass, temp, u, halfrange=True, n_psi=N_PSI):
    """``(D, dD/ds)`` per unit depth of the specular ellipse with semi-axes ``(R, sR)``."""
    psi = (np.arange(n_psi) + 0.5) * 2.0 * math.pi / n_psi
    c, sn2 = np.cos(psi), np.sin(psi) ** 2
    p = 2.0 * incident_pressure(c, n, mass, temp, u, halfrange)
    base = c * c + s * s * sn2
    jac = radius * s * s / base**1.5
    djac = radius * (2.0 * s / base**1.5 - 3.0 * s**3 * sn2 / base**2.5)
    w = 2.0 * math.pi / n_psi
    f = p * (-c) * w
    return float(np.sum(f * jac)), float(np.sum(f * djac))


def area_2d(s, radius):
    """The 2-D body "volume": ``pi R^2 s`` and its ``s``-derivative."""
    return math.pi * radius**2 * s, math.pi * radius**2


def volume(s, radius):
    return 4.0 / 3.0 * math.pi * radius**3 * s, 4.0 / 3.0 * math.pi * radius**3


def frontal_area(s, radius):
    """Projection on the ``y-z`` plane (normal to the stream)."""
    return math.pi * radius**2 * s, math.pi * radius**2


def surface_area(s, radius, n_c=N_C):
    """Surface area and its ``s``-derivative by the same Gauss-map quadrature
    (``int (abc)^2 / h^4 dOmega`` = ``R^2 int G(c, s) dc``)."""
    c, wc = _gauss(n_c, -1.0, 1.0)
    g, dg = _g(c, s)
    return float(radius**2 * np.sum(g * wc)), float(radius**2 * np.sum(dg * wc))


def spheroid_area(s, radius):
    """Closed form for the oblate (``s < 1``) / prolate (``s > 1``) spheroid, axis ``y``."""
    if abs(s - 1.0) < 1e-12:
        return 4.0 * math.pi * radius**2
    if s < 1.0:
        e = math.sqrt(1.0 - s * s)
        return 2.0 * math.pi * radius**2 * (1.0 + s * s / e * math.atanh(e))
    e = math.sqrt(1.0 - 1.0 / (s * s))
    return 2.0 * math.pi * radius**2 * (1.0 + s / e * math.asin(e))


def checks(radius=0.5, n=1.0, temp=273.15, u=100.0):
    """Self-checks: Bird's closed-form p_i, the sphere's Bird C_D, the FG5 half-range
    reference, the spheroid area formula, and dD/ds against a central difference."""
    mass = MASS_AR
    beta = 1.0 / math.sqrt(2.0 * K_BOLTZ * temp / mass)
    c = np.linspace(-1.0, 1.0, 11)
    _nd, p_bird, _t = element_fluxes(-u * beta * c, 0.0 * c, n, mass, temp)
    p_full = incident_pressure(c, n, mass, temp, u, halfrange=False)
    out = {"p_i_full_vs_bird_max_rel": float(np.max(np.abs(p_full / p_bird - 1.0)))}
    q = 0.5 * n * mass * u**2 * math.pi * radius**2
    out["cd_sphere_full"] = drag(1.0, radius, n, mass, temp, u, halfrange=False)[0] / q
    s_ratio = u * beta
    # Bird eq. 7.63-type closed form for the specular sphere (the full-Maxwellian C_D)
    sr = s_ratio
    out["cd_sphere_full_bird"] = (2 * sr * sr + 1) / (math.sqrt(math.pi) * sr**3) * math.exp(
        -sr * sr
    ) + (4 * sr**4 + 4 * sr * sr - 1) / (2 * sr**4) * erf(sr)
    out["cd_sphere_halfrange"] = drag(1.0, radius, n, mass, temp, u)[0] / q
    out["cd_sphere_halfrange_mc"] = (
        reference_drag_halfrange(3, radius, n, mass, temp, u, temp, False, samples=2_000_000) / q
    )
    out["area_quadrature_vs_closed_max_rel"] = max(
        abs(surface_area(s, radius)[0] / spheroid_area(s, radius) - 1.0)
        for s in (0.2, 0.5, 1.0, 1.5)
    )
    worst = 0.0
    for s in (0.3, 0.6, 1.0):
        h = 1e-5
        fd = (
            drag(s + h, radius, n, mass, temp, u)[0] - drag(s - h, radius, n, mass, temp, u)[0]
        ) / (2 * h)
        worst = max(worst, abs(drag(s, radius, n, mass, temp, u)[1] / fd - 1.0))
    out["dD_ds_vs_fd_max_rel"] = worst
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "ellipsoid_reference.json"))
    args = ap.parse_args(argv)
    n, temp, u, mass = 1.0, 273.15, 100.0, MASS_AR
    chk = checks(args.radius, n, temp, u)
    for k, v in chk.items():
        print(f"{k:36s} {v:.6g}")
    rows = []
    for s in np.round(np.arange(0.1, 1.51, 0.1), 2):
        d, dd = drag(s, args.radius, n, mass, temp, u)
        rows.append(
            {
                "s": float(s),
                "drag": d,
                "ddrag_ds": dd,
                "volume": volume(s, args.radius)[0],
                "frontal_area": frontal_area(s, args.radius)[0],
                "surface_area": surface_area(s, args.radius)[0],
            }
        )
        print(
            f"s={s:.2f}  D={d:.4e}  dD/ds={dd:.4e}  D/(pi R^2 s)={d / (math.pi * args.radius**2 * s):.4e}"
        )
    with open(args.out, "w") as fh:
        json.dump(
            {"radius": args.radius, "n": n, "temp": temp, "u": u, "checks": chk, "rows": rows},
            fh,
            indent=1,
        )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
