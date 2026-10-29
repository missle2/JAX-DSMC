"""Exact free-molecular reference for the paper's Newton-problem cases (2026-10-28).

A convex body of revolution about the stream axis ``x`` inside the design cylinder of radius
``R`` and height ``H``: a flat front disk of radius ``r_0`` at ``x = 0``, a lateral surface
``r(x)`` (piecewise linear between ``N + 1`` stations, concave and non-decreasing: the body
is convex, so no molecule hits it twice and nothing is shadowed -- the panel integrals below
are then exact in free-molecular flow) and the base disk of radius ``R`` at ``x = H``.

Per unit area each panel sees the drifting Maxwellian of number density ``n``, speed ``U``
and temperature ``T`` (most probable speed ``sqrt(2 k T / m)``, speed ratio ``s``); with
``c_n`` the velocity component into the surface (mean ``U_n``, deviation ``sigma = sqrt(k T
/ m)``) the truncated-normal moments ``E[c_n^k; c_n > 0]`` give the incident number flux,
normal momentum flux, shear and energy flux.  Specular walls double the normal momentum and
exchange no energy; diffuse walls at ``T_w`` re-emit the incident number flux as a
half-range Maxwellian (normal momentum ``ndot m sigma_w sqrt(pi / 2)``, energy ``2 k T_w``
per molecule).  Newton's resistance law is the specular ``s -> inf`` limit.

The profile minimising the drag is found with SLSQP (``scipy.optimize``) over the station
radii and ``r_0``, optionally with every panel's heat flux at most ``beta`` times the
stagnation value of a flat face (a peak-heating cap: diffuse walls only).

    python3 examples/paper/reference/newton_fm.py [--height 1] [--speed-ratio 9.13] \
        [--wall specular|diffuse] [--beta 0.5] [--stations 60]
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf

K_BOLTZ = 1.380649e-23
MASS = 6.63e-26  # argon, kg
TEMP = 273.15  # free-stream temperature, K (examples/mesh/forward.TEMP)
T_WALL = 300.0  # diffuse wall temperature, K (examples/mesh/forward.T_SURF)


def _positive_moments(mean, sigma):
    """``E[c^k 1(c > 0)]`` for ``k = 0..3`` of ``c ~ N(mean, sigma^2)`` (arrays)."""
    a = mean / sigma
    big_phi = 0.5 * (1.0 + erf(a / math.sqrt(2.0)))
    small_phi = np.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    m0 = big_phi
    m1 = mean * big_phi + sigma * small_phi
    m2 = (mean**2 + sigma**2) * big_phi + mean * sigma * small_phi
    m3 = (mean**3 + 3.0 * mean * sigma**2) * big_phi + (mean**2 + 2.0 * sigma**2) * sigma * small_phi
    return m0, m1, m2, m3


def panel_fluxes(cos_in, sin_t, s, wall, t_wall=T_WALL, temp=TEMP, mass=MASS, n=1.0):
    """Per unit area on a panel whose inward normal makes ``cos_in`` with the stream (``> 0``:
    facing it) and whose tangent carries ``sin_t`` of it: ``(p, tau, q)`` -- the pressure on
    the wall (along the inward normal), the shear along the stream's tangential projection
    and the heat flux into the wall (0 for specular walls)."""
    sigma = math.sqrt(K_BOLTZ * temp / mass)
    u = s * math.sqrt(2.0) * sigma
    _m0, m1, m2, m3 = _positive_moments(u * cos_in, sigma)
    ndot = n * m1
    p_in = n * mass * m2
    ut = u * sin_t
    tau = n * mass * ut * m1
    energy_in = 0.5 * n * mass * (m3 + m1 * (ut**2 + 2.0 * sigma**2))
    if wall == "specular":
        return 2.0 * p_in, np.zeros_like(p_in), np.zeros_like(p_in)
    sigma_w = math.sqrt(K_BOLTZ * t_wall / mass)
    p = p_in + ndot * mass * sigma_w * math.sqrt(math.pi / 2.0)
    q = energy_in - ndot * 2.0 * K_BOLTZ * t_wall
    return p, tau, q


def profile(params, height, radius, stations):
    """``(x, r)`` of the station radii ``params = [r_0, r_1, ..., r_{N-1}]`` (``r_N = R``)."""
    x = np.linspace(0.0, height, stations + 1)
    r = np.concatenate((params, [radius]))
    return x, r


def body_terms(params, height, radius, stations, s, wall):
    """``(drag, panel heat fluxes)`` of the body ``params`` (force in units of the flat
    disk's specular drag at ``s``; heat flux in units of the flat face's diffuse stagnation
    value)."""
    x, r = profile(params, height, radius, stations)
    dx, dr = np.diff(x), np.diff(r)
    length = np.hypot(dx, dr)
    cos_in = dr / length  # inward normal . stream (the panel faces upstream when dr > 0)
    sin_t = dx / length
    area = math.pi * (r[:-1] + r[1:]) * length
    p, tau, q = panel_fluxes(cos_in, sin_t, s, wall)
    lateral = np.sum(area * (p * cos_in + tau * sin_t))
    one = np.ones(1)
    p_front, _t, q_front = panel_fluxes(one, 0.0 * one, s, wall)
    p_base, _t, _q = panel_fluxes(-one, 0.0 * one, s, wall)
    drag = lateral + math.pi * r[0] ** 2 * p_front[0] - math.pi * radius**2 * p_base[0]
    disk, _t, _q = panel_fluxes(one, 0.0 * one, s, "specular")
    q_stag = panel_fluxes(one, 0.0 * one, s, "diffuse")[2][0]
    heat = np.concatenate(([q_front[0]] if r[0] > 1e-9 else [], q)) / q_stag
    return drag / (math.pi * radius**2 * disk[0]), heat


def drag_of(params, height, radius, stations, s, wall):
    return body_terms(params, height, radius, stations, s, wall)[0]


def _slopes(params, height, radius, stations):
    x, r = profile(params, height, radius, stations)
    return np.diff(r) / np.diff(x)


def _monotone(params, height, radius, stations):
    """``dr >= 0`` per panel."""
    return _slopes(params, height, radius, stations)


def _concave(params, height, radius, stations):
    """Slopes non-increasing (a convex body)."""
    return -np.diff(_slopes(params, height, radius, stations))


def _heat_cap(params, height, radius, stations, s, wall, beta):
    """``beta - q_panel / q_stag >= 0`` per lateral panel (a flat front disk sees the
    stagnation value itself, so a cap ``beta < 1`` fixes ``r_0 = 0``: :func:`optimise`)."""
    x, r = profile(params, height, radius, stations)
    dx, dr = np.diff(x), np.diff(r)
    length = np.hypot(dx, dr)
    q = panel_fluxes(dr / length, dx / length, s, wall)[2]
    one = np.ones(1)
    q_stag = panel_fluxes(one, 0.0 * one, s, "diffuse")[2][0]
    return beta - q / q_stag


def optimise(height, radius, stations, s, wall, beta=None, start=None):
    """The minimum-drag convex profile (SLSQP); ``beta`` caps the panel heat flux."""
    if start is None:
        x = np.linspace(0.0, height, stations + 1)[:-1]
        start = radius * np.sqrt(0.2 + 0.8 * x / height)  # a blunted paraboloid
    args = (height, radius, stations)
    cons = [
        {"type": "ineq", "fun": functools.partial(_monotone, height=height, radius=radius,
                                                  stations=stations)},
        {"type": "ineq", "fun": functools.partial(_concave, height=height, radius=radius,
                                                  stations=stations)},
    ]  # fmt: skip
    bounds = [(0.0, radius)] * stations
    if beta is not None:
        bounds[0] = (0.0, 0.0)  # pointed: a flat face would see q_stag
        start = np.concatenate(([0.0], start[1:]))
        cons.append({"type": "ineq", "fun": functools.partial(
            _heat_cap, height=height, radius=radius, stations=stations, s=s, wall=wall,
            beta=beta)})  # fmt: skip
    res = minimize(
        drag_of, start, args=(*args, s, wall), method="SLSQP", constraints=cons,
        bounds=bounds, options={"maxiter": 2000, "ftol": 1e-12},
    )  # fmt: skip
    drag, heat = body_terms(res.x, *args, s, wall)
    x, r = profile(res.x, *args)
    return {
        "height": height, "radius": radius, "speed_ratio": s, "wall": wall, "beta": beta,
        "success": bool(res.success), "message": str(res.message), "drag": float(drag),
        "front_radius": float(r[0]), "q_peak": float(heat.max()) if wall == "diffuse" else 0.0,
        "x": x.tolist(), "r": r.tolist(),
    }  # fmt: skip


def edge_slope(result):
    """``dr/dx`` of the first lateral panel (Newton's solution meets its flat disk at 45
    degrees: ``1``)."""
    x, r = np.asarray(result["x"]), np.asarray(result["r"])
    return float((r[1] - r[0]) / (x[1] - x[0]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--speed-ratio", type=float, default=10.0 * math.sqrt(5.0 / 6.0))
    ap.add_argument("--wall", default="specular", choices=("specular", "diffuse"))
    ap.add_argument("--beta", type=float, nargs="*", default=[])
    ap.add_argument("--stations", type=int, default=60)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    results = [optimise(args.height, args.radius, args.stations, args.speed_ratio, args.wall)]
    for beta in args.beta:
        results.append(optimise(args.height, args.radius, args.stations, args.speed_ratio,
                                args.wall, beta))  # fmt: skip
    for res in results:
        print(f"wall {res['wall']} s {res['speed_ratio']:.3g} H/R {res['height'] / res['radius']:g}"
              f" beta {res['beta']}: drag/disk {res['drag']:.4f}  r0 {res['front_radius']:.3f}"
              f"  edge slope {edge_slope(res):.3f}  q_peak/q_stag {res['q_peak']:.3f}  "
              f"{res['message']}")  # fmt: skip
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
