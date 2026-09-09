"""Decaying-turbulence cases, D in {1, 2, 3} (spec section 10.2).

Same solver as the shock cases, same periodic boundary; only the initial
condition and the post-processing differ.  Each dimension is anchored to a
reference that is a *number*, not a slope to be eyeballed, because a log-log plot
of a noisy spectrum can be made to look like almost any power law:

======  ============================  ==========================================
D       case                          reference
======  ============================  ==========================================
1       compressive, band-limited     steepens into shocks; a field with jump
                                      discontinuities has Fourier coefficients
                                      ~1/k, hence ``E(k) ~ k^-2`` (Burgers).
                                      Use ``--ic broadband``: a single sinusoid
                                      combs ``E(k)`` with its own symmetry
2       Taylor-Green vortex           an **exact** solution of incompressible
                                      Navier-Stokes: every mode decays as
                                      ``exp(-2 nu k^2 t)``, so kinetic energy
                                      decays as ``exp(-4 nu k^2 t)`` and the
                                      fitted rate measures ``nu`` directly
                                      against Chapman-Enskog
3       Taylor-Green vortex           the standard DNS benchmark; dissipation
                                      rate ``-dE/dt`` peaks at ``t* ~ 9`` at
                                      Re = 1600.  Compared against NS-DNS at
                                      *matched* Reynolds number
======  ============================  ==========================================

**The 2-D case is the strongest gate in the project so far.**  It has a closed-form
answer, and the quantity it measures -- the viscosity -- is the transport
coefficient that the whole VSS collision model exists to reproduce.  It fails if
the collision rate, the scattering law or the timestep is wrong, and unlike the
shock thickness it needs no smoothing, no shock-frame alignment and no fitted
maximum slope.

WHAT IS AND IS NOT REACHABLE.  DSMC turbulence is expensive for a reason that is
worth stating before any run is launched.  Cells must resolve the mean free path,
so the cell count scales as ``(L/lambda)^D``, while

    Re ~ 2 Ma sqrt(gamma) (L / lambda)

(using ``nu ~ (1/2) lambda v_th``).  Reynolds number therefore costs
``Re^D`` in cells.  :func:`sizing` prints what a given memory budget buys, and
:func:`report_reach` turns that into a Reynolds number.  On four H200s
(564 GB) that is roughly ``Re ~ 10^3`` in 2-D but only ``Re ~ 10^2`` in 3-D --
the Re = 1600 3-D benchmark is a supercomputer-scale problem (Gallis et al. ran
it on Trinity), and claiming it here would be dishonest.  The 3-D case is
therefore validated against DNS **at the Reynolds number actually run**.
"""

from __future__ import annotations

import math

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from examples.shock.normal import (  # noqa: E402
    AR_ALPHA,
    AR_DIAM,
    AR_MASS,
    AR_OMEGA,
    AR_TREF,
    mfp_vss,
)
from jax_dsmc.constants import K_BOLTZ  # noqa: E402

GAMMA = 5.0 / 3.0


def viscosity_vss(temp, omega=AR_OMEGA, alpha=AR_ALPHA, diam=AR_DIAM, tref=AR_TREF, mass=AR_MASS):
    """Chapman-Enskog viscosity of a VSS gas, Bird eqn (4.64) [Pa s].

        mu = 5 (a+1)(a+2) sqrt(pi m k T_ref)
             / [4 a (5-2w)(7-2w) pi d_ref^2]  * (T/T_ref)^w

    At ``alpha = 1`` this reduces to the VHS form ``15 sqrt(pi m k T)/(2 pi d^2
    (5-2w)(7-2w))``, which is the check that the ``alpha`` factors are the right
    way up.  For argon at 273.15 K it gives 2.116e-5 Pa s against a measured
    2.117e-5 -- as it must, since ``ar.vss`` was fitted to exactly that.

    This is the number the 2-D Taylor-Green case measures, so it is the *target*,
    not an input: nothing in the solver is told about it.
    """
    pref = (
        5.0
        * (alpha + 1.0)
        * (alpha + 2.0)
        * math.sqrt(math.pi * mass * K_BOLTZ * tref)
        / (4.0 * alpha * (5.0 - 2.0 * omega) * (7.0 - 2.0 * omega) * math.pi * diam * diam)
    )
    return pref * (temp / tref) ** omega


def kinematic_viscosity(temp, nrho, **kw):
    """``nu = mu / rho`` [m^2/s], the quantity the decay rate actually contains."""
    return viscosity_vss(temp, **kw) / (AR_MASS * nrho)


def sound_speed(temp, gamma=GAMMA):
    return math.sqrt(gamma * K_BOLTZ * temp / AR_MASS)


def taylor_green(x, u0, kbox, dim):
    """Taylor-Green velocity field at positions ``x``; returns ``f64[N, 3]``.

    2-D (an exact Navier-Stokes solution)::

        u = ( U0 sin(k x) cos(k y),  -U0 cos(k x) sin(k y),  0 )

    3-D (the DNS benchmark)::

        u = ( U0 sin(k x) cos(k y) cos(k z),
             -U0 cos(k x) sin(k y) cos(k z),
              0 )

    Both are divergence-free at ``t = 0``, which matters: a compressive component
    would radiate sound waves that swamp the vortical decay this case measures.
    The 3-D field is *not* an exact solution -- it is the standard initial
    condition whose decay is the benchmark.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    n = x.shape[0]
    if dim == 1:
        # No solenoidal field exists in 1-D, and none is wanted: the 1-D case is
        # deliberately COMPRESSIVE so it steepens into the shock that produces the
        # k^-2 spectrum.  A transverse-only field would simply decay without
        # cascading.
        u = jnp.zeros((n, 3)).at[:, 0].set(u0 * jnp.sin(kbox * x[:, 0]))
        return u
    cx, sx = jnp.cos(kbox * x[:, 0]), jnp.sin(kbox * x[:, 0])
    cy, sy = jnp.cos(kbox * x[:, 1]), jnp.sin(kbox * x[:, 1])
    cz = jnp.cos(kbox * x[:, 2]) if dim == 3 else 1.0
    u = jnp.zeros((n, 3))
    u = u.at[:, 0].set(u0 * sx * cy * cz)
    u = u.at[:, 1].set(-u0 * cx * sy * cz)
    return u


def band_wavevectors(k_max, dim):
    """Integer wavevectors with ``1 <= |n| <= k_max``, one per conjugate pair.

    Half-space only (first non-zero component positive), so that summing
    ``A cos(k.x + phi)`` over the returned list gives a real field with each
    physical mode counted once.  Host-side and static: the count sets a scan length.
    """
    import itertools  # noqa: PLC0415

    out = []
    for n in itertools.product(range(-k_max, k_max + 1), repeat=dim):
        r2 = sum(c * c for c in n)
        if r2 == 0 or r2 > k_max * k_max:
            continue
        lead = next(c for c in n if c != 0)
        if lead > 0:
            out.append(n)
    return np.asarray(out, dtype=float)


def _broadband_body(x, kbox, u, mode):
    """One mode of the broadband sum; ``lax.scan`` body, bound with ``partial``."""
    import jax.numpy as jnp  # noqa: PLC0415

    n, amp, phase = mode
    return u + jnp.cos(kbox * (x @ n) + phase)[:, None] * amp, None


def broadband(x, u0, kbox, dim, key, k_max=4):
    """Random-phase band-limited velocity field at positions ``x``; ``f64[N, 3]``.

    Solenoidal for ``dim >= 2`` (each mode's amplitude is projected perpendicular to
    its own wavevector, so the sum is divergence-free exactly, not approximately) and
    compressive for ``dim == 1``, matching :func:`taylor_green`'s intent in each case.
    Normalised so the RMS speed is ``u0``, which is what the Mach and Reynolds numbers
    in :func:`sizing` are defined against.

    **This is the initial condition a spectrum should be measured from, and
    Taylor-Green is not.**  A single Fourier mode imprints its own symmetry on
    ``E(k)``: ``u = u0 sin(k x)`` is antisymmetric under ``x -> x + L/2``, the
    nonlinearity breaks that only slowly, and the measured 1-D spectrum came out with
    a factor-12 odd/even comb at shells standing 90x clear of the noise floor.  No
    shell-averaged slope can be fitted through that, and any threshold-selected window
    just samples the comb.  Random phases over a band remove the symmetry.

    Taylor-Green remains the right IC for the *viscosity* gate, where being an exact
    Navier-Stokes solution is the entire point.  See spec 22.4.
    """
    import functools  # noqa: PLC0415

    import jax.numpy as jnp  # noqa: PLC0415
    from jax import lax  # noqa: PLC0415

    n = jnp.asarray(band_wavevectors(k_max, dim))
    m = n.shape[0]
    k_a, k_p = jax.random.split(key)
    a = jax.random.normal(k_a, (m, 3), dtype=jnp.float64)
    phase = jax.random.uniform(k_p, (m,), dtype=jnp.float64) * 2.0 * jnp.pi
    # Project each amplitude perpendicular to its own wavevector over the SPATIAL
    # components only; the remaining components of the 3-vector are passive and are
    # zeroed rather than left as an unforced transverse velocity.
    nhat = n / jnp.linalg.norm(n, axis=1, keepdims=True)
    par = jnp.sum(a[:, :dim] * nhat, axis=1, keepdims=True) * nhat
    amp = jnp.zeros((m, 3)).at[:, :dim].set(a[:, :dim] - par if dim > 1 else a[:, :dim])
    # <|u|^2> = (1/2) sum |A_m|^2 exactly, over the random phases.
    scale = u0 / jnp.sqrt(0.5 * jnp.sum(amp**2))
    amp = amp * scale
    u, _ = lax.scan(
        functools.partial(_broadband_body, x, kbox),
        jnp.zeros((x.shape[0], 3), dtype=jnp.float64),
        (n, amp, phase),
    )
    return u


def sizing(dim, cells_per_side, npc, bytes_per_particle=64):
    """Memory and Reynolds number implied by a grid choice.  Pure arithmetic."""
    cells = cells_per_side**dim
    n_part = cells * npc
    return {
        "cells": cells,
        "n_part": n_part,
        "gib": n_part * bytes_per_particle / 2**30,
    }


def reach(dim, gib_budget, npc=50, mach=0.3, cells_per_mfp=1.0, bytes_per_particle=64):
    """Largest grid, and the Reynolds number, a memory budget supports.

    ``Re = u0 L / nu`` with ``L`` the Taylor-Green length ``box / (2 pi)`` and
    ``nu ~ (1/2) lambda v_th``, giving ``Re ~ 2 Ma sqrt(gamma) (L/lambda)``.
    The constant is order-unity, so this is a sizing estimate, not a prediction;
    the code reports the Reynolds number it actually ran from the measured
    viscosity.
    """
    n_part = gib_budget * 2**30 / bytes_per_particle
    cells = n_part / npc
    side = round(cells ** (1.0 / dim))
    box_mfp = side / cells_per_mfp
    ell_mfp = box_mfp / (2.0 * math.pi)
    re = 2.0 * mach * math.sqrt(GAMMA) * ell_mfp
    return {
        "side": side,
        "cells": side**dim,
        "n_part": side**dim * npc,
        "box_mfp": box_mfp,
        "Re": re,
    }
