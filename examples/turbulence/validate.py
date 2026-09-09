"""Turn a completed turbulence run into the numbers its reference predicts."""

from __future__ import annotations

import math

import numpy as np


def energy_history(mom, info):
    """Fluctuation kinetic energy per unit mass, ``f64[steps]``, noise-corrected.

    **The noise floor is the whole difficulty.**  A cell's bulk velocity is the mean
    of ``npc`` thermal velocities, so it carries an error of variance ``v_th^2/npc``
    per component whatever the flow is doing.  The measured energy is therefore

        E_meas(t) = E_signal(t) + (3/2) v_th^2 / npc

    and that offset is **constant in time** while the signal decays exponentially.
    Fitting ``log E_meas`` without removing it flattens the curve -- the fitted decay
    rate is biased low, increasingly so as the vortex decays, and the fit still looks
    excellent because a decaying exponential plus a constant is smooth.  It would
    report a viscosity that is too small and no diagnostic would fire.

    The correction uses the *actual* per-cell counts rather than the nominal ``npc``,
    since a compressible flow moves particles between cells.
    """
    u = np.asarray(mom.u)  # (steps, C, 3)
    cnt = np.asarray(mom.count) if hasattr(mom, "count") else None
    e_meas = 0.5 * np.mean(np.sum((u - u.mean(axis=1, keepdims=True)) ** 2, axis=-1), axis=-1)
    vth2 = info["vth"] ** 2
    # Averaging M independent FIELDS before taking moments divides the floor by M.
    m_ens = max(int(info.get("n_ensemble", 1)), 1)
    if cnt is not None and np.all(cnt > 0):
        floor = 1.5 * vth2 * np.mean(1.0 / np.maximum(cnt, 1.0), axis=-1) / m_ens
    else:
        floor = np.full(e_meas.shape, 1.5 * vth2 / (info["npc"] * m_ens))
    return e_meas - floor, floor


def fit_decay(t, e, t_lo=0.0, t_hi=None, floor=None):
    """Least-squares fit of ``E = E0 exp(-rate t)``; returns ``(rate, E0, r2)``.

    Only the window where the signal is still well above the subtracted noise floor
    is usable: once ``E_signal`` falls to the size of ``floor``, the corrected value
    is a difference of two comparable numbers and its log is meaningless.
    """
    m = (t >= t_lo) & np.isfinite(e) & (e > 0)
    if t_hi is not None:
        m &= t <= t_hi
    if floor is not None:
        m &= e > 0.25 * floor
    if m.sum() < 5:
        return float("nan"), float("nan"), float("nan")
    slope, icept = np.polyfit(t[m], np.log(e[m]), 1)
    pred = icept + slope * t[m]
    ss = 1.0 - np.var(np.log(e[m]) - pred) / max(np.var(np.log(e[m])), 1e-300)
    return -float(slope), float(math.exp(icept)), float(ss)


def report_2d(mom, info):
    """The 2-D gate: the fitted decay rate IS the viscosity.

    For the Taylor-Green field ``u = U0(sin kx cos ky, -cos kx sin ky)``, which is an
    exact solution of the incompressible Navier-Stokes equations, every velocity
    component decays as ``exp(-2 nu k^2 t)``, so the kinetic energy decays as

        E(t) = E(0) exp(-4 nu k^2 t).

    Nothing in the solver is told ``nu``; it emerges from the collision rate and the
    VSS scattering law, so this compares DSMC transport against Chapman-Enskog with
    no free parameter.  The expected systematic is compressibility, which enters at
    ``O(Ma^2)`` -- about 9 % at Ma = 0.3 -- so that, not statistics, sets the
    tolerance.

    **The answer is reported three ways, over two windows, and that is the point.**
    A result that *depends* on the noise correction is a result resting on the
    correction being right, so the correction has to be shown not to matter rather
    than assumed harmless.  But comparing "corrected over the full decay" against
    "uncorrected over the clean window" confounds two things -- the correction, and
    the window length -- and an earlier version did exactly that, reporting a 7.9
    point disagreement as a correction failure when most of it was the extra
    late-time range.  So:

    * **uncorrected, clean window** (``E_meas > 5 * floor``) -- the headline. It
      leans on nothing.
    * **corrected, same window** -- differs from the first only by the correction, so
      the two together say whether the correction is verified or load-bearing.
    * **corrected, full decay** -- differs from the second only by the window, so it
      says what the extra range costs. At Ma = 0.3 an initial acoustic transient and
      the approach to the floor both live there.

    If the first two disagree, raise ``npc`` or use ``run_ensemble``; do not pick one.
    """
    e, floor = energy_history(mom, info)
    t = np.arange(e.size) * info.get("snap_dt", info["dt"])
    k = info["kbox"]
    rate_exact = 4.0 * info["nu"] * k * k

    # The clean window: where the MEASURED energy stands 5x clear of the floor, so
    # that subtracting it is a small correction rather than a difference of
    # comparable numbers.  Chosen from `raw`, which needs no correction to define.
    raw = e + floor
    hi = raw > 5.0 * floor
    t_hi = t[hi].max() if hi.any() else 0.0

    rate_raw, _e1, r2_raw = fit_decay(t, raw, t_hi=t_hi)
    rate_win, _e2, r2_win = fit_decay(t, e, t_hi=t_hi)
    rate_full, _e0, r2_full = fit_decay(t, e, floor=floor)

    nu_raw = rate_raw / (4.0 * k * k)
    nu_win = rate_win / (4.0 * k * k)
    nu_full = rate_full / (4.0 * k * k)
    out = {
        "rate": rate_raw,
        "rate_exact": rate_exact,
        "nu_meas": nu_raw,
        "nu_raw": nu_raw,
        "nu_win": nu_win,
        "nu_full": nu_full,
        "nu_exact": info["nu"],
        "r2": r2_raw,
        "t_hi": t_hi,
        "err_pct": 100.0 * (nu_raw / info["nu"] - 1.0),
        "err_pct_raw": 100.0 * (nu_raw / info["nu"] - 1.0),
        "err_pct_win": 100.0 * (nu_win / info["nu"] - 1.0),
        "err_pct_full": 100.0 * (nu_full / info["nu"] - 1.0),
    }
    n_clean = int(hi.sum())
    print("\n  2-D Taylor-Green: EXACT Navier-Stokes decay  E(t) = E0 exp(-4 nu k^2 t)")
    print(f"    nu Chapman-Enskog (Bird 4.64)                {info['nu']:.4e} m^2/s")
    print(
        f"    nu, NO correction, clean window ({n_clean} pts)   {nu_raw:.4e}"
        f"   ({out['err_pct']:+.2f} %, R^2 {r2_raw:.4f})   <- headline"
    )
    print(
        f"    nu, corrected, SAME window                   {nu_win:.4e}"
        f"   ({out['err_pct_win']:+.2f} %, R^2 {r2_win:.4f})"
    )
    print(
        f"    nu, corrected, full decay                    {nu_full:.4e}"
        f"   ({out['err_pct_full']:+.2f} %, R^2 {r2_full:.4f})"
    )
    spread = abs(out["err_pct"] - out["err_pct_win"])
    if not np.isfinite(out["err_pct"]):
        print(
            "    ** the uncorrected fit has NO usable window: the signal never stands 5x clear "
            "of the floor,\n       so this result rests entirely on the noise correction.  "
            f"Raise npc (now {info['npc']}) or use run_ensemble. **"
        )
    elif spread > 3.0:
        print(
            f"    ** on the SAME window the two differ by {spread:.1f} points, so the answer "
            "rests on the\n       noise correction.  Raise npc or use run_ensemble until they "
            "agree. **"
        )
    else:
        print(
            f"    -> on the same window they agree to {spread:.2f} points, so the noise "
            "correction is verified, not load-bearing"
        )
    window_spread = abs(out["err_pct_win"] - out["err_pct_full"])
    if np.isfinite(window_spread) and window_spread > 3.0:
        print(
            f"    -> the full decay differs by a further {window_spread:.1f} points: that is the "
            "WINDOW, not the\n       correction (acoustic transient at early t, the floor at "
            "late t)"
        )
    print(
        f"    expected systematic: O(Ma^2) = {100 * info['mach'] ** 2:.0f} % from compressibility"
    )
    return out


def report_spectrum(mom, info, at_step=None, n_avg=1, u_field=None):
    """Shell-averaged ``E(k)`` and the noise floor it must stand above.

    Args:
        u_field: optional ``f64[C, 3]`` velocity field to transform instead of
            ``mom.u[at_step]`` -- this is where an **ensemble-averaged** field goes.

    **The floor is not subtracted, and that is deliberate.**  It is tempting: it is a
    known positive bias, exactly as in :func:`energy_history`, and removing it
    appears to extend the usable range by an order of magnitude in ``k``.  It does
    not.  Above the crossover ``E(k) - noise`` is a difference of two comparable
    numbers that fluctuates through zero, so roughly half the shells survive any
    positivity test by chance and the fitted slope is set by that survivorship, not
    by the flow.  Measured on the 1-D run here: subtraction reported "usable to
    k = 507" with a slope of **-0.71**, against a true Burgers ``-2``, and looked
    entirely healthy.  A bias correction that is safe on a scalar (where the signal
    stays far above the floor) is not safe under a logarithm near the crossover.

    The way to see more decades is to **lower the floor**, which is
    ``1.5 v_th^2 / npc`` per cell:

    * more particles per cell -- the floor falls as ``1/npc``; or
    * **ensemble-average the velocity field over independent seeds** before
      transforming.  Taylor-Green and the 1-D sinusoid have deterministic initial
      conditions, so the mean field is reproducible and only the collision noise
      differs between seeds.  Averaging ``M`` fields is exactly equivalent to
      ``M * npc``, and it parallelises perfectly across devices.

    Returns ``(k, E, noise)``.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    from jax_dsmc.sample import energy_spectrum, spectrum_bins  # noqa: PLC0415

    if u_field is not None:
        E = np.asarray(energy_spectrum(jnp.asarray(u_field), info["n"]))
        n_fields = int(info.get("n_ensemble", 1))
    else:
        u = np.asarray(mom.u)
        at = at_step if at_step is not None else u.shape[0] - 1
        lo = max(0, at - n_avg + 1)
        E = np.mean(
            [np.asarray(energy_spectrum(jnp.asarray(u[i]), info["n"])) for i in range(lo, at + 1)],
            axis=0,
        )
        n_fields = 1
    idx, nb = spectrum_bins(info["n"])
    modes = np.bincount(np.asarray(idx), minlength=nb + 1)[:nb].astype(float)
    # White per-cell noise of variance vth^2/npc per component spreads its total
    # energy 1.5 vth^2/npc uniformly over the C modes, so shell k gets its share.
    # Averaging M independent FIELDS before transforming divides it by M.
    total_noise = 1.5 * info["vth"] ** 2 / (info["npc"] * max(n_fields, 1))
    noise = total_noise * modes / max(float(modes.sum()), 1.0)
    return np.arange(nb), E, noise


def clean_window(k, E, noise, headroom=10.0, above_peak=True, k_forcing=0):
    """The **contiguous** run of shells, above the spectral peak, that clears the floor.

    Returns a boolean mask over ``k``.

    Two selection rules, both of which this gate got wrong at some point:

    *Contiguity.*  Selecting every shell that passes ``E > headroom * noise`` lets
    the shells whose noise happened to add survive while their neighbours drop
    out, so the fitted slope measures that survivorship rather than the flow.
    Measured on the 16-seed 1-D run here: the scattered selection admitted
    ``k = 7, 9, 10`` while skipping 6 and 8 and reported **-3.06**; the contiguous
    window ``k = 1..5`` gives **-1.76**.

    *Above the peak.*  An inertial (or Burgers) range is by definition below the
    energy-containing scales, so a window that starts at the box mode is not one.
    Dropping ``k <= k_peak`` is what turns that same 1-D run's plausible-looking
    "-1.76, expected -2" into the correct verdict of no measurable range at all:
    the peak sits at ``k = 2`` and the signal dies by ``k = 6``, leaving three
    shells.

    ``k_forcing`` extends the second rule for a broadband initial condition: energy
    is injected across ``1 <= |k| <= k_max``, so the whole band is energy-containing
    whether or not the instantaneous peak happens to sit inside it, and the window
    must start above ``k_max``.
    """
    ok = headroom * noise < E
    ok = ok & (k > 0)
    stop = int(np.argmin(ok[1:])) + 1 if not ok[1:].all() else len(k)
    mask = np.zeros_like(ok)
    mask[1:stop] = True
    mask = mask & ok
    if above_peak and mask.any():
        peak = k[1:][int(np.argmax(E[1:]))]
        mask = mask & (k > max(peak, k_forcing))
    elif k_forcing:
        mask = mask & (k > k_forcing)
    return mask


# Prefactors on the dissipation-wavenumber scalings.  ONLY the 1-D entry is
# measured: on the Re = 61 ensemble run the k^-2 tail tracked the power law to
# k = 9 and had collapsed by k = 13, so k_d ~ Re/6.  The 2-D and 3-D entries are
# placeholders of 1, i.e. the bare textbook scaling with its O(1) constant unknown.
# Do not read them as calibrated -- an earlier version exported the 1-D 1/8 to all
# three dimensions and consequently reported "NO RANGE" for every H200 configuration,
# which contradicts the standard DNS resolution requirement.
_KD_PREFACTOR = {1: 1.0 / 6.0, 2: 1.0, 3: 1.0}


def dissipation_wavenumber(re, dim):
    """Sizing estimate for the smallest resolvable box-wavenumber, ``k_d / k_box``.

    ``k`` here is the shell index, so ``k = 1`` is the box mode.  The exponents are
    standard -- 1-D Burgers shock thickness ``~ nu/u`` gives ``k_d ~ Re``; the 2-D
    enstrophy cascade gives ``Re^(1/2)``; 3-D Kolmogorov gives ``Re^(3/4)``.

    Sizing arithmetic, not a measurement.  See ``_KD_PREFACTOR`` for which of the
    three constants has actually been measured (one of them).
    """
    return re ** {1: 1.0, 2: 0.5, 3: 0.75}[dim] * _KD_PREFACTOR[dim]


def fit_slope(k, e, sel):
    """``(slope, stderr, r2)`` of ``log E`` against ``log k`` over ``sel``.

    **The standard error is not decoration.**  In D dimensions shell ``k`` holds
    ``~k^(D-1)`` modes, so it is an average over that many random amplitudes -- but
    in 1-D it holds exactly **two** (``k`` and ``-k``) at every ``k``, i.e. one
    complex amplitude, so shell-averaging averages nothing and ``E(k)`` inherits the
    full scatter of a single realisation.  For a Gaussian field that is an
    exponential distribution, ``std(log E) = pi/sqrt(6) = 1.28``, independent of
    particle count.  Measured on the Re = 204 broadband run: **1.00**.

    The consequence is that a 1-D slope carries a large error bar however many
    particles are used -- the Re = 204 run gives ``-2.87 +- 0.65`` over 15 shells,
    which is consistent with Burgers ``-2`` and also with a great deal else.  More
    particles do not shrink it; only more independent *flow* realisations do (see
    :func:`examples.turbulence.case.run_ic_ensemble`).  2-D and 3-D are far better
    off: at ``k = 19`` a 128^2 grid has 116 modes per shell and a 32^3 grid has
    thousands.
    """
    x, y = np.log(k[sel]), np.log(e[sel])
    n = x.size
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    slope, icept = np.polyfit(x, y, 1)
    resid = y - (icept + slope * x)
    sxx = float(np.sum((x - x.mean()) ** 2))
    s_res = float(np.sqrt(np.sum(resid**2) / (n - 2)))
    r2 = 1.0 - np.var(resid) / max(np.var(y), 1e-300)
    return float(slope), s_res / math.sqrt(max(sxx, 1e-300)), float(r2)


def average_spectra(fields, info):
    """Mean ``E(k)`` over independent IC-phase realisations; ``(k, E, noise)``.

    Averaging the **spectra** of independent flows is a different operation from
    averaging the **fields** of one flow across noise seeds, and in 1-D it is the one
    that matters.  Field averaging (:func:`run_ensemble`) lowers the particle-noise
    floor; it cannot touch the scatter described in :func:`fit_slope`, because every
    seed evolves the *same* flow and therefore the same realisation.  The Re = 204
    run spent 2.3 h of ensemble averaging driving a floor that was already 1600x
    below the signal.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    from jax_dsmc.sample import energy_spectrum, spectrum_bins  # noqa: PLC0415

    spectra = [np.asarray(energy_spectrum(jnp.asarray(f), info["n"])) for f in fields]
    idx, nb = spectrum_bins(info["n"])
    modes = np.bincount(np.asarray(idx), minlength=nb + 1)[:nb].astype(float)
    n_seeds = int(info.get("n_ensemble", 1))
    total = 1.5 * info["vth"] ** 2 / (info["npc"] * max(n_seeds, 1))
    noise = total * modes / max(float(modes.sum()), 1.0)
    return np.arange(nb), np.mean(spectra, axis=0), noise
