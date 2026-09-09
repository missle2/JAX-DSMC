"""The spectral-gate window rules (spec 22.5).

These are regression tests for a class of error that produces no exception and no
NaN: a fit window selected using the data being fitted reports a slope whatever the
data is.  Both rules below changed a headline number on the 16-seed 1-D run.
"""

import os
import sys

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.turbulence import broadband  # noqa: E402
from examples.turbulence.case import initial_particles, setup  # noqa: E402
from examples.turbulence.validate import (  # noqa: E402
    clean_window,
    dissipation_wavenumber,
    fit_slope,
)


def _synthetic(n=64, slope=-2.0, peak=2, floor=1e-6):
    """A clean power law above ``peak``, with a floor far below it."""
    k = np.arange(n)
    e = np.zeros(n)
    e[1:] = np.where(k[1:] <= peak, 10.0 * k[1:], 10.0 * peak * (k[1:] / peak) ** slope)
    return k, e, np.full(n, floor)


def test_a_clean_power_law_is_accepted_and_its_slope_recovered():
    # The gate must not simply refuse everything: given a real range it passes.
    k, e, noise = _synthetic()
    sel = clean_window(k, e, noise)
    assert sel.sum() > 20
    assert k[sel].min() == 3  # first shell strictly above the peak at k = 2
    fit = np.polyfit(np.log(k[sel]), np.log(e[sel]), 1)[0]
    assert fit == pytest.approx(-2.0, abs=1e-6)


def test_window_stops_at_the_first_shell_below_the_floor():
    # Shells 7, 9, 10 clear the floor while 6 and 8 do not.  Taking every shell
    # that passes admits the lucky ones and the slope then measures survivorship;
    # on the real run that scattered selection gave -3.06 against a contiguous
    # -1.76.  The window must stop at 6.
    k, e, noise = _synthetic(n=16, floor=1.0)
    e[6] = 0.5
    e[8] = 0.5
    sel = clean_window(k, e, noise, headroom=1.0)
    assert k[sel].max() == 5
    assert not sel[7] and not sel[9]


def test_window_excludes_the_energy_containing_scales():
    # An inertial range lies below the peak by definition, so a window that starts
    # at the box mode is not one.
    k, e, noise = _synthetic(peak=4)
    sel = clean_window(k, e, noise)
    assert k[sel].min() == 5
    assert not sel[:5].any()
    assert clean_window(k, e, noise, above_peak=False)[1]


def test_a_narrow_range_is_reported_as_narrow_not_fitted():
    # The measured 1-D case: peak at k = 2, signal gone by k = 6.
    k = np.arange(16)
    e = np.zeros(16)
    e[1:6] = [117.0, 182.0, 27.9, 10.8, 11.8]
    e[6:] = 0.03
    noise = np.full(16, 0.052)
    sel = clean_window(k, e, noise)
    assert list(k[sel]) == [3, 4, 5]
    assert sel.sum() < 5  # too few to quote -- what __main__ checks


@pytest.mark.parametrize(("dim", "exponent"), [(1, 1.0), (2, 0.5), (3, 0.75)])
def test_dissipation_wavenumber_scales_with_reynolds_number(dim, exponent):
    assert dissipation_wavenumber(10_000.0, dim) / dissipation_wavenumber(
        100.0, dim
    ) == pytest.approx(100.0**exponent)


def test_reynolds_number_of_the_1d_run_admits_no_spectral_range():
    # Re = 61 puts dissipation at k ~ 10, against an energy peak at k = 2: under a
    # decade, and no particle count can widen it.  Spec 22.4.
    assert dissipation_wavenumber(61.1, 1) < 12.0
    assert dissipation_wavenumber(1200.0, 1) > 100.0  # the sizing target


def test_the_1d_prefactor_is_not_exported_to_2d_and_3d():
    # It was, once, and the reach table then said "NO RANGE" for every H200
    # configuration -- contradicting the standard DNS resolution requirement, which
    # is exactly k_d/k_0 ~ Re^(3/4) with an O(1) constant.
    assert dissipation_wavenumber(66.0, 3) > 20.0
    assert dissipation_wavenumber(1515.0, 2) > 30.0


def test_broadband_window_starts_above_the_forcing_band():
    # With energy injected across 1 <= |k| <= k_max the whole band is
    # energy-containing, whether or not the instantaneous peak sits inside it.
    k, e, noise = _synthetic(peak=2)
    e[3:5] *= 3.0  # a bump inside a k_max = 6 band that leaves the peak at k = 2
    sel = clean_window(k, e, noise, k_forcing=6)
    assert k[sel].min() == 7


def _cell_mean_rms(x, u, grid, n, dim):
    """RMS over cells of the per-cell mean of ``u``; host-side, no solver needed."""
    x = np.asarray(x)
    hi = np.asarray(grid.hi)
    idx = np.zeros(x.shape[0], dtype=int)
    for d in range(dim):
        idx = idx * n + np.clip((x[:, d] / hi[d] * n).astype(int), 0, n - 1)
    cnt = np.maximum(np.bincount(idx, minlength=n**dim), 1)
    means = [
        np.bincount(idx, weights=np.asarray(u)[:, c], minlength=n**dim) / cnt for c in range(3)
    ]
    return float(np.sqrt(np.mean(np.square(means))))


def test_broadband_ic_is_identical_across_seeds():
    """The ensemble average is only valid if every seed starts from the same flow.

    Silent failure mode: derive the broadband phases from the run key and each seed
    gets a *different* flow, so averaging M realisations drives the mean field toward
    zero -- while the energy history, the spectrum and every clipping diagnostic still
    report cleanly.  The phases come from ``info["ic_phase_seed"]`` instead.

    Tested by subtracting the field that *should* be there from each seed's particles
    and checking the residual is thermal noise (``v_th/sqrt(npc)``) rather than a
    leftover flow (``u0``).  Those differ by 8x here, so the test has real margin.
    """
    n, npc = 8, 400
    _sp, grid, static, _p, _c, info = setup(
        dim=2, cells_per_side=n, npc=npc, ic="broadband", k_max=2
    )
    noise = info["vth"] / np.sqrt(npc)
    for seed in (7000, 7001):
        st = initial_particles(jax.random.PRNGKey(seed), grid, static, info)
        field = broadband(
            st.x,
            info["u0"],
            info["kbox"],
            2,
            jax.random.PRNGKey(info["ic_phase_seed"]),
            k_max=info["k_max"],
        )
        resid = _cell_mean_rms(st.x, st.v - field, grid, n, 2)
        assert resid < 2.0 * noise, f"seed {seed}: residual {resid:.1f} vs noise {noise:.1f}"

    # Control: subtracting a DIFFERENT phase draw must leave a flow-sized residual,
    # so the assertion above is testing something.
    st = initial_particles(jax.random.PRNGKey(7000), grid, static, info)
    wrong = broadband(st.x, info["u0"], info["kbox"], 2, jax.random.PRNGKey(99), k_max=2)
    assert _cell_mean_rms(st.x, st.v - wrong, grid, n, 2) > 4.0 * noise


def test_slope_stderr_matches_the_scatter_it_claims_to_describe():
    """``fit_slope``'s error bar must be calibrated, or it is worse than none.

    Synthetic 1-D spectra: a true ``k^-2`` law times exponential noise, which is what
    a shell holding one complex Gaussian amplitude gives.  The reported standard
    error has to match the empirical spread of the fitted slopes.
    """
    rng = np.random.default_rng(0)
    k = np.arange(64)
    sel = (k >= 5) & (k <= 19)
    slopes, errs = [], []
    for _ in range(400):
        e = np.zeros(64)
        e[1:] = k[1:] ** -2.0 * rng.exponential(1.0, size=63)
        s, err, _ = fit_slope(k, e, sel)
        slopes.append(s)
        errs.append(err)
    empirical = float(np.std(slopes, ddof=1))
    claimed = float(np.mean(errs))
    assert np.mean(slopes) == pytest.approx(-2.0, abs=0.1)
    assert claimed == pytest.approx(empirical, rel=0.2), f"{claimed:.3f} vs {empirical:.3f}"
    # And it is large: ~0.6 over this window, which is why a bare 1-D slope is not a
    # measurement.  Spec 22.4.
    assert 0.3 < empirical < 1.2


def test_averaging_independent_realisations_shrinks_the_slope_error():
    """The error falls as 1/sqrt(R) when the *spectra* of independent flows are averaged.

    This is the claim ``--ic-ensemble`` rests on, and it is a different operation
    from averaging fields across noise seeds, which cannot reduce this scatter at all.
    """
    rng = np.random.default_rng(1)
    k = np.arange(64)
    sel = (k >= 5) & (k <= 19)
    spread = {}
    for n_real in (1, 4, 16):
        slopes = []
        for _ in range(300):
            e = np.zeros(64)
            e[1:] = k[1:] ** -2.0 * rng.exponential(1.0, size=(n_real, 63)).mean(axis=0)
            slopes.append(fit_slope(k, e, sel)[0])
        spread[n_real] = float(np.std(slopes, ddof=1))
    assert spread[4] == pytest.approx(spread[1] / 2.0, rel=0.25)
    assert spread[16] == pytest.approx(spread[1] / 4.0, rel=0.3)
