"""Decaying-turbulence driver: run, validate, and write frames for an animation.

Examples::

    # 2-D viscosity gate (the strongest quantitative check in the project)
    python3 -m examples.turbulence --dim 2 --side 128 --npc 100 --efolds 2.5

    # 3-D, with a frame every 20 steps on the three mid-planes, for a GIF
    python3 -m examples.turbulence --dim 3 --side 64 --npc 60 --n-eddy 6 \
        --frame-every 20 --movie-dir examples/turbulence/figures/d3

    # 1-D compressive case, for the E(k) ~ k^-2 spectrum
    python3 -m examples.turbulence --dim 1 --side 4096 --npc 200 --n-eddy 3 --spectrum

Stitch the frames with::

    ffmpeg -framerate 20 -pattern_type glob -i 'field_z*.png' -vf scale=900:-1 out.gif
"""

from __future__ import annotations

import argparse
import pickle
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from examples.turbulence import reach  # noqa: E402
from examples.turbulence.case import (  # noqa: E402
    run,
    run_ensemble,
    run_ic_ensemble,
    setup,
)
from examples.turbulence.validate import (  # noqa: E402
    average_spectra,
    clean_window,
    dissipation_wavenumber,
    energy_history,
    fit_decay,
    fit_slope,
    report_2d,
    report_spectrum,
)


def main():  # noqa: PLR0912, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--side", type=int, default=64)
    ap.add_argument("--npc", type=int, default=100)
    ap.add_argument("--mach", type=float, default=0.3)
    ap.add_argument("--cells-per-mfp", type=float, default=1.0)
    ap.add_argument(
        "--ic",
        choices=("taylor-green", "broadband"),
        default="taylor-green",
        help="taylor-green for the VISCOSITY gate (exact NS in 2-D); "
        "broadband for any SPECTRUM (a single mode combs E(k) -- spec 22.4)",
    )
    ap.add_argument(
        "--k-max", type=int, default=4, help="broadband forcing band, 1 <= |k| <= k_max"
    )
    ap.add_argument(
        "--ic-ensemble",
        type=int,
        default=1,
        help="independent FLOW realisations (different IC phases), spectra averaged. "
        "This is the ensemble a 1-D spectrum needs; --ensemble is the one a decay "
        "rate needs.  See validate.fit_slope",
    )
    ap.add_argument("--n-eddy", type=float, default=None)
    ap.add_argument("--efolds", type=float, default=None, help="run this many viscous e-folds")
    ap.add_argument("--stride", type=int, default=8, help="keep every Nth step's moments")
    ap.add_argument("--a-max", type=int, default=64, help="initial per-cell attempt capacity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-every", type=int, default=0, help="in kept SNAPSHOTS, not steps")
    ap.add_argument("--movie-dir", default=None)
    ap.add_argument("--spectrum", action="store_true")
    ap.add_argument(
        "--ensemble",
        type=int,
        default=1,
        help="average the velocity field over this many seeds before "
        "the spectrum; equivalent to that many times npc",
    )
    ap.add_argument("--save", default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--reach", action="store_true", help="print sizing table and exit")
    a = ap.parse_args()

    if a.reach:
        print(
            "memory budget -> reachable grid, Reynolds number, and spectral range (npc 50, Ma 0.3)"
        )
        print("  k_d is the dissipation wavenumber in box modes: a slope needs k_d >> 1,")
        print("  and NO particle count can substitute for it (spec 22.4).")
        for gib, label in ((113.0, "1 x H200"), (226.0, "2 x H200"), (450.0, "4 x H200")):
            for d in (1, 2, 3):
                r = reach(d, gib)
                k_d = dissipation_wavenumber(r["Re"], d)
                verdict = "decades" if k_d > 30 else ("marginal" if k_d > 8 else "NO RANGE")
                print(
                    f"  {label:>9}  D={d}: side {r['side']:>9,}  N {r['n_part']:.2e}"
                    f"  box {r['box_mfp']:.0f} mfp  Re ~ {r['Re']:>8.0f}"
                    f"  k_d ~ {k_d:>7.0f}  {verdict}"
                )
        return None

    kw = {
        "cells_per_side": a.side,
        "npc": a.npc,
        "mach": a.mach,
        "cells_per_mfp": a.cells_per_mfp,
        "a_max": a.a_max,
        "ic": a.ic,
        "k_max": a.k_max,
    }
    n_steps = None
    if a.efolds is not None:
        _, _, _, _, _, i0 = setup(dim=a.dim, **kw)
        rate = 4 * i0["nu"] * i0["kbox"] ** 2
        n_steps = int(a.efolds / rate / i0["dt"])
        print(f"{a.efolds} viscous e-folds -> {n_steps} steps")
    n_eddy = a.n_eddy if a.n_eddy is not None else 4.0

    t0 = time.time()
    u_ens = None
    fields = None
    if a.ic_ensemble > 1:
        fields, info = run_ic_ensemble(
            realisations=a.ic_ensemble,
            seeds=a.ensemble,
            dim=a.dim,
            n_eddy=n_eddy,
            n_steps=n_steps,
            sample_stride=a.stride,
            **kw,
        )
        mom = None
        info["wall"] = time.time() - t0
        print(
            f"  wall {info['wall']:.0f} s   {a.ic_ensemble} flow realisations"
            + (f" x {a.ensemble} noise seeds" if a.ensemble > 1 else "")
        )
    elif a.ensemble > 1:
        # Always keep the whole time series.  A spectrum only needs the final
        # snapshot, and an earlier version skipped the series for that case to save
        # memory -- but `run_ensemble` accumulates in place, so its peak is one run's
        # series either way, and skipping it left `mom = None` for every downstream
        # consumer.  The 1-D ensemble spectrum run then died in `energy_history`
        # after completing, printing nothing and saving nothing.
        res, info = run_ensemble(
            seeds=a.ensemble,
            full=True,
            dim=a.dim,
            n_eddy=n_eddy,
            n_steps=n_steps,
            sample_stride=a.stride,
            **kw,
        )
        mom = res
        u_ens = np.asarray(res.u)[-1]
        info["wall"] = time.time() - t0
        print(
            f"  wall {info['wall']:.0f} s   ensemble of {a.ensemble} "
            f"-> effective npc {a.npc * a.ensemble:,}"
        )
    else:
        mom, info = run(
            dim=a.dim,
            seed=a.seed,
            n_eddy=n_eddy,
            n_steps=n_steps,
            sample_stride=a.stride,
            frame_every=a.frame_every or None,
            movie_dir=a.movie_dir,
            **kw,
        )
        info["wall"] = time.time() - t0
    print(
        f"  wall {info['wall']:.0f} s   clipped {info['clipped']}   violations {info['violations']}"
    )
    if info["clipped"]:
        print("  ** attempts were clipped: under-collision.  Raise --a-max. **")

    if mom is None:
        print("\n  (no decay rate: --ic-ensemble keeps only the final field of each realisation)")
    elif a.dim == 2:
        report_2d(mom, info)
    else:
        e, floor = energy_history(mom, info)
        if e.size < 3:
            # A run this short keeps fewer snapshots than the fit needs.  Say so:
            # the previous behaviour was an IndexError from inside the report.
            print(
                f"\n  ** only {e.size} snapshot(s) kept -- too few to fit a decay rate."
                "  Raise --n-eddy / --efolds, or lower --stride. **"
            )
        else:
            t = np.arange(e.size) * info["snap_dt"]
            rate, e0, r2 = fit_decay(t, e, floor=floor)
            print(f"\n  kinetic energy decay rate {rate:.4e} 1/s  (R^2 {r2:.4f}), E0 {e0:.4e}")
            print(f"  E(0)/noise floor = {e[0] / floor[0]:.1f}")
        if a.dim == 3:
            print(
                "  3-D has no closed form; compare against NS-DNS at Re = "
                f"{info['Re']:.0f} (the Reynolds number ACTUALLY run, not 1600)"
            )

    if a.spectrum:
        if fields is not None:
            k, E, noise = average_spectra(fields, info)
        else:
            k, E, noise = report_spectrum(mom, info, n_avg=min(8, mom.u.shape[0]), u_field=u_ens)
        expect = {1: -2.0, 2: -3.0, 3: -5.0 / 3.0}[a.dim]
        # TWO independent limits have to be cleared before a slope means anything,
        # and they are lifted by different knobs.  Report both, always.
        #
        #   1. Viscosity must not cut the cascade off before it starts.  This is set
        #      by Re alone -- no number of particles can fix it.
        #   2. The signal must stand clear of the sampling floor out to k_d.  This
        #      is set by npc * ensemble.
        k_d = dissipation_wavenumber(info["Re"], a.dim)
        print(
            f"\n  dissipation wavenumber k_d ~ {k_d:.1f}"
            f"   (sizing estimate at Re = {info['Re']:.0f}; see dissipation_wavenumber)"
        )
        if k_d < 4.0:
            print(
                "  ** NO SPECTRAL RANGE CAN EXIST at this Reynolds number, at any particle\n"
                "     count -- viscosity ends the cascade before k = 4.  Raise Re (more cells,\n"
                "     since Re ~ L/lambda), not npc.  The noise floor is not the binding limit. **"
            )
        # The window must be CONTIGUOUS from k = 1.  Taking every shell that passes
        # the threshold lets the ones whose noise happened to add survive while their
        # neighbours drop out, and the fitted slope then measures that survivorship:
        # on this run the scattered selection gave -3.06 over a ragged k = 2..10, the
        # contiguous k = 1..5 gives -1.76.  See validate.clean_window.
        k_f = a.k_max if a.ic == "broadband" else 0
        sel = clean_window(k, E, noise, headroom=10.0, k_forcing=k_f)
        kk = k[sel]
        # No factor-4-in-k rule any more.  That was a proxy for "the fit is well
        # conditioned", and `fit_slope` measures exactly that, directly, as a standard
        # error -- a proxy that rejected a k = 5..19 window for spanning 3.8x while
        # saying nothing about the +-0.65 that actually made it untestable.
        if sel.sum() >= 5:
            slope, err, r2s = fit_slope(k, E, sel)
            # Quote the uncertainty, always.  In 1-D every shell holds exactly two
            # modes, so E(k) carries ~100 % scatter no matter how many particles are
            # used, and a bare slope hides an error bar of +-0.65 (validate.fit_slope).
            n_r = int(info.get("n_realisations", 1))
            print(
                f"  E(k) slope over k = {kk[0]:.0f}..{kk[-1]:.0f} "
                f"(contiguous, 10x clear of the floor, {n_r} realisation"
                f"{'s' if n_r > 1 else ''}):\n"
                f"    {slope:+.2f} +- {err:.2f}   (expected ~{expect}, R^2 {r2s:.3f})"
            )
            if not np.isfinite(err) or err > 0.3:
                print(
                    f"    ** +-{err:.2f} is too wide to test {expect}: this is realisation "
                    "scatter, not\n       particle noise, so raise --ic-ensemble (error ~ "
                    f"1/sqrt(R)), not --npc. **"
                )
            elif abs(slope - expect) > 2.0 * err:
                print(
                    f"    ** {abs(slope - expect) / err:.1f} sigma from {expect}: a real "
                    "disagreement, not scatter. **"
                )
        else:
            top = int(kk.max()) if sel.any() else 0
            print(
                f"  ** NO SPECTRUM ESTABLISHED.  The signal stands 10x clear of the floor\n"
                f"     contiguously only to k = {top}, too narrow to fit a slope "
                f"(expected ~{expect}).\n"
                f"     Floor: raise --npc, or --ensemble M (M fields averaged == M*npc, "
                f"= {a.npc * a.ensemble:,} here).\n"
                f"     Range: raise Re, which needs more cells (--side). **"
            )
        if a.save:
            np.savez(a.save + "_spec.npz", k=k, E=E, noise=noise)

    if a.save:
        np.savez_compressed(
            a.save + ".npz",
            u=np.asarray(mom.u),
            nrho=np.asarray(mom.nrho),
            temp=np.asarray(mom.temp),
            count=np.asarray(mom.count),
        )
        with open(a.save + ".pkl", "wb") as f:
            pickle.dump(info, f)
        print(f"  saved {a.save}.npz")
    return info


if __name__ == "__main__":
    main()
