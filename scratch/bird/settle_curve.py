"""Shock thickness as a function of how long the wave has been allowed to settle.

One long run, sliced into consecutive travel-distance blocks.  Each block is
aligned and measured independently, so the result is the whole settling curve
rather than a single point -- and a single point cannot distinguish "not yet
steady" from "steady but wrong", which is the question.
"""

import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from examples.shock.normal import (  # noqa: E402
    BIRD_L_OVER_LAMBDA_VSS,
    max_slope_thickness,
    shock_frame_average,
)

case = sys.argv[1]
nblock = int(sys.argv[2]) if len(sys.argv) > 2 else 6
d = np.load(case + ".npz")
info = pickle.load(open(case + ".pkl", "rb"))
rho, temp, ux = d["rho"], d["temp"], d["ux"]
per = info["n_per_seed"]
seeds = rho.shape[0] // per
n1, n2 = info["nrho"], info["nrho"] * info["rho2_over_rho1"]
snap_dt = info.get("snap_dt", info["dt"])
settle0 = info["n_settle"] * info["dt"] * info["u_s"] / info["lam1"]

print(f"### {case}: {seeds} seeds x {per} snapshots, settle starts at {settle0:.0f} mfp")
print(f"    travel per snapshot block = {per / nblock * snap_dt * info['u_s'] / info['lam1']:.1f} mfp\n")
print(f"{'block':>6} {'travel (mfp)':>16} {'L/lam @h=0':>11} {'h=1.25':>8} {'2.25':>7} {'3.25':>7} {'u_s fit':>9}")

hs = np.array([5, 9, 13, 17]) * info["dx"] / info["lam1"]
for b in range(nblock):
    lo, hi = b * per // nblock, (b + 1) * per // nblock
    sub = np.concatenate([rho[s * per + lo : s * per + hi] for s in range(seeds)])
    subt = np.concatenate([temp[s * per + lo : s * per + hi] for s in range(seeds)])
    subu = np.concatenate([ux[s * per + lo : s * per + hi] for s in range(seeds)])
    i2 = dict(info)
    i2["n_per_seed"] = hi - lo
    xi, r, t, u, used = shock_frame_average(sub, subt, subu, i2, align="fit")
    Ls = np.array(
        [max_slope_thickness(xi * info["lam1"], r, n1, n2, smooth_cells=k)[0] / info["lam1"]
         for k in (5, 9, 13, 17)]
    )
    # boxcar smoothing adds width in quadrature; extrapolate to zero smoothing
    A = np.polyfit(hs**2, Ls**2, 1)
    L0 = float(np.sqrt(max(A[1], 0.0)))
    t_mid = settle0 + (lo + hi) / 2 * snap_dt * info["u_s"] / info["lam1"]
    print(
        f"{b:>6} {t_mid:>16.0f} {L0:>11.2f} {Ls[0]:>8.2f} {Ls[1]:>7.2f} {Ls[2]:>7.2f}"
        f" {i2['u_s_fit']:>9.1f}"
    )
print(f"\n  Bird 12.11: L/lambda_1 = {BIRD_L_OVER_LAMBDA_VSS}   (exact u_s = {info['u_s']:.1f} m/s)")
