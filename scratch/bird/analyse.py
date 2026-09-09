"""Compare shock-frame alignment estimators on identical saved snapshots."""

import argparse
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from examples.shock.normal import (  # noqa: E402
    ALSMEYER_L_OVER_LAMBDA_HS,
    BIRD_L_OVER_LAMBDA_VSS,
    max_slope_thickness,
    report,
    mfp_alsmeyer,
    mfp_vss,
    tanh_thickness,
    shock_frame_average,
)

ap = argparse.ArgumentParser()
ap.add_argument("case")
a = ap.parse_args()

d = np.load(a.case + ".npz")
info = pickle.load(open(a.case + ".pkl", "rb"))
rho, temp, ux = d["rho"], d["temp"], d["ux"]
# Runs saved before the eqn-4.52 correction carry the old (1.2455x too small)
# mean free path.  The SIMULATION is unaffected -- lam1 only sized the domain --
# so recompute the normalisation rather than re-running.
info["lam1"] = mfp_vss(info["nrho"])
info["lam1_hs"] = mfp_alsmeyer(info["nrho"])
info["cells_per_mfp"] = info["lam1"] / info["dx"]
info.setdefault("n_per_seed", info["steps"] - info["n_settle"])
info.setdefault("seeds", rho.shape[0] // info["n_per_seed"])
n1 = info["nrho"]
n2 = n1 * info["rho2_over_rho1"]
print(f"### {a.case}   snapshots {rho.shape}   dim {info['dim']}   npc {info['npc']}")
print(
    f"    settle {info['n_settle']} steps, {info['steps']} total, wall {info.get('wall', 0):.0f}s"
)

print(f"\n{'align':>9} {'smooth=5':>9} {'9':>7} {'13':>7} {'17':>7}   u_s fit / exact   resid")
for align in ("crossing", "integral", "fit"):
    xi, r, t, u, used = shock_frame_average(rho, temp, ux, info, align=align)
    row = []
    for k in (5, 9, 13, 17):
        L, _ = max_slope_thickness(xi * info["lam1"], r, n1, n2, smooth_cells=k)
        row.append(L / info["lam1"])
    us = info.get("u_s_fit", float("nan"))
    print(
        f"{align:>9} "
        + " ".join(f"{v:7.2f}" for v in row)
        + f"   {us:7.1f} / {info['u_s']:.1f}   {info.get('align_resid_mfp', float('nan')):.3f} mfp"
    )
print(f"{'BIRD':>9} {BIRD_L_OVER_LAMBDA_VSS:7.2f}  (Alsmeyer, HS: {ALSMEYER_L_OVER_LAMBDA_HS})")
xi, r, t, u, used = shock_frame_average(rho, temp, ux, info, align="fit")
hs = np.array([5, 9, 13, 17]) * info["dx"] / info["lam1"]
Ls = np.array(
    [
        max_slope_thickness(xi * info["lam1"], r, n1, n2, smooth_cells=k)[0] / info["lam1"]
        for k in (5, 9, 13, 17)
    ]
)
A = np.polyfit(hs**2, Ls**2, 1)
Lt, dlt, res = tanh_thickness(xi, r, n1, n2, lam=1.0)
print(
    f"\n  max-slope, extrapolated to zero smoothing: L/lam = {np.sqrt(max(A[1], 0)):.2f}  (broadening coeff {A[0]:.2f})"
)
print(f"  tanh fit (no smoothing at all):            L/lam = {Lt:.2f}  (rms residual {res:.4f})")

print()
xi, r, t, u, used = shock_frame_average(rho, temp, ux, info, align="fit")
report(xi, r, t, u, info, used)
np.savez(a.case + "_aligned.npz", xi=xi, rho=r, temp=t, ux=u)
