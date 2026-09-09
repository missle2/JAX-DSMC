"""Resolution convergence of the measured shock thickness."""
import glob, os, pickle, sys
sys.path.insert(0, ".")
import numpy as np
from examples.shock.normal import (BIRD_L_OVER_LAMBDA_VSS, max_slope_thickness,
                                   mfp_alsmeyer, mfp_vss, shock_frame_average, tanh_thickness)
print(f"{'case':>12} {'cells/mfp':>10} {'npc':>6} {'dt/mct':>8} {'L max-slope':>12} {'L h->0':>8} {'L tanh':>8}")
for case in sorted(glob.glob("scratch/bird/sw_*.npz")):
    base = case[:-4]
    if not os.path.exists(base + ".pkl"):
        continue
    d = np.load(case); info = pickle.load(open(base + ".pkl", "rb"))
    info["lam1"] = mfp_vss(info["nrho"]); info["lam1_hs"] = mfp_alsmeyer(info["nrho"])
    info["cells_per_mfp"] = info["lam1"] / info["dx"]
    info.setdefault("n_per_seed", info["steps"] - info["n_settle"])
    xi, r, t, u, used = shock_frame_average(d["rho"], d["temp"], d["ux"], info, align="fit")
    n1 = info["nrho"]; n2 = n1 * info["rho2_over_rho1"]
    hs = np.array([5, 9, 13, 17]) * info["dx"] / info["lam1"]
    Ls = np.array([max_slope_thickness(xi*info["lam1"], r, n1, n2, smooth_cells=k)[0]/info["lam1"]
                   for k in (5, 9, 13, 17)])
    A = np.polyfit(hs**2, Ls**2, 1)
    Lt, _, _ = tanh_thickness(xi, r, n1, n2, lam=1.0)
    print(f"{os.path.basename(base):>12} {info['cells_per_mfp']:>10.2f} {info['npc']:>6} "
          f"{info['dt_over_mct']:>8.4f} {Ls[0]:>12.2f} {np.sqrt(max(A[1],0)):>8.2f} {Lt:>8.2f}")
print(f"\n  Bird 12.11: {BIRD_L_OVER_LAMBDA_VSS}")
