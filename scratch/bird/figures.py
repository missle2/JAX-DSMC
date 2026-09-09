"""Figures for the Bird normal-shock validation."""
import pickle, sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from examples.shock.normal import (
    ALSMEYER_L_OVER_LAMBDA_HS, BIRD_L_OVER_LAMBDA_VSS, max_slope_thickness,
    mfp_alsmeyer, mfp_vss, normalised, shock_frame_average, tanh_thickness)
from jax_dsmc.viz import INK, make_figure, savefig, style_axes, time_color

cases = [("scratch/bird/m2_d%d" % d, d) for d in (1, 2, 3)]
fig, axes = make_figure(1, 2, figsize=(11.0, 4.2))

for ci, (case, dim) in enumerate(cases):
    d = np.load(case + ".npz"); info = pickle.load(open(case + ".pkl", "rb"))
    info["lam1"] = mfp_vss(info["nrho"]); info["lam1_hs"] = mfp_alsmeyer(info["nrho"])
    info.setdefault("n_per_seed", info["steps"] - info["n_settle"])
    xi, r, t, u, used = shock_frame_average(d["rho"], d["temp"], d["ux"], info, align="fit")
    n1 = info["nrho"]; n2 = n1 * info["rho2_over_rho1"]
    t1 = info["temp"]; t2 = t1 * info["t2_over_t1"]
    c = time_color(ci, 3)
    m = np.abs(xi) < 12
    axes[0].plot(xi[m], normalised(r, n1, n2)[m], color=c, lw=1.6, label=f"D = {dim}")
    axes[0].plot(xi[m], normalised(t, t1, t2)[m], color=c, lw=1.2, ls="--")
    L, _ = max_slope_thickness(xi * info["lam1"], r, n1, n2, smooth_cells=5)
    Lt, _, _ = tanh_thickness(xi, r, n1, n2, lam=1.0)
    axes[1].plot([dim], [L / info["lam1"]], "o", color=c, ms=9, label=f"D={dim} max-slope")
    axes[1].plot([dim], [Lt], "s", color=c, ms=7, mfc="none")

axes[0].set_xlabel(r"$\xi = (x - x_{\rm shock}) / \lambda_1$")
axes[0].set_ylabel(r"normalised $\rho$ (solid), $T$ (dashed)")
axes[0].set_title("Bird Fig. 12.30 form: argon, Mach 2\n(temperature leads density)", color=INK)
axes[0].legend(frameon=False, fontsize=8)
axes[0].invert_xaxis()

axes[1].axhline(BIRD_L_OVER_LAMBDA_VSS, color="0.35", ls="-", lw=1.4)
axes[1].text(1.55, BIRD_L_OVER_LAMBDA_VSS + 0.06, "Bird 12.11 DSMC: 6.2", fontsize=8, color="0.3")
axes[1].axhspan(BIRD_L_OVER_LAMBDA_VSS * 0.95, BIRD_L_OVER_LAMBDA_VSS * 1.05,
                color="0.5", alpha=0.12)
axes[1].set_xlim(0.5, 3.5); axes[1].set_xticks([1, 2, 3])
axes[1].set_ylim(5.4, 6.9)
axes[1].set_xlabel("spatial dimension D")
axes[1].set_ylabel(r"$L / \lambda_1$   (Bird eqn 4.52)")
axes[1].set_title("maximum-slope thickness\nfilled = max-slope, open = tanh fit", color=INK)
for ax in axes:
    style_axes(ax)
print(savefig(fig, "examples/shock/figures/bird_normal_shock.png", dpi=150))
