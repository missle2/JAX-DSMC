"""Figures for the turbulence cases: 2-D viscosity decay, 1-D spectrum."""
import pickle, sys, os
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import numpy as np
from examples.turbulence.validate import energy_history, fit_decay
from jax_dsmc.viz import INK, make_figure, savefig, style_axes

fig, axes = make_figure(1, 3, figsize=(16.0, 4.2))

# --- 2-D: exact NS decay, viscosity measurement ---
d = np.load("scratch/bird/visc_d2.npz")
info = pickle.load(open("scratch/bird/visc_d2.pkl", "rb"))
class M: pass
m = M(); m.u = d["u"]; m.count = d["count"]
e, floor = energy_history(m, info)
t = np.arange(e.size) * info["snap_dt"]
rate, e0, r2 = fit_decay(t, e, floor=floor)
tn = t / info["tau_visc"]
axes[0].semilogy(tn, e / e[0], color="#2166ac", lw=1.6, label="DSMC (noise floor removed)")
axes[0].semilogy(tn, e0 * np.exp(-rate * t) / e[0], color="#b2182b", lw=1.3, ls="--",
                 label=f"fit: $\\nu$ = {rate / (4 * info['kbox']**2):.4f}")
axes[0].semilogy(tn, np.exp(-4 * info["nu"] * info["kbox"]**2 * t), color="0.35", lw=1.2, ls=":",
                 label=f"exact NS: $\\nu$ = {info['nu']:.4f}")
axes[0].semilogy(tn, floor / e[0], color="0.6", lw=1.0, label="subtracted noise floor")
axes[0].set_xlabel(r"$t / \tau_{\rm visc}$")
axes[0].set_ylabel(r"$E(t) / E(0)$")
err = 100 * (rate / (4 * info["kbox"]**2) / info["nu"] - 1)
axes[0].set_title(f"2-D Taylor-Green: EXACT Navier-Stokes decay\n"
                  f"$\\nu$ vs Chapman-Enskog: {err:+.2f} %   ($R^2$ = {r2:.4f})", color=INK)
axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")

# --- 1-D: E(k) ~ k^-2, and what ensemble averaging actually bought ---
from examples.turbulence.validate import (  # noqa: E402
    clean_window, dissipation_wavenumber, fit_slope)

single, ens = "scratch/bird/turb_d1_spec.npz", "scratch/bird/turb_d1e_spec.npz"
if os.path.exists(ens):
    z1, z16 = np.load(single), np.load(ens)
    k = z16["k"]
    axes[1].loglog(k[1:], z1["E"][1:], color="#92c5de", lw=1.1, label="DSMC, 1 seed")
    axes[1].loglog(k[1:], z1["noise"][1:], color="0.75", lw=1.0, ls="--",
                   label="floor, 1 seed")
    axes[1].loglog(k[1:], z16["E"][1:], color="#2166ac", lw=1.4, label="DSMC, 16-seed mean")
    axes[1].loglog(k[1:], z16["noise"][1:], color="0.45", lw=1.0, ls="--",
                   label="floor, 16-seed mean")
    kr = k[(k >= 2) & (k <= 12)]
    axes[1].loglog(kr, z16["E"][2] * (kr / 2.0) ** -2.0, color="#b2182b", ls=":", lw=1.4,
                   label=r"$k^{-2}$ (Burgers), for reference")
    k_d = dissipation_wavenumber(61.1, 1)
    axes[1].axvline(k_d, color="0.3", lw=1.0, ls="-.")
    axes[1].annotate(r"$k_d\sim{\rm Re}/6$", (k_d * 1.15, z16["E"][1] * 0.5), fontsize=7.5,
                     color=INK)
    sel = clean_window(k, z16["E"], z16["noise"])
    axes[1].set_xlabel("wavenumber $k$")
    axes[1].set_ylabel("$E(k)$")
    axes[1].set_title("1-D compressive, Re = 61: no spectral range exists\n"
                      f"only k = {k[sel].min():.0f}-{k[sel].max():.0f} clears the floor above "
                      "the peak", color=INK)
    axes[1].legend(frameon=False, fontsize=6.8, loc="upper right")
else:
    axes[1].text(0.5, 0.5, "1-D ensemble spectrum not finished", ha="center",
                 transform=axes[1].transAxes)

# --- 1-D: does the broadband IC remove the comb? ---
bb = "scratch/bird/turb_d1b_spec.npz"
if os.path.exists(bb):
    z = np.load(bb)
    k, E, noise = z["k"], z["E"], z["noise"]
    axes[2].loglog(k[1:], E[1:], color="#2166ac", lw=1.4, label="DSMC, broadband IC")
    axes[2].loglog(k[1:], noise[1:], color="0.45", lw=1.0, ls="--", label="noise floor")
    sel = clean_window(k, E, noise, k_forcing=4)
    kk = k[sel]
    ttl = "1-D broadband IC, Ma = 1, Re = 204"
    if sel.sum() >= 5:
        sl, err, _ = fit_slope(k, E, sel)
        axes[2].loglog(kk, np.exp(np.polyval(np.polyfit(np.log(kk), np.log(E[sel]), 1),
                                             np.log(kk))),
                       color="#b2182b", ls="--", lw=1.5,
                       label=f"fit {sl:.2f} $\\pm$ {err:.2f}, k = {kk[0]:.0f}-{kk[-1]:.0f}")
        ttl += (f"\nslope {sl:.2f} $\\pm$ {err:.2f}: {abs(sl + 2) / err:.1f}$\\sigma$ from "
                r"$-2$, consistent but not a test")
    else:
        top = int(kk.max()) if sel.any() else 0
        ttl += f"\nstill too narrow (clean above the band only to k = {top})"
    kr = k[(k >= 5) & (k <= 40)]
    axes[2].loglog(kr, E[5] * (kr / 5.0) ** -2.0, color="0.35", ls=":", lw=1.3,
                   label=r"$k^{-2}$ (Burgers)")
    axes[2].axvspan(0.9, 4.4, color="0.9", zorder=0)
    axes[2].annotate("forced band", (1.0, E[1:].max() * 0.3), fontsize=7, color=INK)
    axes[2].axvline(33.9, color="0.3", lw=1.0, ls="-.")
    axes[2].annotate("$k_d$", (36.0, E[1:].max() * 0.3), fontsize=7.5, color=INK)
    axes[2].set_xlabel("wavenumber $k$")
    axes[2].set_ylabel("$E(k)$")
    axes[2].set_title(ttl, color=INK)
    axes[2].legend(frameon=False, fontsize=6.8, loc="upper right")
else:
    axes[2].text(0.5, 0.5, "broadband run not finished", ha="center",
                 transform=axes[2].transAxes)

for ax in axes:
    style_axes(ax)
print(savefig(fig, "examples/turbulence/figures/turbulence_gates.png", dpi=150))
