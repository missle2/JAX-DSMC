"""Unsupervised structure of the correction inputs (2026-11-15), on the per-cell features of
scratch/lc_cellfeat.py (coarse cells, as the operator sees them):

1. PCA of the training cells' standardised 9-input features (the models' own coordinates,
   phi / phi_scale): how many independent directions the data has, and the MLP small-signal
   slopes and the lasso coefficients projected on them (the identifiable interpretation).
2. A Gaussian mixture on the training cells (all 11 inputs, asinh of the standardised
   features, whitened): clusters = flow regimes, the regime mix per case.
3. An out-of-training-range gate: a cell's distance to its 5th nearest training cell (same
   space) beyond the 99th percentile of held-out training cells' distances. (The mixture's own
   likelihood is too tight for this: each training case is a thin island in feature space --
   dx / lambda is nearly fixed per case -- so it flags 56-99 % of every held-out shock, also
   where every model interpolates well.) Per case the flagged fraction, with a 32-snapshot
   block bootstrap interval, against each model's error / base.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. python scratch/lc_unsup.py

Figures: examples/learned_collision/figures/round4_{pca,regimes,gate}.png; tables to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lc_round2 as r2
import lc_round3 as r3

from examples.learned_collision import sweep_report as sr
from examples.learned_collision.plot_round1 import INK, MUTED, _style
from jax_dsmc.collide import relax

R, FIG, TRAIN, ALL = r2.R, r2.FIG, r2.TRAIN, r2.ALL
NAMES = relax.FEATURES
N_IN = 9
N_PC = 4
SNAPS = 32
TAIL = 0.01
REGIMES = 6
K_NN = 5
CLUSTER_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
                  "#e34948")  # fmt: skip
TEST = tuple(c for c in ALL if c not in TRAIN)


def load_cells():
    """``{case: f64[cells, 11]}`` standardised by the training RMS (the models' input)."""
    scale = np.array(r2.load("features_v4.json")["phi_scale"])
    raw = np.load(R / "cellfeat_v1.npz")
    return {c: raw[c] / scale for c in ALL}


def training_pool(z, rng, take=None):
    """Equal cells per training case (the smallest case's count), and which case each came
    from."""
    n = take or min(len(z[c]) for c in TRAIN)
    rows = [z[c][rng.choice(len(z[c]), size=n, replace=False)] for c in TRAIN]
    return np.concatenate(rows), np.repeat(np.arange(len(TRAIN)), n)


# --------------------------------------------------------------------------------- 1. PCA


def pca(x):
    """Centred PCA: ``(variance share, loadings[F, F] by column, mean)``."""
    mu = x.mean(axis=0)
    w, v = np.linalg.eigh(np.cov(x - mu, rowvar=False))
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    v *= np.sign(v[np.argmax(np.abs(v), axis=0), np.arange(v.shape[1])])  # largest loading > 0
    return w / w.sum(), v, mu


def mlp_slopes():
    """``{v2|v3: f64[runs, 2, 9]}``: every learning all-case MLP's slopes (as round3)."""
    out = {}
    for v in ("v2", "v3"):
        slopes = []
        for r in r2.load(f"sweep_plan_{v}.json")["runs"]:
            if r["validate"] or (r["weight_decay"] == 0.01 and r["lr"] < 0.01):
                continue
            d = r2.load(f"train_mlp_{r['tag']}.json")
            slopes.append(r3.small_signal(d["theta"], "mlp")[:, :N_IN])
        out[v] = np.array(slopes)
    return out


def pca_report(z, rng):
    x, _ = training_pool({c: z[c][:, :N_IN] for c in ALL}, rng)
    share, v, _ = pca(x)
    share11, _, _ = pca(training_pool(z, rng)[0])
    print("\n== PCA, training cells, standardised 9 inputs (centred) ==")
    print("  variance share " + " ".join(f"{s:.3f}" for s in share))
    print("  cumulative     " + " ".join(f"{s:.3f}" for s in np.cumsum(share)))
    print(f"  condition number of the covariance {share[0] / share[-1]:.0f}")
    print("  11 inputs, cumulative " + " ".join(f"{s:.3f}" for s in np.cumsum(share11)))
    for k in range(N_PC + 2):
        top = np.argsort(-np.abs(v[:, k]))[:5]
        print(f"  PC{k + 1}: " + " ".join(f"{NAMES[i]} {v[i, k]:+.2f}" for i in top))
    slopes = mlp_slopes()
    coef = np.array(r2.load("lasso_in9_seed.json")["outer"][-1]["coef"]).reshape(2, N_IN)
    proj = {g: s @ v for g, s in slopes.items()}  # [runs, 2, 9]: slope along each PC
    lasso = coef @ v
    print("\n== slopes along the PCs (d correction / d PC score, unit = 1 sd of the PC "
          "direction in standardised units) ==")  # fmt: skip
    for row, lab in ((0, "dlog r_s"), (1, "dlog r_q")):
        for g, p in proj.items():
            med = np.median(p[:, row], axis=0)
            agree = (np.sign(p[:, row]) == np.sign(med)).mean(axis=0)
            print(f"  {lab} {g} ({len(p)} runs): " + " ".join(
                f"PC{k + 1} {med[k]:+.4f}({agree[k]:.0%})" for k in range(N_PC + 2)))  # fmt: skip
        print(f"  {lab} lasso:        " + " ".join(f"PC{k + 1} {lasso[row, k]:+.4f}"
                                                     for k in range(N_PC + 2)))  # fmt: skip
    # identifiability: fraction of each run's slope norm outside the leading PCs
    for g, s in slopes.items():
        p = proj[g]
        out = np.linalg.norm(p[:, :, N_PC:], axis=2) / np.linalg.norm(p, axis=2)
        print(f"  {g}: slope-norm fraction outside PC1-{N_PC}: median {np.median(out):.2f}")
    return share, share11, v, proj, lasso


def fig_pca(share, share11, v, proj, lasso, path):
    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_axes([0.04, 0.14, 0.2, 0.74])
    k = np.arange(1, len(share) + 1)
    ax.bar(k, share, 0.7, color="#2a78d6", label="9 inputs: share")
    ax.plot(k, np.cumsum(share), "o-", color=INK, ms=4, lw=1.2, label="9 inputs: cumulative")
    ax.plot(np.arange(1, len(share11) + 1), np.cumsum(share11), "s--", color=MUTED, ms=4,
            lw=1.0, label="11 inputs: cumulative")  # fmt: skip
    ax.axhline(0.9, color=MUTED, lw=0.7, ls=":")
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance share (training cells)")
    ax.set_title(f"spectrum (cond. {share[0] / share[-1]:.0f})", color=INK, fontsize=9,
                 loc="left")  # fmt: skip
    ax.legend(fontsize=7, frameon=False, loc="center right")
    _style(ax)
    ax = fig.add_axes([0.31, 0.14, 0.17, 0.74])
    im = ax.imshow(v[:, :N_PC], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(N_IN):
        for j in range(N_PC):
            ax.text(j, i, f"{v[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v[i, j]) > 0.6 else INK)  # fmt: skip
    ax.set_xticks(range(N_PC))
    ax.set_xticklabels([f"PC{j + 1}\n{share[j]:.0%}" for j in range(N_PC)], fontsize=8)
    ax.set_yticks(range(N_IN))
    ax.set_yticklabels(NAMES[:N_IN], fontsize=8)
    ax.set_title("loadings", color=INK, fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.7)
    for j, (row, lab) in enumerate(((0, r"$d\log r_s$ (stress rate)"),
                                    (1, r"$d\log r_q$ (heat-flux rate)"))):  # fmt: skip
        ax = fig.add_axes([0.55 + 0.23 * j, 0.14, 0.2, 0.74])
        npc = N_PC + 2
        ax.bar(np.arange(npc), lasso[row, :npc], 0.7, color="#b5487a", alpha=0.35,
               label="lasso 9-in (seed CV)", zorder=0)  # fmt: skip
        for g, (dx, col) in {"v2": (-0.12, r3.V_COLOR["v2"]),
                             "v3": (0.12, r3.V_COLOR["v3"])}.items():  # fmt: skip
            p = proj[g][:, row, :npc]
            for kk in range(npc):
                ax.plot(np.full(len(p), kk + dx), p[:, kk], "o", ms=4, color=col, alpha=0.6,
                        mew=0, label=f"{g} MLPs ({len(p)})" if kk == 0 else None)  # fmt: skip
        ax.axhline(0.0, color=INK, lw=0.8)
        ax.axvspan(N_PC - 0.5, npc - 0.5, color="#e4e3df", alpha=0.6, zorder=-1)
        ax.set_xticks(range(npc))
        ax.set_xticklabels([f"PC{kk + 1}" for kk in range(npc)], fontsize=8)
        ax.set_title(lab + " along the PCs", color=INK, fontsize=9, loc="left")
        ax.set_ylim(-0.08, 0.08)
        for kk in range(npc):
            if abs(lasso[row, kk]) > 0.08:
                ax.text(kk, 0.074 * np.sign(lasso[row, kk]), f"{lasso[row, kk]:+.2f}",
                        ha="center", va="center", fontsize=7, color=INK)  # fmt: skip
        if j == 0:
            ax.set_ylabel("slope at an equilibrium cell")
            ax.legend(fontsize=7, frameon=False, loc="lower left")
        _style(ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- 2. mixture and gate


def gmm_space(z):
    return np.arcsinh(z)


def fit_gmm(z, rng):
    """BIC over K on one half of the training pool (reported; the regimes use REGIMES
    components for readability -- BIC keeps falling to K = 10, a continuum, not discrete
    clusters); the whitening; the k-NN gate and its threshold from the other half."""
    x, src = training_pool({c: gmm_space(z[c]) for c in ALL}, rng)
    half = rng.permutation(len(x))
    fit, hold = half[: len(x) // 2], half[len(x) // 2:]
    mu, sd = x[fit].mean(axis=0), x[fit].std(axis=0)
    xw = (x - mu) / sd
    ks = np.arange(1, 11)
    models = [GaussianMixture(k, covariance_type="full", n_init=3, reg_covar=1e-4,
                              random_state=0).fit(xw[fit]) for k in ks]  # fmt: skip
    bic = np.array([m.bic(xw[fit]) for m in models])
    best = models[REGIMES - 1]
    nn = NearestNeighbors(n_neighbors=K_NN).fit(xw[fit])
    thresh = np.quantile(nn.kneighbors(xw[hold])[0][:, -1], 1.0 - TAIL)
    print(f"\n== Gaussian mixture (11 inputs, asinh, whitened; {len(fit)} training cells) ==")
    print("  BIC " + " ".join(f"K{k}:{b / 1e3:.1f}k" for k, b in zip(ks, bic, strict=True)))
    print(f"  regimes: K = {REGIMES}; gate: {K_NN}-NN distance > {thresh:.3f} (held-out "
          f"training {1 - TAIL:.0%} point)")  # fmt: skip
    return best, nn, ks, bic, (mu, sd), thresh


def score(model, nn, white, z):
    """``(out of range?, regime)`` per cell."""
    xw = (gmm_space(z) - white[0]) / white[1]
    return nn.kneighbors(xw)[0][:, -1], model.predict(xw)


def block_ci(flag, rng, n_boot=2000):
    """Mean of ``flag`` and a 90 % interval from a bootstrap over the SNAPS snapshots (cells
    within one snapshot are correlated)."""
    blocks = np.array_split(flag.astype(float), SNAPS)
    sums = np.array([b.sum() for b in blocks])
    counts = np.array([len(b) for b in blocks])
    pick = rng.integers(SNAPS, size=(n_boot, SNAPS))
    boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(flag.mean()), np.quantile(boot, [0.05, 0.95])


def regime_names(model, white):
    """Each component's centre in raw standardised features (median-like: sinh of the mean)."""
    centre = np.sinh(model.means_ * white[1] + white[0])
    out = []
    for c in centre:
        top = np.argsort(-np.abs(c))[:3]
        out.append(", ".join(f"{NAMES[i]} {c[i]:.2g}" for i in top))
    return centre, out


def gate_report(z, model, nn, white, thresh, rng):
    names_k = regime_names(model, white)[1]
    print("\n== regimes (mixture components): centre, top-3 standardised features ==")
    for k, n in enumerate(names_k):
        print(f"  R{k + 1}: {n}")
    comp, ood = {}, {}
    print("\n== per case: cells, regime shares, fraction beyond the k-NN gate ==")
    for c in ALL:
        d, lab = score(model, nn, white, z[c])
        comp[c] = np.bincount(lab, minlength=model.n_components) / len(lab)
        ood[c] = block_ci(d > thresh, rng)
        role = "train" if c in TRAIN else "test "
        print(f"  {c} {role} {len(lab):6d}  " + " ".join(f"{s:4.0%}" for s in comp[c]) +
              f"   out-of-range {ood[c][0]:6.1%} [{ood[c][1][0]:.1%}, {ood[c][1][1]:.1%}]")
    # which regimes the out-of-range cells sit in, per test case
    print("\n== out-of-range cells by nearest regime (test shocks) ==")
    for c in ("K3", "K5", "K6", "K7"):
        d, lab = score(model, nn, white, z[c])
        m = d > thresh
        if m.any():
            share = np.bincount(lab[m], minlength=model.n_components) / m.sum()
            print(f"  {c}: " + " ".join(f"R{k + 1} {s:.0%}" for k, s in enumerate(share)))
    # the features carrying the out-of-range cells: median |z| ratio vs training q99
    pool = np.concatenate([z[c] for c in TRAIN])
    q99 = np.quantile(np.abs(pool), 0.99, axis=0)
    print("\n== out-of-range cells: share with each feature beyond the training q99 ==")
    for c in ("K5", "K7"):
        d, _ = score(model, nn, white, z[c])
        bad = z[c][d > thresh]
        if len(bad):
            frac = (np.abs(bad) > q99).mean(axis=0)
            print(f"  {c} ({len(bad)} cells): " + " ".join(
                f"{NAMES[i]} {frac[i]:.0%}" for i in np.argsort(-frac)[:5]))  # fmt: skip
    return comp, ood, names_k


def error_ratios():
    base = sr.base_rows(R)
    out = {}
    for lab, f, col, lin in r3.EVAL_MODELS:
        rows = r2.eval_rows(f)
        out[lab] = ({c: r2.error_se(rows[c])[0] / r2.error_se(base[c])[0] for c in ALL},
                    col, lin)  # fmt: skip
    return out


def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def gate_vs_error(ood, errs):
    shocks = [c for c in TEST if c[0] == "K"]
    x = np.array([ood[c][0] for c in shocks])
    print("\n== out-of-range fraction vs error / base, held-out shocks "
          f"({', '.join(shocks)}; Spearman, exploratory, n = {len(shocks)}) ==")  # fmt: skip
    for lab, (e, _, lin) in errs.items():
        y = np.array([e[c] for c in shocks])
        print(f"  {lab:24s} {'linear' if lin else 'MLP   '} rho {rank_corr(x, y):+.2f}   " +
              " ".join(f"{c} {e[c]:.2f}" for c in shocks))
    # the gate as a decision (threshold chosen after looking: exploratory)
    flagged = [c for c in TEST if ood[c][0] > 10 * TAIL]
    print(f"  cases flagged (> {10 * TAIL:.0%} of cells out of range): {flagged}")
    for lab, (e, _, lin) in errs.items():
        f = [e[c] for c in flagged]
        u = [e[c] for c in TEST if c not in flagged]
        print(f"  {lab:24s} flagged geo {r2.geo(f):.2f}   unflagged geo {r2.geo(u):.2f}")
    return flagged


def fig_regimes(comp, ks, bic, model, names_k, path):
    fig = plt.figure(figsize=(14, 4.4))
    ax = fig.add_axes([0.05, 0.14, 0.18, 0.74])
    ax.plot(ks, (bic - bic.min()) / 1e3, "o-", color="#2a78d6", ms=5, lw=1.4)
    ax.axvline(model.n_components, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("mixture components K")
    ax.set_ylabel("BIC - min (thousands)")
    ax.set_title("model order (training cells)", color=INK, fontsize=9, loc="left")
    _style(ax)
    ax = fig.add_axes([0.29, 0.14, 0.42, 0.74])
    bottom = np.zeros(len(ALL))
    for k in range(model.n_components):
        h = np.array([comp[c][k] for c in ALL])
        ax.bar(np.arange(len(ALL)), h, 0.75, bottom=bottom, color=CLUSTER_COLORS[k],
               edgecolor="white", linewidth=1.0, label=f"R{k + 1}")  # fmt: skip
        bottom += h
    ax.set_xticks(range(len(ALL)))
    ax.set_xticklabels([f"{c}{'*' if c in TRAIN else ''}" for c in ALL], fontsize=8)
    ax.set_ylabel("share of cells")
    ax.set_title("regime mix per case (* training)", color=INK, fontsize=9, loc="left")
    _style(ax)
    ax = fig.add_axes([0.76, 0.14, 0.22, 0.74])
    ax.axis("off")
    ax.set_title("regime centres (top-3 standardised features)", color=INK, fontsize=9,
                 loc="left")  # fmt: skip
    for k, n in enumerate(names_k):
        ax.text(0.0, 0.95 - 0.1 * k, f"R{k + 1}: " + n.replace(", ", "\n     "), fontsize=6.5,
                color=INK, va="top", transform=ax.transAxes)  # fmt: skip
        ax.plot([-0.06], [0.93 - 0.1 * k], "s", color=CLUSTER_COLORS[k], ms=6,
                transform=ax.transAxes, clip_on=False)  # fmt: skip
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_gate(ood, errs, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    xs = np.arange(len(ALL))
    m = np.array([ood[c][0] for c in ALL])
    lo = np.array([ood[c][1][0] for c in ALL])
    hi = np.array([ood[c][1][1] for c in ALL])
    col = ["#52514e" if c in TRAIN else "#2a78d6" for c in ALL]
    ax.bar(xs, np.maximum(m, 1e-4), 0.7, color=col)
    ax.errorbar(xs, np.maximum(m, 1e-4), yerr=[m - lo, hi - m], fmt="none", ecolor=INK,
                lw=0.9, capsize=2)  # fmt: skip
    ax.axhline(TAIL, color=MUTED, lw=0.8, ls=":")
    ax.text(-0.4, TAIL * 1.15, "training rate (1 %)", fontsize=7, color=MUTED,
            ha="left")  # fmt: skip
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}{'*' if c in TRAIN else ''}" for c in ALL], fontsize=8)
    ax.set_ylabel(f"cells beyond the {K_NN}-NN gate")
    ax.set_title("out-of-range fraction (90 % snapshot-bootstrap; grey = training)",
                 color=INK, fontsize=9, loc="left")  # fmt: skip
    _style(ax)
    ax = axes[1]
    shocks = [c for c in ALL if c[0] == "K"]
    for lab, (e, colr, lin) in errs.items():
        x = np.array([max(ood[c][0], 1e-4) for c in shocks])
        y = np.array([e[c] for c in shocks])
        order = np.argsort(x)
        ax.plot(x[order], y[order], "s--" if lin else "o-", color=colr, ms=5, lw=1.1,
                label=lab, mec="white", mew=0.5)  # fmt: skip
    for c in shocks:
        ax.text(max(ood[c][0], 1e-4), 1.01, c + ("*" if c in TRAIN else ""), fontsize=7,
                color=MUTED, ha="center", transform=ax.get_xaxis_transform())  # fmt: skip
    ax.axhline(1.0, color=INK, lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("out-of-range fraction of the case's cells")
    ax.set_ylabel("error / base")
    ax.set_title("shocks: error vs out-of-range (squares/dashed = linear; * training)",
                 color=INK, fontsize=9, loc="left", pad=14)  # fmt: skip
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    rng = np.random.default_rng(0)
    z = load_cells()
    share, share11, v, proj, lasso = pca_report(z, rng)
    fig_pca(share, share11, v, proj, lasso, FIG / "round4_pca.png")
    model, nn, ks, bic, white, thresh = fit_gmm(z, rng)
    comp, ood, names_k = gate_report(z, model, nn, white, thresh, rng)
    fig_regimes(comp, ks, bic, model, names_k, FIG / "round4_regimes.png")
    errs = error_ratios()
    gate_vs_error(ood, errs)
    fig_gate(ood, errs, FIG / "round4_gate.png")
    print("\nfigures:", ", ".join(sorted(p.name for p in FIG.glob("round4_*.png"))))


if __name__ == "__main__":
    main()
