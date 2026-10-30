"""Which slope fit made the airfoil FD gate's seed 4 an outlier (2026-10-29).

Re-runs one seed of examples/paper/results/fd/af_drag_radius.json (the gate's config) with
``ensemble_device.field_query`` wrapped: every local fit's query position, its number of valid
nodes, the normal matrix's conditioning and the fitted x-slope go to the host through
``jax.debug.callback``.  Production code is unchanged.

    PYTHONPATH=. python3 scratch/fitgate/seed_outlier.py 4 examples/mesh/results/fitgate/af_seed4_diag.npz
"""

import functools
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples import ensemble_mesh as EM
from jax_dsmc import ensemble_backward, ensemble_device

jax.config.update("jax_enable_x64", True)
RECORDS = []
_original = ensemble_device.field_query


def _record(k, n_good, lam_min, lam_max, gx, qx, qv):
    RECORDS.append(
        (
            int(k),
            np.asarray(n_good),
            np.asarray(lam_min),
            np.asarray(lam_max),
            np.asarray(gx),
            np.asarray(qx),
            np.asarray(qv),
        )
    )


def field_query(node_x, node_v, node_valid, node_val, q_x, q_v, scale, k, basis="quadratic_full"):
    out = _original(node_x, node_v, node_valid, node_val, q_x, q_v, scale, k, basis)
    # the same nearest nodes, weights and normal matrix as the fit (ensemble_device.field_query)
    pts = jnp.concatenate((node_x, node_v), axis=-1) / scale
    q = jnp.concatenate((q_x, q_v), axis=-1) / scale
    d2 = jnp.where(node_valid, jnp.sum((pts - q[:, None, :]) ** 2, axis=-1), jnp.inf)
    neg, ids = jax.lax.top_k(-d2, k)
    dist = jnp.sqrt(jnp.maximum(-neg, 0.0))
    good = jnp.isfinite(dist)
    dk = jnp.max(jnp.where(good, dist, 0.0), axis=1, keepdims=True)
    w = jnp.where(good, 1.0 / (1.0 + (dist / jnp.maximum(dk, 1e-12)) ** 2), 0.0)
    chosen = jnp.take_along_axis(pts, ids[..., None], axis=1)
    design = jnp.where(good[..., None], ensemble_device._design(chosen - q[:, None, :], basis), 0.0)
    m = jnp.einsum("qki,qk,qkj->qij", design, w, design)
    m = m / jnp.maximum(jnp.trace(m, axis1=1, axis2=2) / m.shape[-1], 1e-300)[:, None, None]
    eig = jnp.linalg.eigvalsh(m)
    gx = out[1] if out[1].ndim == 2 else out[1][..., 0]
    jax.debug.callback(
        functools.partial(_record, k),
        jnp.sum(good, axis=1),
        eig[:, 0],
        eig[:, -1],
        gx,
        q_x,
        q_v,
    )
    return out


def main():
    seed, out = int(sys.argv[1]), sys.argv[2]
    with open("examples/paper/results/fd/af_drag_radius.json") as fh:
        art = json.load(fh)
    cfg = dict(
        EM.default_config(), **{k: v for k, v in art["config"].items() if k in EM.default_config()}
    )
    ensemble_backward.configure("virtual", 4, 8, "staged", ensemble_backward.CELL_BASIS)
    ensemble_device.field_query = field_query
    t0 = time.time()
    row = EM.ensemble_gradient(
        cfg,
        "radius",
        jax.random.key(art["config"]["seed_start"] + seed),
        32,
        art["config"]["timing_band"],
        "along_pool_plane",
    )
    ref = art["rows"][seed]
    print(
        f"seed {seed}: J {row['objective']:.7g} (artifact {ref['objective']:.7g}); gradient "
        f"{row['gradient']:.4g} (artifact {ref['gradient']:.4g}); {time.time() - t0:.0f}s"
    )
    for name, t in row["variants"].items():
        print(f"  {name}: {t['gradient']:.4g} (artifact {ref['variants'][name]['gradient']:.4g})")
    ks = np.array([r[0] for r in RECORDS])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        k=np.concatenate([np.full(len(r[1]), r[0]) for r in RECORDS]),
        call=np.concatenate([np.full(len(r[1]), i) for i, r in enumerate(RECORDS)]),
        n_good=np.concatenate([r[1] for r in RECORDS]),
        lam_min=np.concatenate([r[2] for r in RECORDS]),
        lam_max=np.concatenate([r[3] for r in RECORDS]),
        gx=np.concatenate([r[4] for r in RECORDS]),
        qx=np.concatenate([r[5] for r in RECORDS]),
        qv=np.concatenate([r[6] for r in RECORDS]),
    )
    print("calls", len(RECORDS), "by k", {int(k): int(np.sum(ks == k)) for k in np.unique(ks)})
    summarise(out)


def summarise(path):
    """Per fit width k: the largest x-slopes (robust z over that k's real queries) with their
    valid-node counts and normal-matrix conditioning, against the typical query's."""
    d = np.load(path)
    real = d["n_good"] > 0
    for k in np.unique(d["k"]):
        m = real & (d["k"] == k)
        g = np.linalg.norm(d["gx"][m], axis=1)
        mad = 1.4826 * np.median(np.abs(g - np.median(g)))
        z = (g - np.median(g)) / max(mad, 1e-300)
        cond = d["lam_max"][m] / np.maximum(d["lam_min"][m], 1e-300)
        print(
            f"k={k}: {m.sum()} queries; |gx| median {np.median(g):.3g}; n_good median "
            f"{np.median(d['n_good'][m]):.0f}; cond median {np.median(cond):.3g}"
        )
        for i in np.argsort(-z)[:8]:
            print(
                f"   z {z[i]:9.1f} |gx| {g[i]:.3g} n_good {d['n_good'][m][i]} cond {cond[i]:.3g} "
                f"x {np.round(d['qx'][m][i], 3)} v {np.round(d['qv'][m][i], 0)}"
            )


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summarise(sys.argv[2])
    else:
        main()
