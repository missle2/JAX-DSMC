"""Do particles tunnel through the thin bodies? (2026-10-27)  ``hit_body`` brackets the entry
root between samples ``ds = min(reach, |v| dt) / n_sample <= h / 2`` apart, so a feature
thinner than ``ds`` along a ray can lie wholly between two samples and be missed.  Per run,
the body at ``--iterate`` (default the last): random rays from points outside the body within
``reach`` of it, the full search length ``reach`` (the coarsest spacing ``h / 2``), against a
reference search at ``h / 128``; reports the share of the reference's hits that ``hit_body``
misses or finds later (a farther crossing), overall and for the initial body (the control).

    PYTHONPATH=. python3 scratch/lsfilter/tunnel.py <tag> [<tag> ...] [--rays 20000]
"""

import argparse
import dataclasses
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

FINE = 128  # reference samples per knot
CHUNK = 2000


def _first_entry(ls, x, u, length, n):
    """The first ``+ -> -`` crossing of ``phi`` along ``x + s u``, ``s in [0, length]`` at
    ``n`` samples (``inf`` if none)."""
    s = jnp.linspace(0.0, 1.0, n + 1)[None, :] * length
    pts = x[:, None, :] + s[..., None] * u[:, None, :]
    phi = O.L.value(ls, pts.reshape(-1, x.shape[1])).reshape(x.shape[0], n + 1)
    cross = (phi[:, :-1] > 0) & (phi[:, 1:] <= 0)
    first = jnp.argmax(cross, axis=1)
    return jnp.where(cross.any(axis=1), s[0, first + 1], jnp.inf)


def rays(ls, lo, hi, n, rng):
    """``n`` starts outside the body within ``reach`` of its bounding box, unit directions."""
    d = ls.coef.ndim
    xs = []
    while sum(len(x) for x in xs) < n:
        x = rng.uniform(lo - ls.reach, hi + ls.reach, size=(4 * n, d))
        phi = np.asarray(O.L.value(ls, jnp.asarray(x)))
        xs.append(x[(phi > 0) & (phi < ls.reach)])
    x = np.concatenate(xs)[:n]
    u = rng.normal(size=(n, d))
    return x, u / np.linalg.norm(u, axis=1, keepdims=True)


def measure(ls, n, seed=0):
    inside = np.argwhere(np.asarray(ls.coef) < 0)
    lo = np.asarray(ls.lo) + ls.h * (inside.min(0) - 2)
    hi = np.asarray(ls.lo) + ls.h * (inside.max(0) + 2)
    x, u = rays(ls, lo, hi, n, np.random.default_rng(seed))
    hit_ref, hit = [], []
    speed = 1.0
    n_fine = int(FINE * ls.reach / ls.h)
    for i in range(0, n, CHUNK):
        xc, uc = jnp.asarray(x[i : i + CHUNK]), jnp.asarray(u[i : i + CHUNK])
        hit_ref.append(np.asarray(_first_entry(ls, xc, uc, ls.reach, n_fine)))
        hit.append(np.asarray(O.L.hit_body(xc, uc * speed, ls)) * speed)
    hit_ref, hit = np.concatenate(hit_ref), np.concatenate(hit)
    true = np.isfinite(hit_ref)
    missed = true & ~np.isfinite(hit)
    later = true & np.isfinite(hit) & (hit > hit_ref + ls.h / FINE * 2)
    return {"rays": n, "hits": int(true.sum()), "missed": float(missed.sum() / max(true.sum(), 1)),
            "later": float(later.sum() / max(true.sum(), 1))}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--rays", type=int, default=20000)
    ap.add_argument("--iterate", type=int, default=None)
    args = ap.parse_args()
    jax.config.update("jax_enable_x64", True)
    for tag in args.tags:
        out = f"{O.RESULTS}/{tag}"
        hist = json.load(open(f"{out}/history.json"))
        cfg = dict(O.default_config(), **hist["config"])
        _grid, ls0 = O.initial_levelset(cfg)
        k = len(hist["rows"]) if args.iterate is None else args.iterate
        while not np.isfinite(np.load(f"{out}/coef_{k}.npy")).all():
            k -= 1
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{out}/coef_{k}.npy")))
        r0 = measure(ls0, args.rays)
        r = measure(ls, args.rays)
        print(f"{tag}: iterate {k}: missed {100 * r['missed']:.2f} %, later "
              f"{100 * r['later']:.2f} % of {r['hits']} hits  (initial body: missed "
              f"{100 * r0['missed']:.2f} %, later {100 * r0['later']:.2f} %)", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
