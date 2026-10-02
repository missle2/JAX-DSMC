"""New vs old level-set hit search: hit/no-hit, hit times, gradients (coef, x, v), and
refinement iteration counts, over random / grazing / on-surface / deep / far rays."""

import dataclasses
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "scratch/ls")
import levelset_old as OLD  # noqa: E402

from jax_dsmc.mesh import levelset as L  # noqa: E402

jax.config.update("jax_enable_x64", True)


def shape(dim, bump, n_sample=16):
    if dim == 2:
        lo, m = L.grid_for_box(np.zeros(2), np.full(2, 10.0), 0.0625)
        ls = L.from_analytic(np.array([5.0, 5.0]), 3.0, lo, 0.0625, m, n_sample=n_sample)
        c, r = np.array([5.0, 5.0]), 3.0
    else:
        lo, m = L.grid_for_box(np.full(3, -2.0), np.full(3, 2.0), 0.1)
        ls = L.from_analytic(np.zeros(3), 1.0, lo, 0.1, m, n_sample=n_sample)
        c, r = np.zeros(3), 1.0
    if bump:
        k = L.knots(ls)
        b = bump * jnp.sin(7.3 * k[:, 0]) * jnp.cos(5.1 * k[:, 1])
        ls = dataclasses.replace(ls, coef=ls.coef + b.reshape(ls.coef.shape))
    return ls, c, r


def rays(key, n, dim, c, r, kind):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    u = jax.random.normal(k1, (n, dim))
    u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
    v = jax.random.normal(k2, (n, 3)) * 400.0
    if dim == 2:
        v = v.at[:, 2].set(0.0)
    if kind == "shell":  # within ~reach of the surface
        rad = r + jax.random.uniform(k3, (n,), minval=-0.05, maxval=0.6)
        x = c + rad[:, None] * u
    elif kind == "grazing":  # tangent-ish rays aimed to pass at tiny distance
        rad = r + jax.random.uniform(k3, (n,), minval=0.01, maxval=0.3)
        x = c + rad[:, None] * u
        # direction: perpendicular to u, tilted inward by a tiny angle
        t = jnp.zeros((n, dim)).at[:, 0].set(-u[:, 1]).at[:, 1].set(u[:, 0])
        ang = jax.random.uniform(k4, (n,), minval=-1e-2, maxval=1e-2)
        dirn = t * jnp.cos(ang)[:, None] - u * jnp.sin(ang)[:, None]
        # start back along the tangent so the closest approach is ahead
        back = jnp.sqrt(jnp.maximum(rad**2 - r**2, 0.0))
        x = x - 0.0 * back[:, None] * dirn
        v = jnp.zeros((n, 3)).at[:, :dim].set(dirn * 400.0)
    elif kind == "surface":  # on the surface, both directions
        x = L.project(SHAPE[0], c + r * u)
    elif kind == "deep":
        x = c + 0.5 * r * u
    else:  # far / anywhere
        x = c + jax.random.uniform(k3, (n,), minval=0.0, maxval=2.0 * r)[:, None] * u
    rem = jax.random.uniform(k4, (n,), minval=1e-5, maxval=3e-4)
    return jnp.asarray(x), v, rem


def tsum(coef, ls, x, v, rem, mod):
    t = mod.hit_body(x, v, dataclasses.replace(ls, coef=coef), rem)
    return jnp.sum(jnp.where(jnp.isfinite(t), t, 0.0))


def txv(x, v, ls, rem, mod):
    t = mod.hit_body(x, v, ls, rem)
    return jnp.sum(jnp.where(jnp.isfinite(t), t, 0.0))


SHAPE = [None]
for dim in (2, 3):
    for bump in (0.0, 0.03):
        ls, c, r = shape(dim, bump)
        SHAPE[0] = ls
        for kind in ("shell", "grazing", "surface", "deep", "far"):
            n = 20000 if dim == 2 else 8000
            x, v, rem = rays(jax.random.PRNGKey(hash((dim, kind)) % 1000), n, dim, c, r, kind)
            for remaining in (rem, None):
                a = np.asarray(jax.jit(L.hit_body)(x, v, ls, remaining))
                b = np.asarray(jax.jit(OLD.hit_body)(x, v, ls, remaining))
                p0 = np.abs(np.asarray(jax.jit(L.value)(ls, x)))
                sens = p0 <= 1e-13
                fa, fb = np.isfinite(a), np.isfinite(b)
                dis = int((fa != fb).sum())
                both = fa & fb
                rel = np.abs(a[both] - b[both]) / np.maximum(np.abs(b[both]), 1e-300)
                absd = np.abs(a[both] - b[both])
                tag = f"D{dim} bump{bump} {kind:8s} {'rem ' if remaining is not None else 'reach'}"
                print(
                    f"{tag}: hits {int(fb.sum()):6d} disagree {dis} "
                    f"max|dt| {absd.max() if both.any() else 0:.2e} "
                    f"max rel {rel.max() if both.any() else 0:.2e} "
                    f"p99.9 rel {np.quantile(rel, 0.999) if both.any() else 0:.2e}",
                    flush=True,
                )
                big = both & ~sens
                relb = np.abs(a[big] - b[big]) / np.maximum(np.abs(b[big]), 1e-300)
                print(
                    f"   excl |phi0|<=1e-13 ({int(sens.sum())} rays): disagree "
                    f"{int(((fa != fb) & ~sens).sum())} max rel {relb.max() if big.any() else 0:.2e}"
                    f" max|dt| {np.abs(a[big] - b[big]).max() if big.any() else 0:.2e}"
                )
                if dis:
                    idx = np.nonzero(fa != fb)[0][:5]
                    print("   new", a[idx], "old", b[idx])
            # gradients (coef, x, v) on the bounded call
            x, v, rem = x[~sens], v[~sens], rem[~sens]
            if x.shape[0] == 0:
                continue
            ga = jax.grad(tsum)(ls.coef, ls, x, v, rem, L)
            gb = jax.grad(tsum)(ls.coef, ls, x, v, rem, OLD)
            sc = float(np.abs(np.asarray(gb)).max()) or 1.0
            gxa, gva = jax.grad(txv, argnums=(0, 1))(x, v, ls, rem, L)
            gxb, gvb = jax.grad(txv, argnums=(0, 1))(x, v, ls, rem, OLD)
            ex = float(np.abs(np.asarray(gxa - gxb)).max()) / max(float(np.abs(gxb).max()), 1e-300)
            ev = float(np.abs(np.asarray(gva - gvb)).max()) / max(float(np.abs(gvb).max()), 1e-300)
            print(
                f"   grad coef rel {float(np.abs(np.asarray(ga - gb)).max()) / sc:.2e}  "
                f"x rel {ex:.2e}  v rel {ev:.2e}",
                flush=True,
            )
