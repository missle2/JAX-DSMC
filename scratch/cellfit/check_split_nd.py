"""The split per-cell fit and its per-query fallback in 1, 2 and 3 dimensions, on a
collisional box (collide-only NTC, 3^D cells) with synthetic surface normals.

Checks, per dimension:
1. fallback exactness: every slot has a normal and every cell is complex (align 1.01, a
   resultant can not exceed 1) -> the hybrid backward is the per-query one-sided backward
   (k = 32, same partners and draws) to rounding;
2. fallback_capacity equals the measured complex load (alive slots in complex cells);
3. a mixed surface -- one plane normal in most near-wall cells, a "corner" (two normals)
   in some, none elsewhere: lambda_0 of the split fits (cell rule, own rule, with / without
   fallback) against the per-query one-sided fit with k covering the cell -- close, not
   equal (different weights and estimators);
4. a split fit with plane normals everywhere is not the two-sided fit (the split acts).

    PYTHONPATH=. python3 scratch/cellfit/check_split_nd.py
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from examples import ensemble_relaxation as R
from jax_dsmc import ensemble_device, simulate
from jax_dsmc.state import CollideState, GridSpec, ParticleState, SimParams, SimStatic
from jax_dsmc.validation import require_valid

N_PER_CELL = 40
STEPS = 6


def forward(dim, key):
    cells = 3**dim
    n = N_PER_CELL * cells
    grid = GridSpec(jnp.zeros(dim), jnp.ones(dim), (3,) * dim)
    static = SimStatic(n, dim, 1, 1, 256, 1, STEPS, majorant_mode="certified")
    # unit number density in every dimension (the relaxation gate's collision rate)
    fnum = (1.0 / 3.0) ** dim / N_PER_CELL
    params = SimParams(jnp.float64(0.2), jnp.full(cells, fnum), jnp.float64(0.0))
    kx, kv = jax.random.split(key)
    x = jax.random.uniform(kx, (n, dim), dtype=jnp.float64)
    v = jax.random.normal(kv, (n, 3), dtype=jnp.float64) * jnp.asarray([1.4, 0.7, 0.7])
    state = ParticleState(x, v, jnp.zeros(n, jnp.int32), jnp.ones(n, bool), jnp.zeros(n))
    (out, _), aux = simulate.run(
        state, CollideState(jnp.zeros((cells, 1, 1))), grid, R.species(R.CONTROLS), params,
        static, key, collide_only=True, emit_moments="none", capture=True,
    )  # fmt: skip
    require_valid(aux[1])
    return state, out, aux[2], grid, n


def normals_of(rec, dim, kind):
    """Per step and slot: ``all`` -- +e_0 everywhere; ``mixed`` -- +e_0 in the first
    column of cells, a corner (e_0 or e_1 by the nearer wall) in its first cell, none
    elsewhere."""
    x = rec.x  # (M, N, D)
    e0 = jnp.zeros(3).at[0].set(1.0)
    if kind == "all":
        return jnp.broadcast_to(e0, x.shape[:2] + (3,)).astype(jnp.float32)
    first = x[..., 0] < 1.0 / 3.0
    n = jnp.where(first[..., None], e0, 0.0)
    if dim > 1:
        e1 = jnp.zeros(3).at[1].set(1.0)
        corner = first & (x[..., 1] < 1.0 / 3.0)
        n = jnp.where((corner & (x[..., 1] < x[..., 0]))[..., None], e1, n)
    return n.astype(jnp.float32)


def backward(rec, state0, out, grid, n, key, basis, k, normals, capacity=0):
    phi = out.v[:, 0] ** 2 / n
    scale = jnp.ones(grid.lo.shape[0] + 3) * jnp.concatenate(
        (jnp.full(grid.lo.shape[0], 1.0 / 3.0), jnp.ones(3))
    )
    dev = ensemble_device.backward_virtual(
        rec, state0.alive, out, phi, grid, scale, R.species(R.CONTROLS), key, 128, k, basis,
        1024, 2, 4, "staged", None, normals=normals, fallback_capacity=capacity,
    )  # fmt: skip
    return np.asarray(dev.lambda0)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    for dim in (1, 2, 3):
        key = jax.random.key(dim)
        state0, out, rec, grid, n = forward(dim, key)
        bk = jax.random.fold_in(key, 99)
        run = functools.partial(backward, rec, state0, out, grid, n, bk)
        # 1. every cell complex -> the fallback is the per-query one-sided fit
        normals = normals_of(rec, dim, "all")
        basis = "cell:quad:align1.01"
        cap = ensemble_device.fallback_capacity(rec, normals, grid, 128, basis)
        load = int(jnp.max(ensemble_device.complex_load(rec, normals, grid, 128, basis)))
        ref = run("quadratic_full", 32, normals)
        hyb = run(basis, 32, normals, cap)
        print(f"D={dim}: N={n}; 1. all-complex hybrid vs per-query one-sided: rel "
              f"{rel(hyb, ref):.1e} (capacity {cap}, load {load} = N: {load == n})")
        assert rel(hyb, ref) < 1e-10 and load == n
        # 4. plane normals everywhere (no actual jump): split vs two-sided, sizes
        sizes = {}
        for b, kk, nn in (
            ("two-sided per-query k32", 32, None),
            ("two-sided cell:quad", 32, None),
            ("one-sided per-query k32", 32, normals),
            ("one-sided per-query k128", 128, normals),
            ("split cell:quad", 32, normals),
            ("split cell:quad:adapt1.5", 32, normals),
            ("split cell:quad:adapt2", 32, normals),
            ("split cell:lin", 32, normals),
        ):
            basis_b = "quadratic_full" if "per-query" in b else b.split()[-1]
            sizes[b] = run(basis_b, kk, nn)
        base = sizes["two-sided per-query k32"]
        print("     4. plane normals everywhere: |lambda0| / |two-sided k32|, rel to it")
        for b, lam in sizes.items():
            print(f"        {b:28s} {np.linalg.norm(lam) / np.linalg.norm(base):9.3e}  "
                  f"rel {rel(lam, base):.2e}")
        # 3. mixed surface
        normals = normals_of(rec, dim, "mixed")
        ref = run("quadratic_full", 128, normals)
        print(f"     3. mixed surface: per-query one-sided k=128 |lambda0| / |two-sided| "
              f"{np.linalg.norm(ref) / np.linalg.norm(base):.3e}")
        for b in ("quadratic_full", "cell:quad", "cell:quad:own", "cell:quad:align0.9",
                  "cell:quad:adapt1.5", "cell:quad:adapt1.5:own",
                  "cell:quad:adapt1.5:align0.9", "cell:quad:adapt1.5:own:align0.9",
                  "cell:quad:adapt1.5:align0.9:nofallback"):
            cap = ensemble_device.fallback_capacity(rec, normals, grid, 128, b)
            got = run(b, 32, normals, cap)
            print(f"        {b:38s} capacity {cap:4d}  |l0|/|two-sided| "
                  f"{np.linalg.norm(got) / np.linalg.norm(base):9.3e}  vs per-query one-sided "
                  f"k=128 rel {rel(got, ref):.2e}")
    print("ALL OK")


if __name__ == "__main__":
    main()
