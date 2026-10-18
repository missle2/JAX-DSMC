"""Toy: leverage of the per-cell quad / cubicv fits at the virtual pull-back's post-collision
queries, for a thermalised cell versus a cold two-population (beam + wall-diffuse) cell.
numpy only, seconds.  Velocities in units of v_s = sqrt(kT/m); positions in cell widths."""

import itertools

import numpy as np

rng = np.random.default_rng(0)


def monomials(poly):
    vel = (2, 3, 4)
    t = [()] + [(a,) for a in range(5)]
    t += list(itertools.combinations_with_replacement(vel, 2)) + [(0, 0), (1, 1)]
    if poly == "cubicv":
        t += list(itertools.combinations_with_replacement(vel, 3))
    return t


def design(o, poly):
    return np.stack([np.prod([o[:, a] for a in m], axis=0) if m else np.ones(len(o))
                     for m in monomials(poly)], axis=1)  # fmt: skip


def queries(v, n):
    """Post-collision velocities of random node pairs (isotropic scattering, equal masses)."""
    i, j = rng.integers(0, len(v), (2, n))
    g = np.linalg.norm(v[i] - v[j], axis=1, keepdims=True)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return 0.5 * (v[i] + v[j]) + 0.5 * g * u


def report(name, v, n_nodes):
    x = rng.uniform(-0.5, 0.5, (n_nodes, 2))
    pts = np.concatenate((x, v), axis=1)
    centre = pts.mean(0)
    qv = queries(v, 4000)
    q = np.concatenate((rng.uniform(-0.5, 0.5, (4000, 2)), qv), axis=1)
    line = f"{name:34s}"
    for poly in ("quad", "cubicv"):
        f = design(pts - centre, poly)
        m = f.T @ f
        minv = np.linalg.inv(m)
        fq = design(q - centre, poly)
        h_node = np.einsum("ni,ij,nj->n", f, minv, f)
        h_q = np.einsum("ni,ij,nj->n", fq, minv, fq)
        line += (f" | {poly}: cond {np.linalg.cond(m):8.1e} node h max {h_node.max():.2f}"
                 f"  query h median {np.median(h_q):6.2f} p99 {np.percentile(h_q, 99):8.1f}")  # fmt: skip
    print(line)


n = 100
report("thermalised shock layer (T x 30)", rng.normal(0, np.sqrt(30), (n, 3)), n)
report("free stream M10 (one Maxwellian)", rng.normal(0, 1, (n, 3)) + [12.9, 0, 0], n)
beam = rng.normal(0, 1, (n // 2, 3)) + [12.9, 0, 0]
wall = rng.normal(0, 1, (n // 2, 3))
wall[:, 1] = np.abs(wall[:, 1]) * 1.0  # half-range off the wall
report("beam + wall-diffuse (50/50)", np.concatenate((beam, wall)), n)
beam = rng.normal(0, 1, (85, 3)) + [12.9, 0, 0]
report("beam + wall-diffuse (85/15)", np.concatenate((beam, wall[:15])), n)


def fractions(name, v):
    x = rng.uniform(-0.5, 0.5, (len(v), 2))
    pts = np.concatenate((x, v), axis=1)
    centre = pts.mean(0)
    q = np.concatenate((rng.uniform(-0.5, 0.5, (4000, 2)), queries(v, 4000)), axis=1)
    f = design(pts - centre, "cubicv")
    fq = design(q - centre, "cubicv")
    h = np.einsum("ni,ij,nj->n", fq, np.linalg.inv(f.T @ f), fq)
    print(f"{name:34s} cubic h > 1 / 2 / 4: "
          + " / ".join(f"{np.mean(h > t):.3f}" for t in (1, 2, 4)))  # fmt: skip


print()
for n in (40, 100, 200):
    fractions(f"thermalised T x 30, n {n}", rng.normal(0, np.sqrt(30), (n, 3)))
    fractions(f"Maxwellian, n {n}", rng.normal(0, 1, (n, 3)))
    beam = rng.normal(0, 1, (n // 2, 3)) + [12.9, 0, 0]
    wall = rng.normal(0, 1, (n - n // 2, 3))
    fractions(f"beam + wall, n {n}", np.concatenate((beam, wall)))
