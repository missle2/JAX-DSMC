"""Host-side (numpy) setup of SPARTA's 3-D benchmark decks: grid, triangles, cut cells, emit tasks.

What SPARTA builds once at ``create_grid`` / ``read_surf`` / ``fix emit/face`` / ``create_particles``:

* ``Surf::compute_tri_normal``: normal = (p2 - p1) x (p3 - p1), normalised; for ``data.sphere`` it
  points out of the body, into the flow (checked).
* ``Grid::surf2grid`` + ``Cut3d``: per cell the triangles that touch the closed cell box
  (``csurfs``), the cell type (OUTSIDE / INSIDE / OVERLAP) and the flow volume of cut cells.
* ``FixEmitFace::create_task``: one task per (boundary cell, emitting face) with
  ``ntarget = mol_inflow(indot, vscale) * nrho * area * dt / fnum``.
* ``CreateParticles::create_local``: per-cell counts from the running flow volume.

Only uniform grids, no split cells (raised), one species.  Everything is float64 numpy; the step
(``sim3d``) reads the arrays as constants.  Large grids are handled without per-cell Python
loops: per-cell tables are only built for cut cells (a sorted list of global cell ids).

**Cut-cell volume (exact, no Monte Carlo).**  For the region R = cell box ∩ body, the divergence
theorem with F = (x - xlo, 0, 0) gives

    V(R) = sum over the body triangles clipped to the cell of n_x * area * (centroid_x - xlo)
           + dx * A,    A = area of the cell's x = xhi face inside the body,

because F vanishes on x = xlo and is tangent to the y and z faces.  A is the same identity one
dimension down on that face (G = (y - ylo, 0)): the body's section segments clipped to the face
plus dy times the length of the face's (y = yhi) edge inside the body, and that length is the
1-D identity on the edge: sum of (z_k - zlo) * sign_k over the body crossings z_k on the edge
plus dz if the edge's upper end is inside.  Crossings and inside/outside of the edge come from
rays along z through grid-node columns (main's top-left-rule parity, ``_column_crossings``).
The sum of V over all cells equals the body volume (``check``).
"""

from __future__ import annotations

import dataclasses
import math
import os

import numpy as np

BOLTZ = 1.380649e-23
OUTSIDE, INSIDE, OVERLAP = 1, 2, 3

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(HERE, "deck", "bench")


# ---------------------------------------------------------------------------------------------
# species
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Species:
    """One SPARTA species with its VSS parameters (ar.species / ar.vss)."""

    name: str = "Ar"
    mass: float = 6.63e-26
    diam: float = 4.11e-10
    omega: float = 0.81
    tref: float = 273.15
    alpha: float = 1.4


def read_species(species_file, vss_file, name):
    """Read one species from SPARTA ``.species`` and ``.vss`` files."""

    def rows(path):
        out = {}
        with open(path) as fh:
            for ln in fh:
                w = ln.split("#")[0].split()
                if w:
                    out[w[0]] = w[1:]
        return out

    sp, vs = rows(species_file)[name], rows(vss_file)[name]
    if int(float(sp[2])) != 0 or int(float(sp[4])) != 0:
        raise NotImplementedError(f"{name}: internal energy modes are not implemented")
    return Species(
        name=name,
        mass=float(sp[1]),
        diam=float(vs[0]),
        omega=float(vs[1]),
        tref=float(vs[2]),
        alpha=float(vs[3]),
    )


def mol_inflow(indot, vscale, fraction=1.0):
    """FixEmit::mol_inflow: inward number flux per unit density (Bird eq. 4.22)."""
    s = indot / vscale
    if s < -3.0:
        return 0.0
    return (
        vscale
        * fraction
        * (math.exp(-s * s) + math.sqrt(math.pi) * s * (1 + math.erf(s)))
        / (2 * math.sqrt(math.pi))
    )


# ---------------------------------------------------------------------------------------------
# surfaces
# ---------------------------------------------------------------------------------------------


def read_surf_tris(path):
    """SPARTA 3-D surface file -> (points (P,3), tris (T,3) 0-based).  Port of main's
    ``read_sparta_surf`` reduced to what the bench needs (ids may be unordered)."""
    with open(path) as fh:
        text = [ln.split("#")[0].strip() for ln in fh]
    npts = ntri = None
    for t in text[1:]:
        w = t.split()
        if len(w) == 2 and w[1] == "points":
            npts = int(w[0])
        elif len(w) == 2 and w[1] == "triangles":
            ntri = int(w[0])
    i = text.index("Points") + 1
    rows = [t.split() for t in text[i:] if t][:npts]
    pts = np.full((npts, 3), np.nan)
    for r in rows:
        pts[int(r[0]) - 1] = [float(v) for v in r[1:4]]
    i = text.index("Triangles") + 1
    rows = [t.split() for t in text[i:] if t][:ntri]
    tris = np.full((ntri, 3), -1, np.int64)
    for r in rows:
        tris[int(r[0]) - 1] = [int(v) - 1 for v in r[-3:]]
    if np.isnan(pts).any() or (tris < 0).any():
        raise ValueError(f"{path}: incomplete Points/Triangles")
    return pts, tris


def tri_box_overlap(v, lo, hi):
    """Inclusive triangle / box separating-axis test, ``bool[M]`` (port of main's
    ``mesh.cut._tri_box_overlap``).  ``v`` (M,3,3); boxes ``lo``/``hi`` (M,3)."""
    c = 0.5 * (lo + hi)
    h = 0.5 * (hi - lo)
    rel = v - c[:, None, :]
    slack = 1e-12 * np.max(hi - lo, axis=1)
    ok = ((rel.min(axis=1) <= h + slack[:, None]) & (rel.max(axis=1) >= -h - slack[:, None])).all(
        axis=1
    )
    edges = np.stack([rel[:, 1] - rel[:, 0], rel[:, 2] - rel[:, 1], rel[:, 0] - rel[:, 2]], axis=1)
    n = np.cross(edges[:, 0], edges[:, 1])
    d = np.einsum("ij,ij->i", n, rel[:, 0])
    r = np.einsum("ij,ij->i", h, np.abs(n)) + slack * np.linalg.norm(n, axis=1)
    ok &= np.abs(d) <= r
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = 1.0
        for j in range(3):
            ax = np.cross(e, edges[:, j])
            proj = np.einsum("ik,ijk->ij", ax, rel)
            r = np.einsum("ij,ij->i", h, np.abs(ax)) + slack * np.linalg.norm(ax, axis=1)
            ok &= (proj.min(axis=1) <= r) & (proj.max(axis=1) >= -r)
    return ok


def _column_crossings(px, py, v, n):
    """Crossings of the vertical lines (px[m], py[m], z) with the triangles: returns
    (m_index, z, sign) with sign = sign(n_z) (+1: the body is left going +z).

    Point-in-projected-triangle uses edge functions with the top-left tie rule (main's
    ``_cross_pairs``), so a line through a shared edge or vertex is counted exactly once."""
    vlo = v[:, :, :2].min(axis=1)
    vhi = v[:, :, :2].max(axis=1)
    out_m, out_z, out_s = [], [], []
    chunk = max(1, 4_000_000 // max(1, len(v)))
    for s0 in range(0, len(px), chunk):
        qx, qy = px[s0 : s0 + chunk], py[s0 : s0 + chunk]
        cand = (
            (qx[:, None] >= vlo[None, :, 0])
            & (qx[:, None] <= vhi[None, :, 0])
            & (qy[:, None] >= vlo[None, :, 1])
            & (qy[:, None] <= vhi[None, :, 1])
        )
        mi, ti = np.nonzero(cand)
        if mi.size == 0:
            continue
        a, b, c = v[ti, 0], v[ti, 1], v[ti, 2]
        x0, y0 = qx[mi], qy[mi]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        flip = area < 0
        b2 = np.where(flip[:, None], c, b)
        c2 = np.where(flip[:, None], b, c)
        inside = area != 0
        for (ix, iy), (jx, jy) in (
            ((a[:, 0], a[:, 1]), (b2[:, 0], b2[:, 1])),
            ((b2[:, 0], b2[:, 1]), (c2[:, 0], c2[:, 1])),
            ((c2[:, 0], c2[:, 1]), (a[:, 0], a[:, 1])),
        ):
            ex, ey = jx - ix, jy - iy
            edge = ex * (y0 - iy) - ey * (x0 - ix)
            top_left = (ey < 0) | ((ey == 0) & (ex > 0))
            inside &= (edge > 0) | ((edge == 0) & top_left)
        mi, ti = mi[inside], ti[inside]
        nz = n[ti, 2]
        good = nz != 0
        mi, ti, nz = mi[good], ti[good], nz[good]
        # plane: n . (p - a) = 0 -> z
        aa = v[ti, 0]
        z = aa[:, 2] - (n[ti, 0] * (qx[mi] - aa[:, 0]) + n[ti, 1] * (qy[mi] - aa[:, 1])) / nz
        out_m.append(mi + s0)
        out_z.append(z)
        out_s.append(np.sign(nz))
    if not out_m:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    return np.concatenate(out_m), np.concatenate(out_z), np.concatenate(out_s)


# Rays along z are traced through points shifted off the grid nodes by these fractions of a cell:
# a node can coincide with a mesh vertex where the ray is tangent to the body (data.sphere's
# (-1, 0, 0) is a node of the 10^3 grid), and there the crossing parity breaks.  The shift
# changes an edge length by O(1e-10 h), i.e. nothing.
COLUMN_SHIFT = (1.0e-10 * (math.sqrt(2.0) - 1.0), 1.0e-10 * (math.sqrt(3.0) - 1.0))


def _inside_column(m, z, s, ncol, zq):
    """Inside the body at heights zq[(ncol, Q)] for each column: parity of crossings above."""
    above = np.zeros(zq.shape, np.int64)
    for q in range(zq.shape[1]):
        np.add.at(above[:, q], m, (z > zq[m, q]).astype(np.int64))
    return above % 2 == 1


def _clip_poly(poly, axis, bound, keep_ge):
    """Sutherland-Hodgman clip of a convex polygon (V,3) by one axis-aligned half-space."""
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for k in range(n):
        cur, prev = poly[k], poly[k - 1]
        cin = cur[axis] >= bound if keep_ge else cur[axis] <= bound
        pin = prev[axis] >= bound if keep_ge else prev[axis] <= bound
        if cin != pin:
            t = (bound - prev[axis]) / (cur[axis] - prev[axis])
            p = prev + t * (cur - prev)
            p[axis] = bound
            out.append(p)
        if cin:
            out.append(cur)
    return np.array(out) if out else np.zeros((0, 3))


def _poly_area_centroid(poly, normal):
    """Area and centroid of a planar polygon (V,3) with unit normal."""
    if len(poly) < 3:
        return 0.0, np.zeros(3)
    p0 = poly[0]
    a = poly[1:-1] - p0
    b = poly[2:] - p0
    cr = np.cross(a, b) @ normal * 0.5  # signed areas of the fan triangles
    cen = (p0 + poly[1:-1] + poly[2:]) / 3.0
    area = cr.sum()
    if area == 0.0:
        return 0.0, p0
    return abs(area), (cr[:, None] * cen).sum(axis=0) / area


def _segment_in_plane(tri, axis, value):
    """Intersection segment of a triangle (3,3) with the plane x[axis] = value, or None.

    Half-open: a vertex on the plane counts as the + side (as if the plane sat at value - 0).
    The segment is the boundary between the triangle's {x < value} and {x >= value} parts, so an
    edge lying in the plane is produced by exactly one of its two triangles (the one on the -
    side) instead of both."""
    d = tri[:, axis] - value
    pos = d >= 0.0
    pts = []
    for k in range(3):
        a, b = tri[k], tri[(k + 1) % 3]
        da, db = d[k], d[(k + 1) % 3]
        if pos[k] != pos[(k + 1) % 3]:
            t = da / (da - db)
            pts.append(a + t * (b - a))
    if len(pts) != 2 or np.array_equal(pts[0], pts[1]):
        return None
    return pts[0], pts[1]


def _clip_seg_2d(p, q, lo, hi):
    """Liang-Barsky clip of a 2-D segment to [lo, hi]; returns the clipped (p, q) or None."""
    t0, t1 = 0.0, 1.0
    d = q - p
    for a in range(2):
        if d[a] == 0.0:
            if p[a] < lo[a] or p[a] > hi[a]:
                return None
            continue
        ta, tb = (lo[a] - p[a]) / d[a], (hi[a] - p[a]) / d[a]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
    if t0 >= t1:
        return None
    return p + t0 * d, p + t1 * d


# ---------------------------------------------------------------------------------------------
# geometry build
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Grid:
    """Uniform 3-D grid: cell (i, j, k) has global id i + nx (j + ny k)."""

    lo: tuple
    hi: tuple
    n: tuple

    @property
    def h(self):
        return tuple((self.hi[a] - self.lo[a]) / self.n[a] for a in range(3))

    @property
    def ncell(self):
        return self.n[0] * self.n[1] * self.n[2]

    def cell_box(self, ijk):
        """lo, hi (M,3) of the cells with integer indices ijk (M,3) -- SPARTA's cell lo/hi."""
        ijk = np.asarray(ijk, np.float64)
        lo = np.array(self.lo)
        h = np.array(self.h)
        return lo + ijk * h, lo + (ijk + 1.0) * h

    def unravel(self, gid):
        gid = np.asarray(gid, np.int64)
        nx, ny, _ = self.n
        return np.stack([gid % nx, (gid // nx) % ny, gid // (nx * ny)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Surf:
    p1: np.ndarray  # (T,3)
    p2: np.ndarray
    p3: np.ndarray
    norm: np.ndarray  # (T,3) unit, into the flow
    area: np.ndarray  # (T,)

    @property
    def verts(self):
        return np.stack([self.p1, self.p2, self.p3], axis=1)


def make_surf(pts, tris) -> Surf:
    p1, p2, p3 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    n = np.cross(p2 - p1, p3 - p1)
    ln = np.linalg.norm(n, axis=1)
    return Surf(p1=p1, p2=p2, p3=p3, norm=n / ln[:, None], area=0.5 * ln)


def body_volume(s: Surf) -> float:
    """Volume enclosed by the closed triangle mesh (normals out of the body)."""
    return float(np.sum(np.einsum("ij,ij->i", s.p1, np.cross(s.p2, s.p3))) / 6.0)


@dataclasses.dataclass(frozen=True)
class CutCells:
    """Per cut cell (sorted by global id): the triangles touching the closed cell box."""

    gid: np.ndarray  # (M,) int64 sorted
    csurfs: np.ndarray  # (M, K) int32, -1 padded (SPARTA order: ascending triangle id)
    nsurf: np.ndarray  # (M,) int32
    volume: np.ndarray  # (M,) flow volume
    xout: np.ndarray  # (M,3) a point of the cell in the flow (SPARTA's xcell), for outside tests


def cut_cells(grid: Grid, s: Surf) -> CutCells:
    """csurfs of every cell the surface touches, and the exact flow volumes (module docstring)."""
    v = s.verts
    h = np.array(grid.h)
    lo = np.array(grid.lo)
    # candidate cells: every cell in each triangle's bounding box (inclusive, +-1 cell for faces)
    pairs_c, pairs_t = [], []
    vlo = np.floor((v.min(axis=1) - lo) / h).astype(np.int64) - 1
    vhi = np.floor((v.max(axis=1) - lo) / h).astype(np.int64) + 1
    nn = np.array(grid.n)
    vlo = np.clip(vlo, 0, nn - 1)
    vhi = np.clip(vhi, 0, nn - 1)
    for t in range(len(v)):
        ii, jj, kk = np.meshgrid(
            np.arange(vlo[t, 0], vhi[t, 0] + 1),
            np.arange(vlo[t, 1], vhi[t, 1] + 1),
            np.arange(vlo[t, 2], vhi[t, 2] + 1),
            indexing="ij",
        )
        ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
        pairs_c.append(ijk)
        pairs_t.append(np.full(len(ijk), t))
    ijk = np.concatenate(pairs_c)
    tid = np.concatenate(pairs_t)
    clo, chi = grid.cell_box(ijk)
    ok = tri_box_overlap(v[tid], clo, chi)
    ijk, tid = ijk[ok], tid[ok]
    gid = ijk[:, 0] + nn[0] * (ijk[:, 1] + nn[1] * ijk[:, 2])
    order = np.lexsort((tid, gid))
    gid, tid, ijk = gid[order], tid[order], ijk[order]
    ucell, first, counts = np.unique(gid, return_index=True, return_counts=True)
    K = int(counts.max())
    M = len(ucell)
    csurfs = -np.ones((M, K), np.int32)
    rank = np.arange(len(gid)) - np.repeat(first, counts)
    csurfs[np.repeat(np.arange(M), counts), rank] = tid
    cijk = grid.unravel(ucell)
    clo, chi = grid.cell_box(cijk)
    volume, xout = _cut_volumes(grid, s, cijk, clo, chi, csurfs)
    return CutCells(
        gid=ucell.astype(np.int64),
        csurfs=csurfs,
        nsurf=counts.astype(np.int32),
        volume=volume,
        xout=xout,
    )


def _cut_volumes(grid: Grid, s: Surf, cijk, clo, chi, csurfs):
    """Exact flow volume of each cut cell and one flow point per cut cell."""
    v = s.verts
    h = np.array(grid.h)
    M = len(cijk)
    # node columns at (x_hi, y_hi) of every cut cell, rays along z
    cols = np.unique(cijk[:, :2] + 1, axis=0)
    px = grid.lo[0] + (cols[:, 0] + COLUMN_SHIFT[0]) * h[0]
    py = grid.lo[1] + (cols[:, 1] + COLUMN_SHIFT[1]) * h[1]
    cm, cz, cs = _column_crossings(px, py, v, s.norm)
    col_index = {tuple(c): m for m, c in enumerate(cols)}
    by_col = [[] for _ in range(len(cols))]
    for m, z, sg in zip(cm, cz, cs, strict=True):
        by_col[m].append((z, sg))
    by_col = [np.array(b) if b else np.zeros((0, 2)) for b in by_col]

    def edge_inside_length(m, zlo, zhi):
        b = by_col[m]
        if len(b) == 0:
            return 0.0
        z, sg = b[:, 0], b[:, 1]
        inside_hi = np.sum(z > zhi) % 2 == 1
        sel = (z >= zlo) & (z <= zhi)
        return float(np.sum((z[sel] - zlo) * sg[sel]) + (zhi - zlo) * inside_hi)

    inside_vol = np.zeros(M)
    for c in range(M):
        lo_c, hi_c = clo[c], chi[c]
        ids = csurfs[c][csurfs[c] >= 0]
        vol = 0.0
        # 3-D: clipped triangle pieces
        for t in ids:
            poly = v[t].copy()
            for a in range(3):
                poly = _clip_poly(poly, a, lo_c[a], True)
                poly = _clip_poly(poly, a, hi_c[a], False)
            area, cen = _poly_area_centroid(poly, s.norm[t])
            vol += s.norm[t, 0] * area * (cen[0] - lo_c[0])
        # 2-D: the x = xhi face inside the body
        face = 0.0
        for t in ids:
            seg = _segment_in_plane(v[t], 0, hi_c[0])
            if seg is None:
                continue
            p, q = seg[0][1:], seg[1][1:]
            cl = _clip_seg_2d(p, q, lo_c[1:], hi_c[1:])
            if cl is None:
                continue
            n2 = s.norm[t, 1:]
            ln2 = np.linalg.norm(n2)
            if ln2 == 0.0:
                continue
            n2 = n2 / ln2
            seglen = np.linalg.norm(cl[1] - cl[0])
            face += n2[0] * seglen * (0.5 * (cl[0][0] + cl[1][0]) - lo_c[1])
        m = col_index[(int(cijk[c, 0]) + 1, int(cijk[c, 1]) + 1)]
        face += (hi_c[1] - lo_c[1]) * edge_inside_length(m, lo_c[2], hi_c[2])
        vol += (hi_c[0] - lo_c[0]) * face
        inside_vol[c] = vol
    full = float(np.prod(h))
    flow = full - inside_vol
    xout = _flow_points(grid, s, clo, chi, csurfs)
    return flow, xout


def _flow_points(grid: Grid, s: Surf, clo, chi, csurfs, tries=64, seed=12345):
    """A point inside each cut cell that is in the flow (global ray parity), for SPARTA's
    point-outside test (``Grid::point_outside_surfs``: xcell)."""
    rng = np.random.default_rng(seed)
    M = len(clo)
    xout = np.full((M, 3), np.nan)
    todo = np.arange(M)
    v = s.verts
    for _ in range(tries):
        if len(todo) == 0:
            break
        x = clo[todo] + rng.random((len(todo), 3)) * (chi[todo] - clo[todo])
        m, z, _sg = _column_crossings(x[:, 0], x[:, 1], v, s.norm)
        above = np.zeros(len(todo), np.int64)
        np.add.at(above, m, (z > x[m, 2]).astype(np.int64))
        ok = above % 2 == 0
        xout[todo[ok]] = x[ok]
        todo = todo[~ok]
    if len(todo):
        # a cut cell with (almost) no flow: any corner; never used if its flow volume is ~0
        xout[todo] = clo[todo]
    return xout


def classify_uncut(grid: Grid, s: Surf, cut_gid):
    """INSIDE cells among the uncut ones (global ids, sorted): cell centres by column parity.
    Only columns that meet the body's bounding box are traced."""
    v = s.verts
    h = np.array(grid.h)
    lo = np.array(grid.lo)
    blo = np.floor((v.min(axis=(0, 1)) - lo) / h).astype(np.int64)
    bhi = np.floor((v.max(axis=(0, 1)) - lo) / h).astype(np.int64)
    nn = np.array(grid.n)
    blo, bhi = np.clip(blo, 0, nn - 1), np.clip(bhi, 0, nn - 1)
    ii, jj = np.meshgrid(np.arange(blo[0], bhi[0] + 1), np.arange(blo[1], bhi[1] + 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    px = lo[0] + (ii + 0.5 + COLUMN_SHIFT[0]) * h[0]
    py = lo[1] + (jj + 0.5 + COLUMN_SHIFT[1]) * h[1]
    m, z, _sg = _column_crossings(px, py, v, s.norm)
    ks = np.arange(blo[2], bhi[2] + 1)
    zc = lo[2] + (ks + 0.5) * h[2]
    inside = _inside_column(m, z, _sg, len(ii), np.broadcast_to(zc, (len(ii), len(ks))).copy())
    ci, ck = np.nonzero(inside)
    gid = ii[ci] + nn[0] * (jj[ci] + nn[1] * ks[ck])
    gid = np.setdiff1d(gid, cut_gid)
    return np.sort(gid)


def clearance(grid: Grid, cut_gid, cap=127):
    """Per cell the Chebyshev radius r (cells) of its cut-free neighbourhood: every cell within
    Chebyshev distance r is uncut (-1 for a cut cell), capped at ``cap``.  A straight move whose
    end cell is within r of its start cell cannot meet a surface (sim3d dense move)."""
    from scipy import ndimage

    nx, ny, nz = grid.n
    free = np.ones((nz, ny, nx), bool)
    k, rem = np.divmod(cut_gid, nx * ny)
    j, i = np.divmod(rem, nx)
    free[k, j, i] = False
    if free.all():
        return np.full(grid.ncell, cap, np.int8)
    d = ndimage.distance_transform_cdt(free, metric="chessboard")  # 0 on cut cells
    return np.minimum(d.ravel() - 1, cap).astype(np.int8)


def check_volumes(grid: Grid, s: Surf, cut: CutCells, inside_gid) -> dict:
    """Total body volume three ways: mesh, and box - flow volumes - inside cells."""
    full = float(np.prod(grid.h))
    vb = body_volume(s)
    v_cells = (full - cut.volume).sum() + full * len(inside_gid)
    return {"body_volume": vb, "cells_volume": float(v_cells), "rel_err": float(v_cells / vb - 1.0)}


# ---------------------------------------------------------------------------------------------
# the whole deck
# ---------------------------------------------------------------------------------------------

FACE_AXIS = (0, 0, 1, 1, 2, 2)  # XLO XHI YLO YHI ZLO ZHI
FACE_SIGN = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)  # inward normal along the axis


@dataclasses.dataclass(frozen=True)
class Geometry3:
    grid: Grid
    surf: Surf | None
    cut: CutCells | None
    inside_gid: np.ndarray  # sorted global ids of INSIDE cells
    clearance: np.ndarray | None  # (C,) int8, None when there are no surfaces
    task_gid: np.ndarray  # (T,) emit/face tasks: boundary cell
    task_face: np.ndarray  # (T,) face id 0..5
    task_rate: np.ndarray  # (T,) ntarget / dt  (mol_inflow * nrho * area / fnum)
    vscale: float
    n_create: int  # particles create_particles makes
    volume_check: dict | None

    def flow_volume(self):
        """Flow volume of every cell, (C,) float64."""
        full = float(np.prod(self.grid.h))
        vol = np.full(self.grid.ncell, full)
        vol[self.inside_gid] = 0.0
        if self.cut is not None:
            vol[self.cut.gid] = self.cut.volume
        return vol


def emit_tasks(deck, grid: Grid, cell_type_of, vscale):
    """FixEmitFace::create_task for ``fix emit/face <mix> all``: every boundary cell face whose
    cell is OUTSIDE.  OVERLAP boundary cells raise (not needed by the bench decks)."""
    nx, ny, nz = grid.n
    h = grid.h
    gids, faces, rates = [], [], []
    for f in range(6):
        a = FACE_AXIS[f]
        hi_side = f % 2 == 1
        other = [b for b in range(3) if b != a]
        ii, jj = np.meshgrid(np.arange(grid.n[other[0]]), np.arange(grid.n[other[1]]), indexing="ij")
        ijk = np.zeros((ii.size, 3), np.int64)
        ijk[:, a] = grid.n[a] - 1 if hi_side else 0
        ijk[:, other[0]] = ii.ravel()
        ijk[:, other[1]] = jj.ravel()
        gid = ijk[:, 0] + nx * (ijk[:, 1] + ny * ijk[:, 2])
        ct = cell_type_of(gid)
        if np.any(ct == OVERLAP):
            raise NotImplementedError("emit/face on cut cells is not needed for the bench decks")
        gid = gid[ct == OUTSIDE]
        normal = np.zeros(3)
        normal[a] = FACE_SIGN[f]
        indot = float(np.dot(deck.vstream, normal))
        area = h[other[0]] * h[other[1]]
        rate = mol_inflow(indot, vscale) * deck.nrho * area / deck.fnum
        gids.append(gid)
        faces.append(np.full(len(gid), f, np.int8))
        rates.append(np.full(len(gid), rate))
    return np.concatenate(gids), np.concatenate(faces), np.concatenate(rates)


def build(deck) -> Geometry3:
    grid = deck.grid
    sp = deck.species
    vscale = math.sqrt(2.0 * BOLTZ * deck.temp / sp.mass)
    surf = cut = clear = check = None
    inside = np.zeros(0, np.int64)
    if deck.surf_file is not None:
        pts, tris = read_surf_tris(deck.surf_file)
        surf = make_surf(pts, tris)
        cut = cut_cells(grid, surf)
        inside = classify_uncut(grid, surf, cut.gid)
        clear = clearance(grid, cut.gid)
        check = check_volumes(grid, surf, cut, inside)
        if abs(check["rel_err"]) > 1e-8:
            raise RuntimeError(f"cut-cell volumes do not add up to the body: {check}")

    def cell_type_of(gid):
        t = np.full(len(gid), OUTSIDE, np.int32)
        if cut is not None:
            t[np.isin(gid, cut.gid)] = OVERLAP
            t[np.isin(gid, inside)] = INSIDE
        return t

    if deck.emit:
        task_gid, task_face, task_rate = emit_tasks(deck, grid, cell_type_of, vscale)
    else:
        task_gid = np.zeros(0, np.int64)
        task_face = np.zeros(0, np.int8)
        task_rate = np.zeros(0)
    geo = Geometry3(
        grid=grid,
        surf=surf,
        cut=cut,
        inside_gid=inside,
        clearance=clear,
        task_gid=task_gid,
        task_face=task_face,
        task_rate=task_rate,
        vscale=vscale,
        n_create=0,
        volume_check=check,
    )
    if deck.n_create:
        n = int(deck.n_create)
    else:  # create_particles n 0: np = nrho * flowvol / fnum
        n = int(deck.nrho * float(geo.flow_volume().sum()) / deck.fnum)
    return dataclasses.replace(geo, n_create=n)


def create_counts(geo: Geometry3, n: int, seed: int = 0):
    """CreateParticles::create_local on one process: per cell
    ntarget = n * volsum / flowvol - nprev, npercell = int(ntarget) + (U < frac)."""
    vol = geo.flow_volume()
    volsum = np.cumsum(vol)
    total = volsum[-1]
    target = n * volsum / total
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(vol), np.int64)
    # sequential in SPARTA (nprev is the running count); vectorised: nprev_c = sum of counts
    # before c, so ntarget_c - int(ntarget_c) is the fractional part of the running target
    # minus nprev -- done exactly with the cumulative floor + one Bernoulli per cell
    base = np.floor(target).astype(np.int64)
    frac = target - base
    extra = (rng.random(len(vol)) < frac).astype(np.int64)
    cum = base + extra
    cum = np.maximum.accumulate(cum)
    counts[0] = cum[0]
    counts[1:] = np.diff(cum)
    counts[vol <= 0.0] = 0
    return counts


# ---------------------------------------------------------------------------------------------
# cache (build once, e.g. on a login node, before the GPU allocation)
# ---------------------------------------------------------------------------------------------

CACHE_VERSION = 1


def cache_path(deck, cache_dir):
    return os.path.join(cache_dir, f"geo_v{CACHE_VERSION}_{deck.name}_{deck.x}x{deck.y}x{deck.z}.npz")


def build_cached(deck, cache_dir=None) -> Geometry3:
    """``build`` with an on-disk cache (``cache_dir`` or $SPARTA_JAX_GEO_CACHE; none: no cache).
    The cache holds everything ``build`` computes; the deck numbers that enter it (nrho, fnum,
    vstream, temp) are part of the deck name and size, which are the key."""
    cache_dir = cache_dir or os.environ.get("SPARTA_JAX_GEO_CACHE")
    if not cache_dir:
        return build(deck)
    path = cache_path(deck, cache_dir)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        grid = deck.grid
        surf = cut = None
        if deck.surf_file is not None:
            pts, tris = read_surf_tris(deck.surf_file)
            surf = make_surf(pts, tris)
            cut = CutCells(
                gid=z["cut_gid"], csurfs=z["cut_csurfs"], nsurf=z["cut_nsurf"], volume=z["cut_volume"], xout=z["cut_xout"]
            )
        return Geometry3(
            grid=grid,
            surf=surf,
            cut=cut,
            inside_gid=z["inside_gid"],
            clearance=z["clearance"] if "clearance" in z.files else None,
            task_gid=z["task_gid"],
            task_face=z["task_face"],
            task_rate=z["task_rate"],
            vscale=float(z["vscale"]),
            n_create=int(z["n_create"]),
            volume_check=None,
        )
    geo = build(deck)
    os.makedirs(cache_dir, exist_ok=True)
    arrays = {
        "inside_gid": geo.inside_gid,
        "task_gid": geo.task_gid,
        "task_face": geo.task_face,
        "task_rate": geo.task_rate,
        "vscale": np.float64(geo.vscale),
        "n_create": np.int64(geo.n_create),
    }
    if geo.cut is not None:
        arrays.update(
            cut_gid=geo.cut.gid,
            cut_csurfs=geo.cut.csurfs,
            cut_nsurf=geo.cut.nsurf,
            cut_volume=geo.cut.volume,
            cut_xout=geo.cut.xout,
            clearance=geo.clearance,
        )
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return geo
