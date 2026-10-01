"""Host-side (numpy) setup of the SPARTA circle deck: grid, surface lines, cut cells, inflow tasks.

Mirrors what SPARTA builds once at ``create_grid`` / ``read_surf`` / ``fix emit/face``:

* ``Surf::compute_line_normal``: 2-D line normal = z x (p2 - p1), normalised (points into the flow
  for SPARTA's clockwise-listed body lines).
* ``Grid::surf2grid`` / ``Cut2d``: per cell the list of lines that touch it (``csurfs``), the cell
  type (OUTSIDE / INSIDE / OVERLAP) and the flow volume (area in 2-D) of cut cells.
* ``FixEmitFace::create_task``: one insertion task per cell face on an inflow boundary, with
  ``ntarget = mol_inflow(indot, vscale, 1) * nrho * area * dt / fnum``.

Everything here is float64 numpy; the JAX step reads the resulting arrays as constants.
"""

from __future__ import annotations

import dataclasses
import math
import os

import numpy as np

BOLTZ = 1.380649e-23

OUTSIDE, INSIDE, OVERLAP = 1, 2, 3

HERE = os.path.dirname(os.path.abspath(__file__))
DECK_DIR = os.path.join(HERE, "deck")


@dataclasses.dataclass(frozen=True)
class Species:
    mass: float = 6.63e-26  # ar.species
    diam: float = 4.11e-10  # ar.vss
    omega: float = 0.81
    tref: float = 273.15
    alpha: float = 1.4


@dataclasses.dataclass(frozen=True)
class Deck:
    """The numbers of in.circle_ar."""

    lo: tuple = (0.0, 0.0)
    hi: tuple = (10.0, 10.0)
    nx: int = 20
    ny: int = 20
    nrho: float = 1.0
    fnum: float = 0.001
    vstream: tuple = (100.0, 0.0, 0.0)
    temp: float = 273.15
    dt: float = 1.0e-4
    surf_file: str = os.path.join(DECK_DIR, "data.circle")
    tsurf: float = 300.0
    acc: float = 0.0  # surf_collide diffuse 300.0 0.0 -> always specular
    species: Species = Species()


def read_surf(path):
    """Parse a SPARTA 2-D surface file: returns (points (P,2), lines (L,2) 0-based p1,p2)."""
    with open(path) as fh:
        text = [ln.split("#")[0].strip() for ln in fh]
    npts = next(int(t.split()[0]) for t in text if t.endswith("points"))
    nlin = next(int(t.split()[0]) for t in text if t.endswith("lines"))
    i = text.index("Points") + 1
    rows = [t for t in text[i:] if t][:npts]
    pts = np.array([[float(r.split()[1]), float(r.split()[2])] for r in rows])
    i = text.index("Lines") + 1
    rows = [t for t in text[i:] if t][:nlin]
    lines = np.array([[int(r.split()[1]) - 1, int(r.split()[2]) - 1] for r in rows], dtype=np.int64)
    return pts, lines


def _clip_segment_to_box(p1, p2, lo, hi):
    """Liang-Barsky: does segment p1-p2 touch the closed box [lo, hi]?"""
    t0, t1 = 0.0, 1.0
    d = p2 - p1
    for a in range(2):
        if d[a] == 0.0:
            if p1[a] < lo[a] or p1[a] > hi[a]:
                return False
            continue
        ta = (lo[a] - p1[a]) / d[a]
        tb = (hi[a] - p1[a]) / d[a]
        t0 = max(t0, min(ta, tb))
        t1 = min(t1, max(ta, tb))
    return t0 <= t1


def _clip_polygon_to_box(poly, lo, hi):
    """Sutherland-Hodgman clip of a polygon (V,2) against an axis-aligned box."""
    out = list(poly)
    for a in range(2):
        for bound, keep_ge in ((lo[a], True), (hi[a], False)):
            inp, out = out, []
            if not inp:
                break
            for k in range(len(inp)):
                cur, prev = inp[k], inp[k - 1]
                cin = cur[a] >= bound if keep_ge else cur[a] <= bound
                pin = prev[a] >= bound if keep_ge else prev[a] <= bound
                if cin != pin:
                    t = (bound - prev[a]) / (cur[a] - prev[a])
                    out.append(prev + t * (cur - prev))
                if cin:
                    out.append(cur)
    return np.array(out) if out else np.zeros((0, 2))


def _area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _inside_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for k in range(n):
        (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def mol_inflow(indot, vscale, fraction=1.0):
    """FixEmit::mol_inflow: inward number flux per unit density (Bird 4.22)."""
    s = indot / vscale
    if s < -3.0:
        return 0.0
    return (
        vscale
        * fraction
        * (math.exp(-s * s) + math.sqrt(math.pi) * s * (1 + math.erf(s)))
        / (2 * math.sqrt(math.pi))
    )


@dataclasses.dataclass(frozen=True)
class Geometry:
    """All static arrays of the deck (numpy)."""

    deck: Deck
    dx: float
    dy: float
    ncell: int
    cell_type: np.ndarray  # (C,) OUTSIDE / INSIDE / OVERLAP
    volume: np.ndarray  # (C,) flow area
    csurfs: np.ndarray  # (C, K) line ids touching the cell, -1 padded
    nsurf: np.ndarray  # (C,) number of lines in the cell
    cut_prefix: np.ndarray  # (ny+1, nx+1) 2-D prefix sum of the OVERLAP indicator
    p1: np.ndarray  # (L, 2)
    p2: np.ndarray  # (L, 2)
    norm: np.ndarray  # (L, 2)
    length: np.ndarray  # (L,)
    task_cell: np.ndarray  # (T,) emit/face tasks (xlo face cells)
    task_lo: np.ndarray  # (T,) y-extent of each task face
    task_hi: np.ndarray
    task_ntarget: np.ndarray  # (T,)
    vscale: float
    body_polygon: np.ndarray  # closed polygon of the body (ccw), for inside tests


def build(deck: Deck | None = None) -> Geometry:  # noqa: PLR0915
    deck = deck or Deck()
    nx, ny = deck.nx, deck.ny
    lo, hi = np.array(deck.lo), np.array(deck.hi)
    dx, dy = (hi[0] - lo[0]) / nx, (hi[1] - lo[1]) / ny
    pts, lines = read_surf(deck.surf_file)
    p1, p2 = pts[lines[:, 0]], pts[lines[:, 1]]
    d = p2 - p1
    length = np.hypot(d[:, 0], d[:, 1])
    norm = np.stack([-d[:, 1], d[:, 0]], axis=1) / length[:, None]  # z x delta
    # body polygon: walk the lines from p2 to p1 (lines are listed clockwise w.r.t. the body)
    poly = pts[lines[::-1, 1]]
    if np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - poly[:, 1] * np.roll(poly[:, 0], -1)) < 0:
        poly = poly[::-1]

    ncell = nx * ny
    cell_type = np.zeros(ncell, np.int32)
    volume = np.zeros(ncell)
    lists = []
    for c in range(ncell):
        i, j = c % nx, c // nx
        clo = np.array([lo[0] + i * dx, lo[1] + j * dy])
        chi = np.array([lo[0] + (i + 1) * dx, lo[1] + (j + 1) * dy])
        ids = [m for m in range(len(p1)) if _clip_segment_to_box(p1[m], p2[m], clo, chi)]
        lists.append(ids)
        full = dx * dy
        if ids:
            cell_type[c] = OVERLAP
            volume[c] = full - _area(_clip_polygon_to_box(poly, clo, chi))
        elif _inside_polygon(0.5 * (clo + chi), poly):
            cell_type[c] = INSIDE
            volume[c] = 0.0
        else:
            cell_type[c] = OUTSIDE
            volume[c] = full
    K = max(1, *(len(s) for s in lists))
    csurfs = -np.ones((ncell, K), np.int32)
    for c, s in enumerate(lists):
        csurfs[c, : len(s)] = s
    nsurf = np.array([len(s) for s in lists], np.int32)
    cut = (cell_type == OVERLAP).reshape(ny, nx).astype(np.int32)
    cut_prefix = np.zeros((ny + 1, nx + 1), np.int32)
    cut_prefix[1:, 1:] = cut.cumsum(0).cumsum(1)

    # fix emit/face ar xlo: one task per cell on the xlo boundary (all OUTSIDE cells here)
    sp = deck.species
    vscale = math.sqrt(2.0 * BOLTZ * deck.temp / sp.mass)
    indot = deck.vstream[0]  # inward normal of xlo is +x
    task_cell, task_lo, task_hi, task_nt = [], [], [], []
    for j in range(ny):
        c = j * nx + 0
        if cell_type[c] != OUTSIDE:
            raise NotImplementedError("emit/face on cut cells is not needed for this deck")
        area = dy
        nt = mol_inflow(indot, vscale) * deck.nrho * area * deck.dt / deck.fnum
        task_cell.append(c)
        task_lo.append(lo[1] + j * dy)
        task_hi.append(lo[1] + (j + 1) * dy)
        task_nt.append(nt)
    return Geometry(
        deck=deck,
        dx=dx,
        dy=dy,
        ncell=ncell,
        cell_type=cell_type,
        volume=volume,
        csurfs=csurfs,
        nsurf=nsurf,
        cut_prefix=cut_prefix,
        p1=p1,
        p2=p2,
        norm=norm,
        length=length,
        task_cell=np.array(task_cell, np.int32),
        task_lo=np.array(task_lo),
        task_hi=np.array(task_hi),
        task_ntarget=np.array(task_nt),
        vscale=vscale,
        body_polygon=poly,
    )
