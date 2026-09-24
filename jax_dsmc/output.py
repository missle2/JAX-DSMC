"""Host-side output: ParaView VTK series, checkpoints, and the run summary.

**Everything here runs outside the jitted step** (spec section 10.2).  Nothing
in this module may be called from inside ``jax.jit``, ``lax.scan`` or a
differentiated region: it writes files, formats strings and branches on
concrete values, all of which are either impossible or silently wrong under
tracing.  The call site is the host loop, after ``jax.block_until_ready`` on
whatever it is about to write.

ParaView output is **three** file kinds, not one
------------------------------------------------
A bare per-step ``.vtr`` is what a first draft reaches for and it is wrong in
two ways that only appear once you actually open the result: ParaView sees N
disconnected datasets rather than one animatable series, and there is no way to
look at the particles at all, only at their cell averages.  For DSMC the
particles are frequently the thing you want to see.

===================  =======================================================
``step_%05d.vtr``    per-cell moments on the rectilinear grid
``part_%05d.vtp``    the particles: points plus ``v``, ``species``, ``alive``
``*.pvd``            ``(timestep -> filename)`` collection; **this is what
                     makes ParaView animate**
``*.pvtr``/``*.pvtp``  one piece per rank plus a master file, for sharded runs
===================  =======================================================

:class:`VTKSeries` drives all four; :func:`write_vtr`, :func:`write_vtp`,
:func:`write_pvd`, :func:`write_pvtr` and :func:`write_pvtp` are the primitives
underneath it.

**Particle output must be subsampled, and the subsample must be stable across
frames.**  At the ``N ~ 1e9`` target, writing every particle is not an option.
The selection is therefore a *deterministic function of particle identity* — a
stride, or a hash of the particle index — and never a fresh random draw per
frame.  A per-frame random subsample shows a different population every step,
which turns an animation into visual noise and makes it impossible to follow
anything.  :func:`stable_subsample_indices` implements both, reusing the
section 1 RNG shape: the selection comes from a fixed key and the particle
index, never from state.

**M5 migrates the cell fields from ``.vtr`` to ``.vtu``, and nothing else
changes.**  Cut and split cells are not a rectilinear grid, so ``.vtr`` cannot
represent them.  The ``.pvd`` collection layer and the ``.vtp`` particle path
are unaffected.  ``.vtr`` is kept until then rather than writing ``.vtu`` from
the start because on a uniform mesh ``.vtr`` *implies* the geometry, while
``.vtu`` must store every point and every cell explicitly — a real cost at
large ``C``.
"""

from __future__ import annotations

import base64
import math
import os
import sys
from typing import Any, TextIO

import numpy as np

# --- low-level VTK XML --------------------------------------------------------

_VTK_TYPE = {
    np.dtype(np.float64): "Float64",
    np.dtype(np.float32): "Float32",
    np.dtype(np.int32): "Int32",
    np.dtype(np.int64): "Int64",
    np.dtype(np.uint8): "UInt8",
}

_XML_HEADER = '<?xml version="1.0"?>'


def _vtk_dtype(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """Coerce to a VTK-representable dtype and return ``(array, type name)``.

    ``bool`` becomes ``UInt8``, which is how ParaView expects a flag; anything
    unrecognised becomes ``Float64``.
    """
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8), "UInt8"
    name = _VTK_TYPE.get(arr.dtype)
    if name is None:
        return arr.astype(np.float64), "Float64"
    return arr, name


def _data_array(
    name: str, data: np.ndarray, ncomp: int, binary: bool, indent: str = " " * 8
) -> str:
    """One ``<DataArray>`` element, inline-base64 or ascii."""
    arr, vtk_type = _vtk_dtype(np.ascontiguousarray(data))
    flat = arr.ravel()
    attrs = f'type="{vtk_type}" Name="{name}" NumberOfComponents="{ncomp}"'
    if binary:
        # VTK inline binary: base64(UInt64 byte count) and base64(payload) as
        # two INDEPENDENT base64 strings, concatenated.  Encoding header and
        # payload together is the classic way to produce a file ParaView
        # silently renders as garbage.
        payload = flat.tobytes()
        header = base64.b64encode(np.array([len(payload)], dtype=np.uint64).tobytes()).decode()
        body = base64.b64encode(payload).decode()
        return f'{indent}<DataArray {attrs} format="binary">\n{header}{body}\n{indent}</DataArray>'
    if vtk_type.startswith("Float"):
        text = " ".join(repr(float(x)) for x in flat)
    else:
        text = " ".join(str(int(x)) for x in flat)
    return f'{indent}<DataArray {attrs} format="ascii">\n{text}\n{indent}</DataArray>'


def _write_lines(path: str, lines: list[str]) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join([*lines, ""]))
    return path


# --- rectilinear grid: per-cell moments ---------------------------------------


def _cell_axes(lo: np.ndarray, hi: np.ndarray, n: tuple[int, ...]) -> list[np.ndarray]:
    """Point coordinates along each of the three VTK axes.

    Axes beyond ``D`` get a single degenerate cell of unit thickness, so a 1D or
    2D run still produces a file ParaView can open.
    """
    axes: list[np.ndarray] = []
    for d in range(3):
        if d < len(n):
            axes.append(np.linspace(float(lo[d]), float(hi[d]), int(n[d]) + 1))
        else:
            axes.append(np.array([0.0, 1.0]))
    return axes


def _to_vtk_cell_order(field: np.ndarray, n: tuple[int, ...]) -> np.ndarray:
    """Reorder a ``[C, ...]`` field from our row-major ravel to VTK's x-fastest.

    ``grid.cell_of`` ravels ``(i0, ..., i_{D-1})`` row-major, so the **last**
    axis varies fastest.  VTK requires the **first** (x) axis to vary fastest.
    The fix is a reversal of the spatial axes only; the component axes are left
    alone.
    """
    comps = field.shape[1:]
    spatial = field.reshape(tuple(int(k) for k in n) + comps)
    order = tuple(reversed(range(len(n)))) + tuple(range(len(n), spatial.ndim))
    return np.ascontiguousarray(spatial.transpose(order)).reshape(-1, *comps)


def _field_items(
    fields: dict[str, Any], C: int, n: tuple[int, ...]
) -> list[tuple[str, np.ndarray]]:
    """Validate, flatten and reorder every per-cell field.  A split-cell grid's sub-cells
    follow its ``C = prod(n)`` Cartesian cells (``GridSpec.n_sub``); the rectilinear
    grid shows the Cartesian cells, i.e. each split cell's largest flow region."""
    items = []
    for name, raw in fields.items():
        arr = np.asarray(raw)
        if arr.shape[0] < C:
            raise ValueError(f"field {name!r} has leading dim {arr.shape[0]}, expected C={C}")
        arr = arr[:C].reshape(C, -1)
        items.append((name, _to_vtk_cell_order(arr, n)))
    return items


def write_vtr(
    path: str,
    lo: Any,
    hi: Any,
    n: tuple[int, ...],
    fields: dict[str, Any],
    binary: bool = True,
) -> str:
    """Write per-cell fields as a VTK ``.vtr`` rectilinear grid.

    Correct and compact for a uniform mesh: the spacing is *implied* by the
    coordinate arrays rather than stored per cell.  M5's cut/split cells will
    need ``.vtu`` instead (see the module docstring).

    Args:
        path: output filename; ``.vtr`` is appended if missing.
        lo: ``f64[D]`` domain lower corner (``grid.lo``).
        hi: ``f64[D]`` domain upper corner (``grid.hi``).
        n: ``tuple[int, ...]`` cells per axis (``grid.n``), STATIC, length ``D``.
        fields: name -> array with leading dimension ``C = prod(n)``.  ``[C]``
            writes a scalar, ``[C, 3]`` a vector, ``[C, 3, 3]`` a 9-component
            tensor; anything else is flattened per cell.
        binary: inline base64 (default) or ascii.  Ascii is human-readable and
            about 4x larger; use it when debugging the writer itself.

    Returns:
        The path written.

    Reference: SPARTA ``dump_grid.cpp``; VTK XML RectilinearGrid.
    """
    if not path.endswith(".vtr"):
        path += ".vtr"
    n = tuple(int(k) for k in n)
    C = math.prod(n)

    axes = _cell_axes(
        np.asarray(lo, dtype=np.float64).reshape(-1),
        np.asarray(hi, dtype=np.float64).reshape(-1),
        n,
    )
    ext = " ".join(f"0 {len(a) - 1}" for a in axes)

    lines = [
        _XML_HEADER,
        '<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64">',
        f'  <RectilinearGrid WholeExtent="{ext}">',
        f'    <Piece Extent="{ext}">',
        "      <Coordinates>",
    ]
    for axis_name, a in zip("XYZ", axes, strict=True):
        lines.append(_data_array(f"{axis_name}_COORDINATES", a, 1, binary))
    lines += ["      </Coordinates>", "      <CellData>"]
    for name, arr in _field_items(fields, C, n):
        lines.append(_data_array(name, arr, arr.shape[1], binary))
    lines += ["      </CellData>", "    </Piece>", "  </RectilinearGrid>", "</VTKFile>"]
    return _write_lines(path, lines)


# --- polydata: the particles themselves ---------------------------------------


def stable_subsample_indices(
    N: int,
    max_points: int,
    key: Any = None,
    method: str = "stride",
) -> np.ndarray:
    """Choose which particle slots to write, **stably across frames**.

    The selection is a deterministic function of the particle *index* only, so
    the same slots appear in every frame and an animation follows the same
    population.  A fresh random draw per frame would show a different set every
    step: visually indistinguishable from noise, and impossible to track
    anything through.

    Args:
        N: particle array capacity.
        max_points: how many to write; ``>= N`` returns everything.
        key: PRNG key for ``method="hash"``.  Fix it once for the whole run —
            reusing one key across frames is the entire point, and matches the
            section 1 contract's shape (derive from a key and an index, never
            from state).
        method: ``"stride"`` takes every ``ceil(N/max_points)``-th slot: cheapest,
            perfectly reproducible, and unbiased as long as slot index carries no
            physical meaning.  ``"hash"`` ranks slots by a fixed threefry hash of
            the index, which is what you want if the allocator ever correlates
            slot index with anything physical.

    Returns:
        ``int64[m]`` sorted slot indices, ``m = min(N, max_points)``.

    Caveat worth knowing: identity here is the *array slot*, and a dead slot
    refilled by inflow (M5) is a different molecule in the same slot.  The
    subsample is stable; the molecule occupying it need not be.
    """
    if max_points >= N:
        return np.arange(N, dtype=np.int64)
    if method == "stride":
        stride = max(1, math.ceil(N / max_points))
        return np.arange(0, N, stride, dtype=np.int64)[:max_points]
    if method == "hash":
        import jax

        if key is None:
            key = jax.random.key(0)
        u = np.asarray(jax.random.uniform(key, (N,), dtype=np.float64))
        idx = np.argpartition(u, max_points)[:max_points]
        return np.sort(idx).astype(np.int64)
    raise ValueError(f"unknown subsample method {method!r}; use 'stride' or 'hash'")


def write_vtp(
    path: str,
    x: Any,
    point_data: dict[str, Any] | None = None,
    binary: bool = True,
) -> str:
    """Write particles as VTK ``.vtp`` polydata (points rendered as vertices).

    Args:
        path: output filename; ``.vtp`` is appended if missing.
        x: ``f64[n, D]`` positions.  ``D < 3`` is zero-padded to 3, because VTK
            points are always 3D.  Subsample *before* calling this, with
            :func:`stable_subsample_indices`.
        point_data: name -> ``[n]`` or ``[n, k]`` per-particle arrays.  Give it
            at least ``v``, ``species`` and ``alive``; ``bool`` is written as
            ``UInt8``.
        binary: inline base64 (default) or ascii.

    Returns:
        The path written.

    The ``<Verts>`` block is not optional: without it ParaView loads the points
    but renders nothing.
    """
    if not path.endswith(".vtp"):
        path += ".vtp"
    pts = np.asarray(x, dtype=np.float64)
    n_pts = pts.shape[0]
    if pts.shape[1] < 3:
        pts = np.pad(pts, ((0, 0), (0, 3 - pts.shape[1])))

    lines = [
        _XML_HEADER,
        '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
        "  <PolyData>",
        f'    <Piece NumberOfPoints="{n_pts}" NumberOfVerts="{n_pts}" NumberOfLines="0" '
        'NumberOfStrips="0" NumberOfPolys="0">',
        "      <Points>",
        _data_array("Points", pts, 3, binary),
        "      </Points>",
        "      <Verts>",
        _data_array("connectivity", np.arange(n_pts, dtype=np.int64), 1, binary),
        _data_array("offsets", np.arange(1, n_pts + 1, dtype=np.int64), 1, binary),
        "      </Verts>",
        "      <PointData>",
    ]
    for name, raw in (point_data or {}).items():
        arr = np.asarray(raw)
        if arr.shape[0] != n_pts:
            raise ValueError(f"point field {name!r} has {arr.shape[0]} rows, expected {n_pts}")
        arr = arr.reshape(n_pts, -1)
        lines.append(_data_array(name, arr, arr.shape[1], binary))
    lines += ["      </PointData>", "    </Piece>", "  </PolyData>", "</VTKFile>"]
    return _write_lines(path, lines)


def write_vtp_surface(
    path: str,
    points: Any,
    elements: Any,
    cell_data: dict[str, Any] | None = None,
    binary: bool = True,
) -> str:
    """Write an embedded surface (2-D lines / 3-D triangles) as VTK ``.vtp`` polydata.

    Args:
        path: output filename; ``.vtp`` is appended if missing.
        points: ``f64[P, D]`` (``D < 3`` zero-padded).
        elements: ``i32[E, D]`` vertex indices -- lines in 2-D, triangles in 3-D.
        cell_data: name -> ``[E]`` or ``[E, k]`` per-element arrays (pressure, shear,
            heat flux, hit counts).

    Lines go in the ``<Lines>`` block and triangles in ``<Polys>``, so ParaView renders
    the surface itself rather than its vertices.
    """
    if not path.endswith(".vtp"):
        path += ".vtp"
    pts = np.asarray(points, dtype=np.float64)
    el = np.asarray(elements, dtype=np.int64)
    n_pts = pts.shape[0]
    n_el, d = el.shape
    if pts.shape[1] < 3:
        pts = np.pad(pts, ((0, 0), (0, 3 - pts.shape[1])))
    block = "Lines" if d == 2 else "Polys"
    counts = {"NumberOfVerts": 0, "NumberOfLines": 0, "NumberOfStrips": 0, "NumberOfPolys": 0}
    counts["NumberOfLines" if d == 2 else "NumberOfPolys"] = n_el
    piece = " ".join(f'{k}="{v}"' for k, v in counts.items())
    lines = [
        _XML_HEADER,
        '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
        "  <PolyData>",
        f'    <Piece NumberOfPoints="{n_pts}" {piece}>',
        "      <Points>",
        _data_array("Points", pts, 3, binary),
        "      </Points>",
        f"      <{block}>",
        _data_array("connectivity", el.ravel(), 1, binary),
        _data_array("offsets", np.arange(1, n_el + 1, dtype=np.int64) * d, 1, binary),
        f"      </{block}>",
        "      <CellData>",
    ]
    for name, raw in (cell_data or {}).items():
        arr = np.asarray(raw)
        if arr.shape[0] != n_el:
            raise ValueError(f"cell field {name!r} has {arr.shape[0]} rows, expected {n_el}")
        arr = arr.reshape(n_el, -1)
        lines.append(_data_array(name, arr, arr.shape[1], binary))
    lines += ["      </CellData>", "    </Piece>", "  </PolyData>", "</VTKFile>"]
    return _write_lines(path, lines)


def write_vtu_flow(  # noqa: PLR0915 -- one writer, kept whole
    path: str,
    lo: Any,
    hi: Any,
    n: tuple[int, ...],
    mask: np.ndarray,
    fields: dict[str, Any],
    binary: bool = True,
) -> str:
    """Write the **flow region** of a cut-cell grid as a VTK ``.vtu`` of sub-cell boxes.

    ``mask`` is ``bool[C, sub^D]`` (:func:`jax_dsmc.mesh.cut.flow_mask`): every True
    sub-cell becomes one quad (2-D) / hexahedron (3-D) carrying its parent cell's field
    values.  The body is thereby carved out of the picture at ``h / sub`` resolution while
    every value stays exactly the per-cell value of the ``.vtr`` -- this file changes how
    the cells are drawn, not what they contain.
    """
    if not path.endswith(".vtu"):
        path += ".vtu"
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    D = len(n)
    C = int(np.prod(n))
    sub = round(mask.shape[1] ** (1.0 / D))
    if sub**D != mask.shape[1]:
        raise ValueError(f"mask width {mask.shape[1]} is not sub^{D}")
    dx = (hi - lo) / np.asarray(n, dtype=float)
    cell_idx = np.stack(np.unravel_index(np.arange(C), n), axis=1)  # (C, D), row-major
    sub_idx = np.stack(np.indices((sub,) * D), axis=-1).reshape(-1, D)  # last axis fastest
    # cells entirely in the flow stay ONE box; only partially-masked cells are subdivided
    # (a 20^3 grid subdivided everywhere at sub = 4 is 100 MB per frame)
    whole = mask.all(axis=1)
    part = ~whole & mask.any(axis=1)
    cells_p, subs_p = np.nonzero(mask & part[:, None])
    corner = np.concatenate(
        [
            lo + cell_idx[whole] * dx,
            lo + (cell_idx[cells_p] * sub + sub_idx[subs_p]) * (dx / sub),
        ]
    )
    size = np.concatenate(
        [np.broadcast_to(dx, (int(whole.sum()), D)), np.broadcast_to(dx / sub, (cells_p.size, D))]
    )
    cells = np.concatenate([np.nonzero(whole)[0], cells_p])
    M = corner.shape[0]
    if D == 2:
        offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        vtk_type, npts = 9, 4  # VTK_QUAD
    elif D == 3:
        offsets = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=float,
        )
        vtk_type, npts = 12, 8  # VTK_HEXAHEDRON
    else:
        raise NotImplementedError("write_vtu_flow needs D = 2 or 3")
    pts = corner[:, None, :] + offsets[None, :, :] * size[:, None, :]  # (M, npts, D)
    # share the corner points between neighbouring boxes (8x fewer points in 3-D): snap to
    # the sub-grid lattice so equal corners compare equal
    lattice = np.rint((pts.reshape(-1, D) - lo) / (dx / sub)).astype(np.int64)
    uniq, conn = np.unique(lattice, axis=0, return_inverse=True)
    pts = lo + uniq * (dx / sub)
    if D < 3:
        pts = np.pad(pts, ((0, 0), (0, 3 - D)))
    lines = [
        _XML_HEADER,
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{pts.shape[0]}" NumberOfCells="{M}">',
        "      <Points>",
        _data_array("Points", pts, 3, binary),
        "      </Points>",
        "      <Cells>",
        _data_array("connectivity", conn.reshape(-1).astype(np.int64), 1, binary),
        _data_array("offsets", np.arange(1, M + 1, dtype=np.int64) * npts, 1, binary),
        _data_array("types", np.full(M, vtk_type, dtype=np.uint8), 1, binary),
        "      </Cells>",
        "      <CellData>",
    ]
    for name, raw in fields.items():
        arr = np.asarray(raw)
        if arr.shape[0] != C:
            raise ValueError(f"field {name!r} has {arr.shape[0]} rows, expected C = {C}")
        arr = arr.reshape(C, -1)[cells]
        lines.append(_data_array(name, arr, arr.shape[1], binary))
    lines += ["      </CellData>", "    </Piece>", "  </UnstructuredGrid>", "</VTKFile>"]
    return _write_lines(path, lines)


# --- collections: what makes ParaView animate ---------------------------------


def write_pvd(path: str, entries: list[tuple[float, str]]) -> str:
    """Write a ``.pvd`` collection binding a file series to simulation times.

    Args:
        path: output filename; ``.pvd`` is appended if missing.
        entries: ``(time, filename)`` pairs.  Filenames are stored **relative to
            the .pvd**, which is what makes the output directory movable.

    Returns:
        The path written.

    Without this file ParaView shows a pile of unrelated datasets and the time
    slider does nothing.  It is the cheapest file in the tree and the one that
    makes the rest usable.
    """
    if not path.endswith(".pvd"):
        path += ".pvd"
    base = os.path.dirname(os.path.abspath(path))
    lines = [
        _XML_HEADER,
        '<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for t, fname in entries:
        rel = os.path.relpath(os.path.abspath(fname), base)
        lines.append(f'    <DataSet timestep="{float(t)!r}" group="" part="0" file="{rel}"/>')
    lines += ["  </Collection>", "</VTKFile>"]
    return _write_lines(path, lines)


def _piece_field_decls(fields: dict[str, Any], tag: str) -> list[str]:
    """``<PDataArray>`` declarations: the master file states names and widths only."""
    out = []
    for name, raw in fields.items():
        arr = np.asarray(raw)
        ncomp = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
        _, vtk_type = _vtk_dtype(arr)
        out.append(
            f'      <PDataArray {tag}type="{vtk_type}" Name="{name}" NumberOfComponents="{ncomp}"/>'
        )
    return out


def write_pvtr(
    path: str,
    lo: Any,
    hi: Any,
    n: tuple[int, ...],
    fields: dict[str, Any],
    pieces: list[str],
    piece_extents: list[str] | None = None,
) -> str:
    """Master file for a sharded ``.vtr`` dump.  **Stub — untested against a real run.**

    Sharded output does not exist yet (M1-M3 are single-process), so this is
    specified rather than exercised: the shape is here so that adding ranks is a
    change to the caller rather than to this module.  ``piece_extents`` must give
    each piece's ``Extent`` in point indices; if omitted, every piece is declared
    over the whole extent, which is only correct for a single piece.

    Args:
        path: output filename; ``.pvtr`` is appended if missing.
        lo, hi, n: as :func:`write_vtr`.
        fields: used only for names, dtypes and component counts.
        pieces: the per-rank ``.vtr`` filenames.
        piece_extents: per-piece ``Extent`` strings.
    """
    if not path.endswith(".pvtr"):
        path += ".pvtr"
    n = tuple(int(k) for k in n)
    axes = _cell_axes(
        np.asarray(lo, dtype=np.float64).reshape(-1),
        np.asarray(hi, dtype=np.float64).reshape(-1),
        n,
    )
    ext = " ".join(f"0 {len(a) - 1}" for a in axes)
    base = os.path.dirname(os.path.abspath(path))
    lines = [
        _XML_HEADER,
        '<VTKFile type="PRectilinearGrid" version="1.0" byte_order="LittleEndian">',
        f'  <PRectilinearGrid WholeExtent="{ext}" GhostLevel="0">',
        "    <PCoordinates>",
        '      <PDataArray type="Float64" Name="X_COORDINATES"/>',
        '      <PDataArray type="Float64" Name="Y_COORDINATES"/>',
        '      <PDataArray type="Float64" Name="Z_COORDINATES"/>',
        "    </PCoordinates>",
        "    <PCellData>",
        *_piece_field_decls(fields, ""),
        "    </PCellData>",
    ]
    extents = piece_extents or [ext] * len(pieces)
    for piece, pext in zip(pieces, extents, strict=True):
        rel = os.path.relpath(os.path.abspath(piece), base)
        lines.append(f'    <Piece Extent="{pext}" Source="{rel}"/>')
    lines += ["  </PRectilinearGrid>", "</VTKFile>"]
    return _write_lines(path, lines)


def write_pvtp(path: str, point_data: dict[str, Any], pieces: list[str]) -> str:
    """Master file for a sharded ``.vtp`` dump.  **Stub — untested against a real run.**

    Same status and rationale as :func:`write_pvtr`.
    """
    if not path.endswith(".pvtp"):
        path += ".pvtp"
    base = os.path.dirname(os.path.abspath(path))
    lines = [
        _XML_HEADER,
        '<VTKFile type="PPolyData" version="1.0" byte_order="LittleEndian">',
        '  <PPolyData GhostLevel="0">',
        "    <PPoints>",
        '      <PDataArray type="Float64" Name="Points" NumberOfComponents="3"/>',
        "    </PPoints>",
        "    <PPointData>",
        *_piece_field_decls(point_data, ""),
        "    </PPointData>",
    ]
    for piece in pieces:
        rel = os.path.relpath(os.path.abspath(piece), base)
        lines.append(f'    <Piece Source="{rel}"/>')
    lines += ["  </PPolyData>", "</VTKFile>"]
    return _write_lines(path, lines)


def moment_fields(m: Any) -> dict[str, np.ndarray]:
    """Flatten a ``CellMoments`` into the name -> array dict the writers want."""
    return {
        "nrho": np.asarray(m.nrho),
        "u": np.asarray(m.u),
        "temp": np.asarray(m.temp),
        "P": np.asarray(m.P),
        "q": np.asarray(m.q),
        "count": np.asarray(m.count),
    }


def particle_fields(state: Any, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Per-particle arrays for :func:`write_vtp`, restricted to ``idx``."""
    return {
        "v": np.asarray(state.v)[idx],
        "species": np.asarray(state.species)[idx],
        "alive": np.asarray(state.alive)[idx].astype(np.uint8),
        "slot": idx.astype(np.int64),
    }


class VTKSeries:
    """A ParaView-animatable output directory: grids, particles and collections.

    Usage, host-side, outside the jitted step::

        series = VTKSeries("out/bkw")
        for step, t in enumerate(times):
            series.add_fields(step, t, grid.lo, grid.hi, grid.n, moment_fields(m))
            series.add_particles(step, t, state)
        series.close()  # writes fields.pvd and particles.pvd

    ``close`` is what makes the run animate; call it even on an aborted run, or
    the frames already written stay a pile of unrelated files.  It is safe to
    call repeatedly, so calling it every N steps keeps a live run viewable.
    """

    def __init__(
        self,
        directory: str,
        max_particles: int = 20_000,
        subsample: str = "stride",
        key: Any = None,
        binary: bool = True,
    ) -> None:
        """
        Args:
            directory: created if absent; all files land in it.
            max_particles: cap on particles per frame.  See
                :func:`stable_subsample_indices` for why the selection is
                index-derived and frame-stable.
            subsample: ``"stride"`` or ``"hash"``.
            key: fixed PRNG key for ``"hash"``.  Fixed for the whole series.
            binary: inline base64 (default) or ascii DataArrays.
        """
        self.directory = directory
        self.max_particles = int(max_particles)
        self.subsample = subsample
        self.key = key
        self.binary = binary
        self.field_entries: list[tuple[float, str]] = []
        self.particle_entries: list[tuple[float, str]] = []
        self._idx: np.ndarray | None = None
        self._idx_n: int = -1
        os.makedirs(directory, exist_ok=True)

    def indices_for(self, N: int) -> np.ndarray:
        """The frame-stable subsample for capacity ``N``, computed once and cached."""
        if self._idx is None or self._idx_n != N:
            self._idx = stable_subsample_indices(N, self.max_particles, self.key, self.subsample)
            self._idx_n = N
        return self._idx

    def add_fields(
        self,
        step: int,
        time: float,
        lo: Any,
        hi: Any,
        n: tuple[int, ...],
        fields: dict[str, Any],
    ) -> str:
        """Write one ``step_%05d.vtr`` frame and record it in the collection."""
        path = os.path.join(self.directory, f"step_{int(step):05d}.vtr")
        write_vtr(path, lo, hi, n, fields, binary=self.binary)
        self.field_entries.append((float(time), path))
        return path

    def add_particles(self, step: int, time: float, state: Any) -> str:
        """Write one ``part_%05d.vtp`` frame from a subsample of ``state``."""
        x = np.asarray(state.x)
        idx = self.indices_for(x.shape[0])
        path = os.path.join(self.directory, f"part_{int(step):05d}.vtp")
        write_vtp(path, x[idx], particle_fields(state, idx), binary=self.binary)
        self.particle_entries.append((float(time), path))
        return path

    def close(self) -> tuple[str | None, str | None]:
        """Write ``fields.pvd`` and ``particles.pvd``.  Idempotent."""
        fields_pvd = particles_pvd = None
        if self.field_entries:
            fields_pvd = write_pvd(os.path.join(self.directory, "fields.pvd"), self.field_entries)
        if self.particle_entries:
            particles_pvd = write_pvd(
                os.path.join(self.directory, "particles.pvd"), self.particle_entries
            )
        return fields_pvd, particles_pvd


# --- checkpoints --------------------------------------------------------------


def _key_to_uint32(key: Any) -> np.ndarray:
    """Serialise a PRNG key (typed or raw) as ``uint32[...]``."""
    import jax

    try:
        return np.asarray(jax.random.key_data(key))
    except (TypeError, ValueError):
        # already a raw uint32 key array
        return np.asarray(key, dtype=np.uint32)


def key_from_uint32(data: Any, impl: str = "threefry2x32") -> Any:
    """Rebuild a typed PRNG key from the ``uint32`` data a checkpoint stored."""
    import jax

    return jax.random.wrap_key_data(np.asarray(data, dtype=np.uint32), impl=impl)


def _checkpoint_arrays(
    state: Any,
    moments: Any,
    base_key: Any,
    step: int | None,
    extra: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Assemble the flat name -> array dict both backends write."""
    out: dict[str, np.ndarray] = {}
    if state is not None:
        out["x"] = np.asarray(state.x)
        out["v"] = np.asarray(state.v)
        out["species"] = np.asarray(state.species)
        out["alive"] = np.asarray(state.alive)
        out["logh_acc"] = np.asarray(state.logh_acc)
    if moments is not None:
        for name, arr in moment_fields(moments).items():
            out["moments/" + name] = arr
    if base_key is not None:
        out["base_key"] = _key_to_uint32(base_key)
    if step is not None:
        out["step"] = np.asarray(int(step), dtype=np.int64)
    for name, value in (extra or {}).items():
        out[name] = np.asarray(value)
    return out


def save_checkpoint(
    path: str,
    state: Any = None,
    moments: Any = None,
    base_key: Any = None,
    step: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Write a restartable checkpoint.  HDF5 if ``path`` ends ``.h5``/``.hdf5``, else npz.

    Stores ``x``, ``v``, ``species``, ``alive``, ``logh_acc``, the per-cell
    moments, the RNG **base key** and the step index.

    The base key and step are what make a restart resumable at all: section 1's
    contract reconstructs every draw from ``(base_key, step, STAGE_ID)`` by
    ``fold_in``, so saving particle state alone gives a run that continues with
    the wrong random numbers — silently, and with no way to tell from the output.

    Host-side only.  Arrays are pulled to numpy, which forces any outstanding
    computation; call after ``jax.block_until_ready`` if you are timing.
    """
    arrays = _checkpoint_arrays(state, moments, base_key, step, extra)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    if path.endswith((".h5", ".hdf5")):
        import h5py

        with h5py.File(path, "w") as fh:
            for name, arr in arrays.items():
                fh.create_dataset(name, data=arr)
        return path

    if not path.endswith(".npz"):
        path += ".npz"
    np.savez_compressed(path, **{k.replace("/", "__"): v for k, v in arrays.items()})
    return path


def load_checkpoint(path: str) -> dict[str, np.ndarray]:
    """Read a checkpoint back as a flat ``dict`` of numpy arrays.

    Keys match what :func:`save_checkpoint` wrote, with ``moments/<name>``
    preserved for both backends.  Rebuild the PRNG key with
    :func:`key_from_uint32`.  The state containers are left to the caller, so
    this module keeps no dependency on ``types.py``.
    """
    if path.endswith((".h5", ".hdf5")):
        import h5py

        out: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as fh:
            fh.visititems(lambda name, obj: _collect_h5(out, name, obj))
        return out

    with np.load(path, allow_pickle=False) as data:
        return {k.replace("__", "/"): data[k] for k in data.files}


def _collect_h5(out: dict[str, np.ndarray], name: str, obj: Any) -> None:
    """``h5py`` visitor: copy every dataset into ``out``."""
    import h5py

    if isinstance(obj, h5py.Dataset):
        out[name] = obj[()]


# --- the run summary ----------------------------------------------------------


def _fmt_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.4e}" for x in np.asarray(v).reshape(-1)) + "]"


def summary(
    moments: Any = None,
    diag: Any = None,
    state: Any = None,
    step: int | None = None,
    ncollide: Any = None,
    nattempt: Any = None,
    A_max: int | None = None,
    max_cells: int = 8,
    stream: TextIO | None = None,
) -> str:
    """Print — and return — a human-readable report of one step (spec 10.2, path 4).

    Covers the per-cell moments, the collision bookkeeping, and the counters
    that make a silent failure audible:

    * ``attempts_clipped`` — cells where the required attempt count exceeded
      ``A_max``.  Nonzero means the run **under-collided with no error**
      (section 8.2, gap U4).
    * ``majorant_violations`` — candidates with ``q > Sigma``.  Each has a zero
      score and therefore a **biased gradient** (section 7.2.1, gap U3).
    * ``min_majorant_margin`` — ``min(Sigma/q)``; target ``>= 2`` (section
      7.2.2).  A margin near 1 is how a violation count of zero becomes nonzero
      on the next seed.

    Args:
        moments: a ``CellMoments`` or ``None``.
        diag: a ``Diagnostics`` or ``None``.
        state: a ``ParticleState`` or ``None``; only ``alive`` is read.
        step: step index, printed in the header.
        ncollide: accepted collisions this step.
        nattempt: candidate pairs tested this step.
        A_max: the static per-cell attempt bound, for context on clipping.
        max_cells: how many cells to print before eliding.
        stream: destination; defaults to ``sys.stdout``.  Pass ``io.StringIO()``
            to capture.

    Returns:
        The report as a string.

    Host-side only: it formats concrete values and would trace into nothing
    useful inside ``jit``.
    """
    out: list[str] = []
    out.append("=== jax-dsmc" + (f" step {int(step)}" if step is not None else "") + " ===")

    if state is not None:
        alive = np.asarray(state.alive)
        out.append(f"particles : {int(alive.sum())} alive / {alive.size} capacity")

    if moments is not None:
        out.extend(_moment_lines(moments, max_cells))

    if ncollide is not None or nattempt is not None:
        nc = int(np.asarray(ncollide).sum()) if ncollide is not None else -1
        na = int(np.asarray(nattempt).sum()) if nattempt is not None else -1
        rate = f"{nc / na:.4f}" if na > 0 else "n/a"
        out.append(f"collisions: ncollide {nc}  nattempt {na}  accept_rate {rate}")

    if diag is not None:
        out.extend(_diagnostic_lines(diag, A_max))

    text = "\n".join(out)
    print(text, file=stream if stream is not None else sys.stdout)
    return text


def _moment_lines(m: Any, max_cells: int) -> list[str]:
    """The per-cell moment table."""
    nrho = np.asarray(m.nrho)
    u = np.asarray(m.u)
    temp = np.asarray(m.temp)
    q = np.asarray(m.q)
    cnt = np.asarray(m.count)
    C = nrho.shape[0]
    out = [
        f"cells     : {C}",
        f"{'cell':>5}  {'count':>8}  {'nrho':>12}  {'temp':>10}  {'|u|':>12}  {'|q|':>12}",
    ]
    shown = min(C, max_cells)
    for i in range(shown):
        out.append(
            f"{i:>5}  {cnt[i]:>8.1f}  {nrho[i]:>12.5e}  {temp[i]:>10.4f}  "
            f"{np.linalg.norm(u[i]):>12.5e}  {np.linalg.norm(q[i]):>12.5e}"
        )
    if shown < C:
        out.append(f"{'...':>5}  ({C - shown} more cells)")
    live = cnt > 0
    if live.any():
        out.append(
            f"totals    : nrho {nrho[live].mean():.5e}  temp {temp[live].mean():.4f}  "
            f"u_mean {_fmt_vec(u[live].mean(axis=0))}"
        )
    return out


def _diagnostic_lines(diag: Any, A_max: int | None) -> list[str]:
    """The silent-failure counters, with the warning text spelled out."""
    clipped = int(np.asarray(diag.attempts_clipped))
    maxatt = int(np.asarray(diag.max_attempts))
    viol = int(np.asarray(diag.majorant_violations))
    margin = float(np.asarray(diag.min_majorant_margin))
    cap = f" (A_max {A_max})" if A_max is not None else ""
    out = [f"attempts  : max_attempts {maxatt}{cap}  attempts_clipped {clipped}"]
    if clipped > 0:
        out.append(
            f"  !! CLIPPED: n > A_max in {clipped} cell(s) -- silent under-collision (8.2, gap U4)"
        )
    out.append(f"majorant  : violations {viol}  min_margin {margin:.4f}")
    if viol > 0:
        out.append(f"  !! {viol} candidate(s) had q > Sigma: zero score, BIASED gradient (7.2.1)")
    elif 0.0 < margin < 2.0:
        out.append(
            f"  !  min(Sigma/q) = {margin:.3f} < 2 target (7.2.2) -- violations are one seed away"
        )
    return out
