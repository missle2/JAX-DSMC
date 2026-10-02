"""Containers and conventions shared by the embedded-surface (cut-cell) machinery.

The Cartesian grid of :mod:`jax_dsmc.grid` is kept; a body is embedded as a surface
and the cells it overlaps are *cut* (SPARTA's design: ``cut2d.cpp``/``cut3d.cpp``,
``grid_surf.cpp``, ``update.cpp``).  Everything downstream of ``cell_of`` /
``cell_volume`` / ``sort.build_table`` is untouched; this package only adds per-cell
tables and a surface-aware ``first_hit``.

Conventions (fixed here, relied on everywhere in ``jax_dsmc.mesh``):

* **Orientation.** Element normals point **into the flow** (SPARTA: 2-D
  ``z x (p2 - p1)``, 3-D ``(p2 - p1) x (p3 - p1)``).  A particle can only hit an
  element from the flow side, ``v . n < 0``.
* **Surface ids.** The tracker numbers surfaces ``0 .. 2D-1`` for the box faces
  (``2*axis + side``, as :class:`jax_dsmc.boundary.Faces`) and ``2D + e`` for element
  ``e``; an analytic body or a level set is the single id ``2D``.
* **Per-cell layout.** Every table is indexed by the flat cell id of ``grid.cell_of``
  and padded to a static width (``-1`` = empty slot).  Nothing here does ``grid.n``
  arithmetic beyond ``cell_of`` / ``cell_bounds``, so a hierarchical grid later only
  has to replace those two.
* **Cell types.** ``OUTSIDE`` (all flow), ``INSIDE`` (all body, volume 0), ``CUT``.
* **Statics.** Array *shapes* (``E``, ``S_max``, ``S_cell``) are static by
  construction; the only Python statics are ``radius`` (stencil dilation in cells)
  and ``kind``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from jax_dsmc.state import static_field

OUTSIDE, INSIDE, CUT = 0, 1, 2
"""Cell classification values of ``Mesh.cell_type`` (SPARTA ``cinfo[].type``)."""

SPECULAR, DIFFUSE = 0, 1
"""Surface-model kinds (``Mesh.model_kind``); DIFFUSE with ``acc = 0`` is specular,
exactly as SPARTA's ``surf_collide diffuse T 0.0``."""


@register_dataclass
@dataclass(frozen=True)
class Surface:
    """A polygonal surface: ``P`` points, ``E`` elements (lines in 2-D, triangles in 3-D).

    Fields:
        points: ``f64[P, D]`` — **traced**; the only thing a shape deformation moves.
        elements: ``i32[E, D]`` — vertex indices (0-based); fixed connectivity.
        model: ``i32[E]`` — index of the surface model (row of the extended
            ``WallParams``, see ``jax_dsmc.mesh.geometry``) for each element.
        group: ``i32[E]`` — tally group of each element (SPARTA ``surf group``).
    """

    points: jax.Array
    elements: jax.Array
    model: jax.Array
    group: jax.Array

    @property
    def dim(self) -> int:
        return int(self.points.shape[1])

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])


@register_dataclass
@dataclass(frozen=True)
class AnalyticBody:
    """A circle (2-D) or sphere (3-D): exact hits and normals for the adjoint reference.

    Fields:
        centre: ``f64[D]`` **traced**.
        radius: ``f64`` scalar **traced**.
    Its surface id is ``2D`` and it belongs to model 0 / group 0.  ``grazing`` (STATIC)
    is the cut-off of the hit-time derivative's ``1 / (v . n)`` (see
    :func:`jax_dsmc.mesh.analytic.hit_body`) and of the diffuse normal score's
    ``1 / (v_out . n)`` (:func:`jax_dsmc.shape.normal_score`); ``0`` = exact.
    """

    centre: jax.Array
    radius: jax.Array
    grazing: float = static_field(default=0.0)


@register_dataclass
@dataclass(frozen=True)
class LevelSet:
    """A body given implicitly as ``phi(x) < 0`` with ``phi`` a uniform cubic B-spline.

    The shape representation of the optimiser (:mod:`jax_dsmc.mesh.levelset`): ``phi``
    is C2, so its normal ``grad phi / |grad phi|`` is C1 and the hit normal varies
    smoothly with the coefficients -- the pathwise shape derivative of the replayed wall
    map is then complete for specular walls (no element-assignment jumps).

    Fields:
        coef: ``f64[m_1, ..., m_D]`` **traced** -- the design variables.  Coefficient
            ``j`` is centred at ``lo + j h``.
        lo: ``f64[D]`` origin of the coefficient grid (traced array, never a control).
        h: STATIC knot spacing (one value for every axis).
        n_sample: STATIC number of ray samples of the hit search; the search covers
            ``reach = n_sample * h / 2`` at spacing ``h / 2``.
        n_bisect: STATIC cap on the refinement iterations (safeguarded Newton) of the bracket.
        grazing: STATIC cut-off ``eps`` of the hit-time derivative: its ``1 / (v . grad
            phi)`` is evaluated with ``|v . grad phi| >= eps |v| |grad phi|`` (the primal
            hit is exact; ``0`` = exact derivative, heavy-tailed at grazing incidence).
            The same ``eps`` floors the diffuse normal score's ``1 / (v_out . n)`` at
            ``eps |v_out|`` (:func:`jax_dsmc.shape.normal_score`).
    Its surface id is ``2D`` and it belongs to model 0 / group 0, like the analytic body.
    """

    coef: jax.Array
    lo: jax.Array
    h: float = static_field()
    n_sample: int = static_field(default=8)
    n_bisect: int = static_field(default=40)
    grazing: float = static_field(default=0.0)

    @property
    def dim(self) -> int:
        return int(self.coef.ndim)

    @property
    def reach(self) -> float:
        """Length along the ray that the hit search covers."""
        return self.n_sample * self.h / 2.0


@register_dataclass
@dataclass(frozen=True)
class Mesh:
    """Per-cell cut-cell tables for one grid and one body.

    Fields (``C = prod(grid.n)`` leaf cells; ``-1`` pads every id table):
        body: :class:`Surface`, :class:`AnalyticBody` or :class:`LevelSet`.
        cell_type: ``i32[C]`` — ``OUTSIDE`` / ``INSIDE`` / ``CUT``.
        volume: ``f64[C]`` — flow volume of every cell (full, zero, or the cut
            fraction times the full volume; includes ``grid.transverse``).
        candidates: ``i32[C, S_max]`` — elements whose bounding box overlaps the cell
            dilated by ``radius`` cells (the stencil tracker's list).
        csurfs: ``i32[C, S_cell]`` — elements overlapping the cell itself
            (``radius = 0``; the cell-walk tracker's list, SPARTA ``csurfs``).
        outside_corner: ``i32[C]`` — index (``0 .. 2^D - 1``, x fastest) of a corner of
            the cell known to lie in the flow, ``-1`` for INSIDE cells; the reference
            point of the inside/outside test (SPARTA ``point_outside_surfs``).
        normals: ``f64[E, D]`` (unit, into the flow) and ``areas: f64[E]`` (length in
            2-D, area in 3-D) of the elements; ``(1, D)`` / ``(1,)`` for an analytic
            body.
        radius: STATIC int — stencil dilation in cells.
        kind: STATIC ``"surface" | "circle" | "sphere" | "levelset"``.
        n_sub, split_of, split_table, sub_parent: the split cells' sub-cells (SPARTA
            ``nsplit > 1``), as the :class:`jax_dsmc.state.GridSpec` fields of the same
            names (:func:`jax_dsmc.mesh.cut.with_volume` copies them); ``volume`` then
            has ``C + n_sub`` entries, the sub-cells' after the Cartesian cells'.
    """

    body: Surface | AnalyticBody | LevelSet
    cell_type: jax.Array
    volume: jax.Array
    candidates: jax.Array
    csurfs: jax.Array
    outside_corner: jax.Array
    normals: jax.Array
    areas: jax.Array
    radius: int = static_field(default=1)
    kind: str = static_field(default="surface")
    n_sub: int = static_field(default=0)
    split_of: jax.Array = dataclasses.field(default_factory=lambda: jnp.zeros(0, jnp.int32))
    split_table: jax.Array = dataclasses.field(default_factory=lambda: jnp.zeros((0, 1), jnp.int32))
    sub_parent: jax.Array = dataclasses.field(default_factory=lambda: jnp.zeros(0, jnp.int32))

    @property
    def n_elements(self) -> int:
        return int(self.normals.shape[0])
