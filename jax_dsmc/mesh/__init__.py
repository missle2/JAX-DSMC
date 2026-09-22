"""Embedded surfaces on the Cartesian grid: SPARTA-style cut cells and surface tracking.

Modules:
    types      containers and conventions (read its docstring first)
    surface    polygonal surfaces: SPARTA file reader, generators, normals, checks
    analytic   circle / sphere bodies with exact hits and normals
    levelset   C1 level-set bodies (cubic B-spline): the optimiser's shape representation
    cut        host (numpy) cell classification, stencils, flow volumes, split check
    geometry   ``first_hit`` over box faces and elements (JAX, jitted)
    walk       the SPARTA cell-by-cell tracker (robust reference)
"""

from jax_dsmc.mesh.types import (
    CUT,
    DIFFUSE,
    INSIDE,
    OUTSIDE,
    SPECULAR,
    AnalyticBody,
    LevelSet,
    Mesh,
    Surface,
)
