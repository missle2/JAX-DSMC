"""Where the mesh examples' figures go: ``examples/mesh/figures/<folder>/<name>``, one folder
per topic (the layout is listed in ``figures/README.md``).  Every script that writes a
figure asks :func:`figure_path` for its path, so the layout lives in one place.

Folders: ``performance`` (speed and profiles), ``<case>_forward`` (a forward case's fields,
fill, surface loads, mesh -- ``circle_forward``, ``sphere_forward``, ...), ``meshing`` (cut-cell
and level-set checks), ``optc`` (shape optimisation runs and their convergence), ``squash``
(the aspect-ratio optimisation), ``movie`` (optimiser movies and their frames), ``gates``
(adjoint gates and estimator studies).  ``optc`` and ``movie`` have one level of sub-folders:
``optc/{optab,optax,optm,optrules,gates}`` and ``movie/{coarse,optab,squash,runs_kn0.5,
runs_kn0.5_M10,optax,optrules}`` (see :func:`movie_folder`); ``tracker_lane.png`` stays at
the top level (folder ``""``).
"""

from __future__ import annotations

import os

FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def figure_path(folder: str, name: str) -> str:
    """``figures/<folder>/<name>``, the folder created if needed (``""``: the top level)."""
    directory = os.path.join(FIGURES, folder)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name)


def forward_folder(case: str) -> str:
    """The folder of a forward case's figures: ``circle`` -> ``circle_forward``."""
    return f"{case}_forward"


def movie_folder(tag: str) -> str:
    """The ``movie/`` sub-folder of a run's movie and frame directory (``movie``: none fits)."""
    if "coarse" in tag:
        return "movie/coarse"
    if "ellipse" in tag:
        return "movie/squash"
    if "_ab_" in tag or tag.startswith(("circle_long_S4", "circle_long_M10_S4")):
        return "movie/optab"
    if tag.startswith("circle_long_kn0.5_M10"):
        return "movie/runs_kn0.5_M10"
    if tag.startswith("circle_long_kn0.5"):
        return "movie/runs_kn0.5"
    return "movie"
