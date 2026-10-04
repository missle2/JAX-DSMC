"""Where the mesh examples' figures go: ``examples/mesh/figures/<folder>/<name>``, one folder
per topic (the layout is listed in ``figures/README.md``).  Every script that writes a
figure asks :func:`figure_path` for its path, so the layout lives in one place.

Folders: ``performance`` (speed and profiles), ``<case>_forward`` (a forward case's fields,
fill, surface loads, mesh -- ``circle_forward``, ``sphere_forward``, ...), ``meshing`` (cut-cell
and level-set checks), ``optc`` (shape optimisation runs and their convergence), ``squash``
(the aspect-ratio optimisation), ``movie`` (optimiser movies and their frames); the
adjoint gate figures stay at the top level (folder ``""``).
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
