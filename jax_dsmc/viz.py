"""Matplotlib figures for milestone demos (spec section 10.2, path 1).

**Headless by construction.**  The ``Agg`` backend is selected *before*
``pyplot`` is imported — after that first import the backend is fixed and a
later ``matplotlib.use`` is at best ignored — and nothing here ever calls
``show()``.  These figures run in CI and double as milestone acceptance
evidence, so they must render identically on a machine with no display.

Like everything in ``output.py``, this is **host-side and outside the jitted
step**: it takes concrete arrays, formats them, and writes files.

Colour
------
One fixed, validated palette, assigned by the *job* the colour does rather than
by series order:

* an ordered sequence of times is **magnitude**, so it gets a single-hue blue
  ordinal ramp, light (early) to dark (late) — never a set of unrelated hues;
* analytic-vs-measured is **identity**, so it gets two categorical hues (blue,
  orange) that are separable under deuteranopia and protanopia;
* the equilibrium reference is not a series at all and is drawn as a recessive
  grey dashed line.

The categorical triple was checked with the data-viz palette validator on the
all-pairs list: worst CVD Delta E 9.2, worst normal-vision Delta E 24.0.  These
figures are written as light-mode PNGs, so only the light steps are used.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")  # MUST precede the pyplot import; see the module docstring

import matplotlib.pyplot as plt
import numpy as np

# --- palette (see the module docstring for provenance) ------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d9d8d4"

SERIES_ANALYTIC = "#2a78d6"  # categorical slot 1, blue
SERIES_MEASURED = "#eb6834"  # categorical slot 2, orange
SERIES_ERROR = "#1baf7a"  # categorical slot 3, aqua
REFERENCE = "#8a8a86"  # equilibrium / guide lines: recessive, not a series

TIME_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
"""Ordinal blue ramp, early -> late.

Starts at step 250 rather than 100: an *ordinal* ramp's lightest step must still
clear 2:1 against the surface, whereas a continuous sequential ramp may recede
into it.
"""

LINE_WIDTH = 2.0
MARKER_SIZE = 6.0  # points; ~8px at the default dpi


def time_color(i: int, n: int) -> str:
    """Colour for the ``i``-th of ``n`` ordered times, from the ordinal ramp.

    Never cycles: with more times than ramp steps the ends are reused by
    interpolation of position, which keeps the ordering readable, but past ~6
    curves a panel is better split than recoloured.
    """
    if n <= 1:
        return TIME_RAMP[len(TIME_RAMP) // 2]
    pos = i * (len(TIME_RAMP) - 1) / (n - 1)
    return TIME_RAMP[round(pos)]


# --- figure plumbing ----------------------------------------------------------


def make_figure(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] | None = None):
    """A figure with the house style applied.  Returns ``(fig, axes_flat_list)``.

    ``axes`` always comes back as a flat list, whatever the layout, so callers
    index it uniformly instead of branching on the shape numpy hands back.
    """
    if figsize is None:
        figsize = (5.6 * ncols, 4.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    flat = list(np.atleast_1d(axes).ravel())
    for ax in flat:
        style_axes(ax)
    return fig, flat


def style_axes(ax) -> None:
    """Recessive grid and axes, primary ink for text.  Applied by :func:`make_figure`."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def savefig(fig, path: str, dpi: int = 150) -> str:
    """Save and close.  Never ``show()`` — there is no display in CI."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


# --- the milestone panels -----------------------------------------------------


def draw_speed_distribution(
    ax,
    vmag: Any,
    analytic: list[Any],
    labels: list[str],
    samples: list[Any] | None = None,
    equilibrium: Any = None,
    bins: int = 60,
    xlabel: str = "speed $|v| / v_{th}$",
    ylabel: str = "speed pdf  $4\\pi v^2 f$",
) -> Any:
    """Distribution at several times against the analytic solution.

    Args:
        ax: target axes.
        vmag: ``[m]`` abscissa the analytic curves are evaluated on.
        analytic: one ``[m]`` curve per time, **in time order** — the ordinal
            colour ramp encodes that order, so an unsorted list mis-states the
            physics.
        labels: one label per curve.
        samples: optional per-time particle speeds, histogrammed as a stepped
            density and drawn in the measured hue.  ``None`` entries are skipped,
            which is what the analytic-only fallback needs.
        equilibrium: optional ``[m]`` Maxwellian, drawn as a recessive guide.
        bins: histogram bin count.

    Returns:
        ``ax``.
    """
    vmag = np.asarray(vmag)
    n = len(analytic)
    if equilibrium is not None:
        ax.plot(
            vmag,
            np.asarray(equilibrium),
            color=REFERENCE,
            linestyle="--",
            linewidth=1.4,
            label="Maxwellian ($t\\to\\infty$)",
            zorder=1,
        )
    for i, (curve, label) in enumerate(zip(analytic, labels, strict=True)):
        ax.plot(
            vmag,
            np.asarray(curve),
            color=time_color(i, n),
            linewidth=LINE_WIDTH,
            label=label,
            zorder=3,
        )
    for i, sample in enumerate(samples or []):
        if sample is None:
            continue
        counts, edges = np.histogram(np.asarray(sample), bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(
            centers,
            counts,
            color=SERIES_MEASURED,
            linewidth=1.0,
            marker="o",
            markersize=MARKER_SIZE * 0.5,
            linestyle="none",
            alpha=0.75,
            label="DSMC" if i == 0 else None,
            zorder=2,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return ax


def draw_moment_relaxation(
    ax,
    t: Any,
    analytic: Any,
    measured: Any = None,
    rate: float | None = None,
    xlabel: str = "time $\\nu t$  (mean collisions per particle)",
    ylabel: str = "$1 - \\langle |v|^4\\rangle / 15 v_{th}^4$",
) -> Any:
    """The exponentially relaxing moment, on a log ordinate.

    A single y-axis, always: the analytic and measured curves are the same
    quantity in the same units, and a second scale would be a lie about the
    comparison.  The fitted rate, when given, is annotated in the corner rather
    than drawn as a third line.
    """
    t = np.asarray(t)
    ax.semilogy(
        t,
        np.asarray(analytic),
        color=SERIES_ANALYTIC,
        linewidth=LINE_WIDTH,
        label="BKW exact",
        zorder=3,
    )
    if measured is not None:
        ax.semilogy(
            t,
            np.asarray(measured),
            color=SERIES_MEASURED,
            linewidth=0.0,
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor="none",
            markeredgewidth=1.4,
            label="DSMC",
            zorder=4,
        )
    if rate is not None:
        ax.text(
            0.97,
            0.95,
            f"exact rate  $\\nu/3$ = {rate:.4g} s$^{{-1}}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return ax


def draw_error_history(
    ax,
    t: Any,
    err: Any,
    label: str = "$L_2$ error vs. analytic",
    xlabel: str = "time $\\nu t$  (mean collisions per particle)",
    ylabel: str = "$\\|f_{DSMC} - f_{BKW}\\|_2 / \\|f_{BKW}\\|_2$",
    note: str | None = None,
) -> Any:
    """Relative ``L2`` error of the distribution against the analytic solution."""
    t = np.asarray(t)
    ax.semilogy(
        t,
        np.asarray(err),
        color=SERIES_ERROR,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE * 0.7,
        label=label,
        zorder=3,
    )
    if note:
        ax.text(
            0.03,
            0.05,
            note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def draw_empty_panel(ax, message: str) -> Any:
    """A panel that says why it is empty, instead of a blank box or a crash.

    Used when the solver modules are not importable yet: the analytic half of
    the milestone figure still renders, and this states plainly what is missing.
    """
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color=INK_SECONDARY,
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return ax


def draw_cell_profile(
    ax,
    centers: Any,
    values: Any,
    label: str,
    reference: Any = None,
    reference_label: str = "analytic",
    xlabel: str = "$x$",
    ylabel: str = "",
) -> Any:
    """A per-cell moment profile along one axis, optionally against a reference.

    The M3/M4 Couette-vs-Chapman-Enskog panel; here so that ``sample.py``'s
    output has a plotting path from the start.
    """
    centers = np.asarray(centers).reshape(-1)
    if reference is not None:
        ax.plot(
            centers,
            np.asarray(reference),
            color=SERIES_ANALYTIC,
            linewidth=LINE_WIDTH,
            label=reference_label,
            zorder=2,
        )
    ax.plot(
        centers,
        np.asarray(values),
        color=SERIES_MEASURED,
        linewidth=0.0,
        marker="o",
        markersize=MARKER_SIZE,
        markerfacecolor="none",
        markeredgewidth=1.4,
        label=label,
        zorder=3,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return ax


def draw_fd_vs_ad(
    ax,
    ad: Any,
    fd: Any,
    xlabel: str = "finite difference $\\partial J/\\partial \\theta$",
    ylabel: str = "adjoint (jax.grad) $\\partial J/\\partial \\theta$",
) -> Any:
    """Gradient validation: AD against central finite differences.

    The identity line is the reference, drawn recessively; the points are the
    measurement.  Spec 13.4 requires two FD modes — plot them as two calls, not
    as two y-scales.
    """
    ad = np.asarray(ad).reshape(-1)
    fd = np.asarray(fd).reshape(-1)
    lim = float(max(np.max(np.abs(ad)), np.max(np.abs(fd)), 1e-300))
    ax.plot([-lim, lim], [-lim, lim], color=REFERENCE, linestyle="--", linewidth=1.2, zorder=1)
    ax.plot(
        fd,
        ad,
        color=SERIES_MEASURED,
        linewidth=0.0,
        marker="o",
        markersize=MARKER_SIZE,
        markerfacecolor="none",
        markeredgewidth=1.4,
        zorder=3,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    return ax


# --- field snapshots: the cheap, field-based output path ----------------------
#
# SPARTA's `dump image` renders the *fields* to an image per frame rather than
# writing state.  That is the right default for a long run: a 128x128 field PNG is
# tens of kilobytes, where one frame of particle output at N ~ 1e6 is tens of
# megabytes and at the 1e9 target is not writable at all.  Particles remain
# available through `output.VTKSeries.add_particles`, but they are opt-in and
# subsampled; fields are what you watch a run with.

FIELD_CMAP_DIVERGING = "RdBu_r"  # velocities: signed, zero is meaningful
FIELD_CMAP_SEQUENTIAL = "viridis"  # densities, temperatures: positive magnitudes


@dataclasses.dataclass(frozen=True)
class Plane:
    """An axis-aligned cutting plane through a 3-D field.

    Args:
        axis: the axis the plane is **normal** to -- ``0`` for a ``x = const``
            plane, ``1`` for ``y = const``, ``2`` for ``z = const``.
        index: cell index along that axis.  ``None`` means use ``fraction``.
        fraction: position along the axis in ``[0, 1)`` when ``index`` is ``None``;
            ``0.0`` is the ``x = 0`` face, ``0.5`` the mid-plane.

    Only axis-aligned planes are supported.  A **general** plane (arbitrary normal
    and offset) needs interpolation onto the cut, which is a different operation
    with its own accuracy question -- nearest-cell would alias badly on a shock
    front, and trilinear smears it.  The type exists so that adding it later is a
    new constructor here rather than a change at every call site: everything
    downstream takes a ``Plane`` and asks it for a 2-D array plus axis labels.
    """

    axis: int = 0
    index: int | None = None
    fraction: float = 0.0

    def index_for(self, n: tuple[int, ...]) -> int:
        if self.index is not None:
            return int(self.index) % int(n[self.axis])
        return min(int(self.fraction * n[self.axis]), int(n[self.axis]) - 1)

    def label(self, n: tuple[int, ...]) -> str:
        return f"{'xyz'[self.axis]} = cell {self.index_for(n)}"

    def cut(self, grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Return ``(2-D slice, the two surviving axis numbers)``."""
        i = self.index_for(grid.shape)
        keep = tuple(a for a in range(grid.ndim) if a != self.axis)
        return np.take(grid, i, axis=self.axis), keep


PLANES_ORIGIN = (Plane(0, fraction=0.0), Plane(1, fraction=0.0), Plane(2, fraction=0.0))
"""The three faces ``x = 0``, ``y = 0``, ``z = 0`` -- the default 3-D view."""

PLANES_MID = (Plane(0, fraction=0.5), Plane(1, fraction=0.5), Plane(2, fraction=0.5))
"""The three mid-planes; usually more informative for a structure in the interior."""


def field_to_grid(field: Any, n: tuple[int, ...]) -> np.ndarray:
    """Reshape a per-cell ``(C,)`` array to ``n``, with x leading.

    Matches ``grid.py``'s row-major stride convention exactly (last axis fastest),
    so ``out[i, j, k]`` is the cell at index ``(i, j, k)``.  Any other reshape
    silently transposes the picture.
    """
    return np.asarray(field).reshape(tuple(int(x) for x in n))


def masked_field(field: Any, n: tuple[int, ...], mask: np.ndarray) -> np.ndarray:
    """Upsample a per-cell field by the sub-cell ``mask`` (``bool[C, sub^D]``,
    :func:`jax_dsmc.mesh.cut.flow_mask`): the cell value on every flow sub-cell, ``NaN``
    inside the body.  Returns the ``(n_0 sub, n_1 sub[, n_2 sub])`` grid, x leading, that
    :func:`draw_field` draws when given ``mask``."""
    g = field_to_grid(field, n)
    D = g.ndim
    sub = round(mask.shape[1] ** (1.0 / D))
    m = mask.reshape((*g.shape, *((sub,) * D)))
    # interleave: (n0, n1, sub, sub) -> (n0, sub, n1, sub) -> (n0 sub, n1 sub)
    order = [a for d in range(D) for a in (d, D + d)]
    m = np.transpose(m, order).reshape(tuple(int(x) * sub for x in g.shape))
    big = np.kron(g, np.ones((sub,) * D))
    return np.where(m, big, np.nan)


def draw_field(
    ax,
    field: Any,
    n: tuple[int, ...],
    lo: Any,
    hi: Any,
    *,
    label: str = "",
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    plane: Plane | None = None,
    mask: np.ndarray | None = None,
) -> Any:
    """Draw one per-cell field, dispatching on dimension.

    ``D = 1`` is a profile against x; ``D = 2`` is a heatmap; ``D = 3`` is the
    axis-aligned cut named by ``plane`` (default: the ``x = 0`` face).  A 3-D field
    has to be reduced somehow, and a *slice* is preferred over a projection because
    averaging along a direction hides exactly the structure -- a shock front -- that
    these pictures exist to show.

    Returns the mappable (or ``None`` in 1-D) so the caller can attach a colourbar.
    ``mask`` (``bool[C, sub^D]``) blanks the body of a cut-cell grid at sub-cell
    resolution (:func:`masked_field`); the values stay per cell.
    """
    g = field_to_grid(field, n) if mask is None else masked_field(field, n, mask)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = g.ndim

    if d == 1:
        x = (
            np.linspace(lo[0], hi[0], g.shape[0], endpoint=False)
            + 0.5 * (hi[0] - lo[0]) / g.shape[0]
        )
        ax.plot(x, g, color=SERIES_MEASURED, lw=LINE_WIDTH)
        ax.set_xlabel("x [m]")
        ax.set_ylabel(label)
        if vmin is not None and vmax is not None:
            ax.set_ylim(vmin, vmax)
        return None

    ax0, ax1 = 0, 1
    if d == 3:
        pl = plane or Plane(0, fraction=0.0)
        if mask is None:
            g, (ax0, ax1) = pl.cut(g)
        else:
            # the upsampled grid: slice at the same fraction of the finer resolution
            fine = Plane(
                pl.axis,
                index=None
                if pl.index is None
                else pl.index * (g.shape[pl.axis] // int(n[pl.axis])),
                fraction=pl.fraction,
            )
            g, (ax0, ax1) = fine.cut(g)
        label = f"{label}   [{pl.label(tuple(int(x) for x in n))}]"

    # imshow wants (row, col) = (vertical, horizontal), so transpose: the first
    # surviving axis runs horizontally.
    im = ax.imshow(
        g.T,
        origin="lower",
        aspect="auto",
        cmap=cmap or FIELD_CMAP_SEQUENTIAL,
        vmin=vmin,
        vmax=vmax,
        extent=(lo[ax0], hi[ax0], lo[ax1], hi[ax1]),
    )
    ax.set_xlabel(f"{'xyz'[ax0]} [m]")
    ax.set_ylabel(f"{'xyz'[ax1]} [m]")
    ax.set_title(label, fontsize=10)
    ax.grid(False)
    return im


def draw_surface(
    ax,
    points: Any,
    elements: Any,
    *,
    plane: Plane | None = None,
    lo: Any = None,
    hi: Any = None,
    color: str = INK,
    lw: float = 1.2,
) -> None:
    """Overlay an embedded surface on a field picture drawn by :func:`draw_field`.

    2-D: every line element is drawn.  3-D: the triangles are intersected with the
    axis-aligned ``plane`` (the same one handed to :func:`draw_field`; ``lo``/``hi``
    and the grid resolution are needed to locate it) and the resulting segments are
    drawn in the two surviving axes.  An analytic body is passed as ``points`` of
    shape ``(1, D)`` (the centre) and ``elements`` a float radius.
    """
    pts = np.asarray(points, dtype=float)
    d = pts.shape[1]
    ax.set_aspect("equal")  # a circle must look like one
    if not hasattr(elements, "shape") or np.asarray(elements).ndim == 0:
        # analytic circle / sphere: a circle of the given radius in the drawing plane
        r = float(elements)
        c = pts[0]
        if d == 3:
            pl = plane or Plane(0, fraction=0.0)
            keep = [a for a in range(3) if a != pl.axis]
            z = _plane_position(pl, lo, hi)
            if z is None or abs(z - c[pl.axis]) >= r:
                return
            r = float(np.sqrt(r * r - (z - c[pl.axis]) ** 2))
            c = c[keep]
        t = np.linspace(0.0, 2.0 * np.pi, 361)
        ax.plot(c[0] + r * np.cos(t), c[1] + r * np.sin(t), color=color, lw=lw)
        return
    el = np.asarray(elements)
    if d == 2:
        seg = pts[el]  # (E, 2, 2)
        for a, b in seg:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, solid_capstyle="round")
        return
    pl = plane or Plane(0, fraction=0.0)
    z = _plane_position(pl, lo, hi)
    if z is None:
        return
    keep = [a for a in range(3) if a != pl.axis]
    tri = pts[el]  # (E, 3, 3)
    for t in tri:
        cut = _triangle_plane_segment(t, pl.axis, z)
        if cut is not None:
            ax.plot(cut[:, keep[0]], cut[:, keep[1]], color=color, lw=lw)


def _plane_position(plane: Plane, lo: Any, hi: Any) -> float | None:
    """Coordinate of an axis-aligned plane; ``fraction`` is along ``[lo, hi]``."""
    if lo is None or hi is None:
        return None
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if plane.index is not None:
        return None  # a cell index needs the resolution; callers pass ``fraction``
    return float(lo[plane.axis] + plane.fraction * (hi[plane.axis] - lo[plane.axis]))


def _triangle_plane_segment(tri: np.ndarray, axis: int, z: float) -> np.ndarray | None:
    """The segment where a triangle crosses the plane ``x[axis] = z`` (``(2, 3)`` or ``None``)."""
    s = tri[:, axis] - z
    pts = []
    for i in range(3):
        j = (i + 1) % 3
        if (s[i] < 0.0) != (s[j] < 0.0):
            w = s[i] / (s[i] - s[j])
            pts.append(tri[i] + w * (tri[j] - tri[i]))
        elif s[i] == 0.0:
            pts.append(tri[i])
    if len(pts) < 2:
        return None
    return np.asarray(pts[:2])


def field_snapshot(
    path: str,
    fields: dict[str, Any],
    n: tuple[int, ...],
    lo: Any,
    hi: Any,
    *,
    title: str = "",
    limits: dict[str, tuple[float, float]] | None = None,
    cmaps: dict[str, str] | None = None,
    dpi: int = 110,
    plane: Plane | None = None,
    surface: tuple[Any, Any] | None = None,
    mask: np.ndarray | None = None,
) -> str:
    """Write one PNG with a panel per named field.

    ``limits`` fixes the colour/axis range per field.  **Pass it for a series.**
    Per-frame autoscaling makes an animation unreadable: the colours stop meaning
    a value and start meaning "relative to this frame", so a decaying shock looks
    like it is not decaying at all.  :class:`FieldMovie` handles this for you.
    ``surface = (points, elements)`` overlays an embedded surface on every panel
    (:func:`draw_surface`; an analytic body as ``(centre[None, :], radius)``).
    """
    limits = limits or {}
    cmaps = cmaps or {}
    names = list(fields)
    fig, axes = make_figure(1, len(names), figsize=(4.6 * len(names), 3.8))
    for ax, name in zip(axes, names, strict=True):
        vmin, vmax = limits.get(name, (None, None))
        im = draw_field(
            ax,
            fields[name],
            n,
            lo,
            hi,
            label=name,
            cmap=cmaps.get(name),
            vmin=vmin,
            vmax=vmax,
            plane=plane,
            mask=mask,
        )
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        if surface is not None and im is not None:
            draw_surface(ax, surface[0], surface[1], plane=plane, lo=lo, hi=hi)
    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    return savefig(fig, path, dpi=dpi)


class FieldMovie:
    """A directory of per-frame field PNGs with colour limits fixed across frames.

    Usage, host-side and outside the jitted step::

        movie = FieldMovie("out/shock", n=grid.n, lo=grid.lo, hi=grid.hi)
        movie.set_limits({"u_x": (-U0, U0), "nrho": (0.0, 4 * nrho)})
        for step, m in enumerate(frames):
            movie.add(step, {"u_x": m.u[:, 0], "nrho": m.nrho}, time=step * dt)

    If ``set_limits`` is not called, the limits are taken from the **first** frame
    and then held; that is usually wrong for a flow that steepens, so prefer
    setting them from the physics you expect.  Either way they are fixed, because
    an animation whose colour scale moves under it cannot be read.
    """

    def __init__(
        self,
        directory: str,
        n: tuple[int, ...],
        lo: Any,
        hi: Any,
        cmaps: dict[str, str] | None = None,
        dpi: int = 110,
        planes: tuple[Plane, ...] | None = None,
    ) -> None:
        self.directory = directory
        self.n = tuple(int(x) for x in n)
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.cmaps = cmaps or {}
        self.dpi = dpi
        # 3-D defaults to the three faces x=0, y=0, z=0; 1-D and 2-D have nothing
        # to cut, so a single `None` plane means "draw the field as it is".
        self.planes = (
            planes if planes is not None else (PLANES_ORIGIN if len(self.n) == 3 else (None,))
        )
        self.limits: dict[str, tuple[float, float]] = {}
        self.frames: list[str] = []
        os.makedirs(directory, exist_ok=True)

    def set_limits(self, limits: dict[str, tuple[float, float]]) -> None:
        self.limits.update(limits)

    def add(self, step: int, fields: dict[str, Any], time: float | None = None) -> list[str]:
        """Write one PNG per configured plane.  Returns the paths written.

        In 3-D the default is three files per frame -- the ``x = 0``, ``y = 0`` and
        ``z = 0`` faces -- because one slice of a 3-D flow says almost nothing about
        the other two directions.  In 1-D and 2-D there is nothing to cut, so one
        file is written.
        """
        for name, f in fields.items():
            self.limits.setdefault(name, (float(np.min(f)), float(np.max(f))))
        t = "" if time is None else f"   t = {time:.3e} s"
        out = []
        for pl in self.planes:
            tag = "" if pl is None else f"_{'xyz'[pl.axis]}{pl.index_for(self.n)}"
            path = os.path.join(self.directory, f"field{tag}_{int(step):05d}.png")
            field_snapshot(
                path,
                fields,
                self.n,
                self.lo,
                self.hi,
                title=f"step {step}{t}",
                limits=self.limits,
                cmaps=self.cmaps,
                dpi=self.dpi,
                plane=pl,
            )
            out.append(path)
        self.frames.extend(out)
        return out

    def contact_sheet(self, path: str, field: str, every: int = 1, ncols: int = 4) -> str:
        """One PNG showing the time evolution of a single field as a grid of panels.

        Cheaper to look at than an animation and it survives being pasted into a
        report, which is usually what you actually want from a milestone run.
        """
        picks = self.frames[::every]
        nrows = math.ceil(len(picks) / ncols)
        fig, axes = make_figure(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows))
        for ax in axes[len(picks) :]:
            ax.set_visible(False)
        for ax, p in zip(axes, picks, strict=False):
            ax.imshow(plt.imread(p))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        fig.suptitle(f"{field}: {len(picks)} frames", color=INK)
        return savefig(fig, path, dpi=90)
