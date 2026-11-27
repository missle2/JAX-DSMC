"""Minimum length scale by a spread skeleton (Hermann & Noel 2026, "Length scale control for
level set-based topology optimization through spread skeletons", Struct. Multidisc. Optim.
69:119, section 3.1), on a voxel grid with an ersatz material.

The skeleton (the ridge of the distance from the surface) is found from the direction of the
gradient of a one-step heat solution, then spread to half the minimum length scale ``r_s`` by
a Helmholtz smoothing of that direction; where a member is thinner than ``2 r_s`` the smoothed
direction field ``eta`` has ``|eta| < 1`` at the surface:

1. heat (their eq. 15, one implicit step from 0, ``theta = 1`` on the surface), solved in the
   solid: ``(H / tau) theta - div(kappa grad theta) + beta (1 - H)(theta - 1) = 0``;
2. smoothing (eq. 16, natural Neumann at the surface): ``-div(r^2 kappa grad eta) + (H + e)
   eta = H g``, ``g = grad theta / |grad theta|``, ``r = r_s / (2 sqrt 3)``;
3. the corrected spread skeleton (eq. 20) ``S_c = (sqrt(max(1 - (|eta| / eta_c)^p, 0)^2 + xi^2)
   - xi) / (sqrt(1 + xi^2) - xi)``;
4. the penalty (eqs. 21-22), the activated share of the surface: ``P = int_Gamma gamma S_c /
   int_Gamma 1``, ``gamma = (1 + tanh(-beta_L (S_c - S_th))) / 2`` with ``S_c`` frozen inside
   ``gamma`` (the paper leaves the activation out of the sensitivities); ``S_th`` is ramped
   0 -> 1 by the caller (their eq. 23).

The solid is the smoothed Heaviside ``H = (1 - tanh(phi / w)) / 2`` of the level set at the
voxel centres (spacing at most ``r_s / 1.5``, the paper's resolution bound), with conductivity
``kappa = kappa_void + (1 - kappa_void) H``: the void conducts a thousandth as well, so a
member's fields do not leak to its neighbours, while ``eta``'s small void reaction makes its
void values the (nearly constant) extension of the surface values -- the surface delta, which
straddles the surface, then reads the solid side's ``|eta|``.  Both solves are symmetric
positive definite, by Jacobi-preconditioned CG (``jax.scipy.sparse.linalg.cg``, implicitly
differentiated), so ``dP/dcoef`` is reverse mode through the B-spline evaluation and both
solves.  The voxel block is cropped to the body's bounding box plus a margin (static, from the
current coefficients), its counts rounded up to multiples of :data:`ROUND` voxels so few
shapes are compiled.

No nested function definitions (module-level functions bound with ``functools.partial``).
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc.mesh import levelset as L
from jax_dsmc.mesh.types import LevelSet

jax.config.update("jax_enable_x64", True)

ROUND = 8  # voxel counts are rounded up to multiples of this (static shapes)


@dataclasses.dataclass(frozen=True)
class SkeletonSettings:
    """The measure's constants (paper values where they have one).

    r_s: half the minimum length scale (m).
    spacing: voxel spacing (m); 0 = ``min(h / 2, r_s / 1.5)``.
    heat_scale: sqrt(tau) / r_s, the heat step's diffusion length over ``r_s``.
    eta_c, xi, p: eq. 20's correction, transition and exponent (0.937, 1e-3, 2).
    beta_l: eq. 22's activation sharpness (25).
    kappa_void: the void's conductivity relative to the solid's.
    reaction_void: the smoothing's void reaction (relative to the solid's 1).
    width: the Heaviside / delta half-width in voxels.
    tol, maxiter: CG tolerance (relative residual) and iteration cap.
    """

    r_s: float
    spacing: float = 0.0
    heat_scale: float = 2.0
    eta_c: float = 0.937
    xi: float = 1e-3
    p: float = 2.0
    beta_l: float = 25.0
    kappa_void: float = 1e-3
    reaction_void: float = 1e-6
    width: float = 1.0
    tol: float = 1e-10
    maxiter: int = 4000


def voxel_spacing(ls: LevelSet, s: SkeletonSettings) -> float:
    return float(s.spacing) if s.spacing else min(float(ls.h) / 2.0, s.r_s / 1.5)


def voxel_block(ls: LevelSet, lo, hi, s: SkeletonSettings):
    """``(origin, counts, spacing)`` of the cropped voxel block: the body's bounding box (where
    ``phi < 0`` on a coarse scan) plus ``3 r_s + 4`` voxels, clipped to ``[lo, hi]``, counts
    rounded up to multiples of :data:`ROUND`.  ``None`` when there is no body."""
    hv = voxel_spacing(ls, s)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    x, _cell = L._subgrid(ls, lo, hi, 2)
    phi = _phi_chunked(ls, x)
    inside = phi < 0.0
    if not inside.any():
        return None
    pts = np.asarray(x)[inside]
    margin = 3.0 * s.r_s + 4.0 * hv + float(ls.h)
    b_lo = np.maximum(pts.min(axis=0) - margin, lo)
    b_hi = np.minimum(pts.max(axis=0) + margin, hi)
    counts = tuple(int(ROUND * math.ceil(math.ceil((b_hi[a] - b_lo[a]) / hv) / ROUND))
                   for a in range(lo.size))  # fmt: skip
    centre = 0.5 * (b_lo + b_hi)
    origin = centre - 0.5 * hv * np.asarray(counts)
    return origin, counts, hv


def _phi_chunked(ls, x):
    """``phi`` at ``x`` (numpy), in chunks of :data:`levelset.SUBGRID_CHUNK`."""
    n = x.shape[0]
    pad = -n % L.SUBGRID_CHUNK
    xp = jnp.concatenate((x, jnp.broadcast_to(x[:1], (pad, x.shape[1]))))
    phi = jax.lax.map(functools.partial(L._phi_chunk, ls=ls),
                      xp.reshape(-1, L.SUBGRID_CHUNK, x.shape[1]))  # fmt: skip
    return np.asarray(phi).ravel()[:n]


def _centres(origin, counts, hv):
    axes = [origin[a] + hv * (np.arange(n) + 0.5) for a, n in enumerate(counts)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1)


def _laplace(u, faces, hv):
    """``-div(k grad u)`` on the voxel block (zero flux through the block's faces); ``faces[a]``
    the conductivities of the interior faces normal to axis ``a``."""
    out = jnp.zeros_like(u)
    for a, k in enumerate(faces):
        flux = k * jnp.diff(u, axis=a) / hv**2
        pad_lo = [(0, 0)] * u.ndim
        pad_hi = [(0, 0)] * u.ndim
        pad_lo[a] = (1, 0)
        pad_hi[a] = (0, 1)
        out = out + jnp.pad(flux, pad_hi) - jnp.pad(flux, pad_lo)
    return -out


def _face_sum(faces, shape, hv):
    """Per voxel, the sum of its faces' conductivities over ``hv^2`` (the Laplacian's diagonal)."""
    out = jnp.zeros(shape)
    for a, k in enumerate(faces):
        pad_lo = [(0, 0)] * len(shape)
        pad_hi = [(0, 0)] * len(shape)
        pad_lo[a] = (1, 0)
        pad_hi[a] = (0, 1)
        out = out + jnp.pad(k, pad_hi) + jnp.pad(k, pad_lo)
    return out / hv**2


def _faces(kappa):
    return [0.5 * (jnp.take(kappa, jnp.arange(kappa.shape[a] - 1), axis=a)
                   + jnp.take(kappa, jnp.arange(1, kappa.shape[a]), axis=a))
            for a in range(kappa.ndim)]  # fmt: skip


def _operator(u, reaction, faces, scale, hv):
    return reaction * u + scale * _laplace(u, faces, hv)


def _jacobi(u, diag):
    return u / diag


def _solve(reaction, faces, scale, hv, rhs, s):
    op = functools.partial(_operator, reaction=reaction, faces=faces, scale=scale, hv=hv)
    diag = reaction + scale * _face_sum(faces, rhs.shape, hv)
    pre = functools.partial(_jacobi, diag=diag)
    sol, _ = jax.scipy.sparse.linalg.cg(op, rhs, tol=s.tol, maxiter=s.maxiter, M=pre)
    return sol


def _phi_and_norm(coef, ls, x):
    phi, grad = L.value_and_grad(dataclasses.replace(ls, coef=coef), x)
    return phi, L.safe_norm(grad)


def fields(coef, ls, x, counts, hv, s: SkeletonSettings):
    """The voxel fields ``(phi, |grad phi|, H, theta, eta [D, ...], S_c)`` on the block whose
    centres are ``x`` (``f64[N, D]``, C order, ``counts`` per axis)."""
    phi, gnorm = _phi_and_norm(coef, ls, x)
    phi = phi.reshape(counts)
    gnorm = gnorm.reshape(counts)
    w = s.width * hv
    solid = 0.5 * (1.0 - jnp.tanh(phi / w))
    kappa = s.kappa_void + (1.0 - s.kappa_void) * solid
    faces = _faces(kappa)
    tau = (s.heat_scale * s.r_s) ** 2
    beta = 1e4 / hv**2
    reaction = solid / tau + beta * (1.0 - solid)
    theta = _solve(reaction, faces, 1.0, hv, beta * (1.0 - solid), s)
    grads = jnp.stack(jnp.gradient(theta, hv), axis=0)
    norm = jnp.sqrt(jnp.sum(grads**2, axis=0) + (1e-9 / hv) ** 2)
    g = grads / norm
    r2 = (s.r_s / (2.0 * math.sqrt(3.0))) ** 2
    react = solid + s.reaction_void
    eta = jnp.stack([_solve(react, faces, r2, hv, solid * g[a], s) for a in range(len(counts))])
    mag = jnp.sum(eta**2, axis=0) / s.eta_c**2  # (|eta| / eta_c)^2
    ratio = mag ** (0.5 * s.p)
    base = jnp.maximum(1.0 - ratio, 0.0)
    s_c = (jnp.sqrt(base**2 + s.xi**2) - s.xi) / (math.sqrt(1.0 + s.xi**2) - s.xi)
    return phi, gnorm, solid, theta, eta, s_c


def penalty(coef, ls, x, counts, hv, threshold, s: SkeletonSettings):
    """``P = sum gamma S_c delta / sum delta`` over the voxels (``delta`` the smoothed surface
    delta ``sech^2(phi / w) |grad phi| / 2w``), ``gamma`` with ``S_c`` frozen."""
    phi, gnorm, _solid, _theta, _eta, s_c = fields(coef, ls, x, counts, hv, s)
    w = s.width * hv
    delta = 0.5 / w / jnp.cosh(phi / w) ** 2 * gnorm
    gamma = 0.5 * (1.0 + jnp.tanh(-s.beta_l * (jax.lax.stop_gradient(s_c) - threshold)))
    return jnp.sum(gamma * s_c * delta) / jnp.sum(delta)


@functools.partial(jax.jit, static_argnames=("counts", "s"))
def _value_and_grad(coef, ls, x, counts, hv, threshold, s):
    return jax.value_and_grad(penalty)(coef, ls, x, counts, hv, threshold, s)


def penalty_and_gradient(ls: LevelSet, lo, hi, threshold: float, s: SkeletonSettings):
    """``(P, dP/dcoef)`` of the body ``ls`` in the box ``[lo, hi]`` at activation threshold
    ``threshold`` (the paper's ``S_th``; 1 = every member penalised)."""
    block = voxel_block(ls, lo, hi, s)
    if block is None:
        return 0.0, jnp.zeros_like(ls.coef)
    origin, counts, hv = block
    x = jnp.asarray(_centres(origin, counts, hv))
    value, grad = _value_and_grad(ls.coef, ls, x, tuple(counts), float(hv), float(threshold), s)
    return float(value), grad


def surface_fields(ls: LevelSet, lo, hi, s: SkeletonSettings):
    """Diagnostics: the block ``(origin, counts, hv)`` and its fields as numpy arrays."""
    block = voxel_block(ls, lo, hi, s)
    if block is None:
        return None
    origin, counts, hv = block
    x = jnp.asarray(_centres(origin, counts, hv))
    out = fields(ls.coef, ls, x, tuple(counts), float(hv), s)
    names = ("phi", "gnorm", "solid", "theta", "eta", "s_c")
    return {"origin": origin, "counts": counts, "hv": hv,
            **{k: np.asarray(v) for k, v in zip(names, out, strict=True)}}  # fmt: skip


def ramp(iterate: int, start: int, end: int) -> float:
    """Eq. 23's threshold ``S_th``: 0 up to ``start``, linear to 1 at ``end``, then 1."""
    if iterate <= start:
        return 0.0
    if iterate >= end:
        return 1.0
    return (iterate - start) / (end - start)
