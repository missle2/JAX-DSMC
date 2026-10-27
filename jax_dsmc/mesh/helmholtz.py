"""Helmholtz (PDE) filter of a B-spline level set: the regularisation of the filtered level-set
formulation (van Dijk et al. 2013, section 3.4 / 4.2; Kreissl & Maute 2012; the Hilbertian
velocity extension of Allaire, de Gournay, Burger).

The design is a field ``s`` on the box ``D``; the level set is its filtered image ``phi = F s``,
``(-r^2 Lap + I) phi = s`` in the weak form with natural (Neumann) conditions on the box faces.
Both live in the level set's cubic B-spline space (:mod:`jax_dsmc.mesh.levelset`), so with the
Gram matrices ``M_ij = int_D B_i B_j`` and ``K_ij = int_D grad B_i . grad B_j``,

    A = r^2 K + M,      F = A^{-1} M      (coefficients of phi from those of s),
    dJ/ds = F^T dJ/dphi = M A^{-1} dJ/dphi.

``F`` keeps constants (``K 1 = 0`` on the box) and damps a Fourier mode of wavelength ``l`` by
``1 / (1 + (2 pi r / l)^2)``; its kernel is ``exp(-|x| / r) / 2r`` in 1-D (variance ``2 r^2``: a
Gaussian of ``sigma = sqrt(2) r`` has the same second moment).  ``A^{-1} dJ/dphi`` is the H1 Riesz
representer of the shape derivative (the Hilbertian descent velocity), so the steepest-descent
step ``ds = -dJ/ds``, ``dphi = F ds`` is a twice-filtered velocity.

The box is a tensor product and so are ``M`` and ``K`` (``M = M_1 (x) M_2``, ``K = K_1 (x) M_2 +
M_1 (x) K_2``): with the generalised eigenvectors ``U_a`` of each axis (``U^T K_a U = Lambda_a``,
``U^T M_a U = I``), ``A^{-1} = U D^{-1} U^T`` with the diagonal ``D = 1 + r^2 sum_a Lambda_a``,
applied exactly by mode products -- no sparse solve.  Coefficients whose basis function does
not reach into the box carry no information and are left out (zero in and out).
"""

from __future__ import annotations

import functools

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh

GAUSS = 4  # points per knot interval: exact for the degree-6 products of cubic B-splines


def _basis_np(f):
    """``levelset._basis`` in numpy: the weights of coefficients ``i-1 .. i+2`` at ``f`` and their
    ``f``-derivatives, ``[N, 4]`` each."""
    g = 1.0 - f
    w = np.stack([g**3 / 6.0, (3 * f**3 - 6 * f**2 + 4) / 6.0, (-3 * f**3 + 3 * f**2 + 3 * f + 1) / 6.0,
                  f**3 / 6.0], axis=-1)  # fmt: skip
    dw = np.stack([-(g**2) / 2.0, (3 * f**2 - 4 * f) / 2.0, (-3 * f**2 + 2 * f + 1) / 2.0,
                   f**2 / 2.0], axis=-1)  # fmt: skip
    return w, dw


def gram_1d(m: int, u_lo: float, u_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """``(M, K)`` of the ``m`` cubic B-splines centred at ``u = 0 .. m-1`` (unit knot spacing)
    over ``u in [u_lo, u_hi]``: ``M_jk = int B_j B_k du``, ``K_jk = int B_j' B_k' du``."""
    x, wq = leggauss(GAUSS)
    cuts = np.unique(np.r_[np.arange(np.floor(u_lo), np.ceil(u_hi) + 1), u_lo, u_hi])
    cuts = cuts[(cuts >= u_lo) & (cuts <= u_hi)]
    a, b = cuts[:-1], cuts[1:]
    u = (0.5 * (b - a))[:, None] * x[None, :] + (0.5 * (a + b))[:, None]  # [I, Q]
    wt = (0.5 * (b - a))[:, None] * wq[None, :]
    u, wt = u.ravel(), wt.ravel()
    i = np.floor(u).astype(int)
    w, dw = _basis_np(u - i)
    idx = i[:, None] + np.arange(-1, 3)[None, :]  # [N, 4]
    ok = (idx >= 0) & (idx < m)
    w, dw = np.where(ok, w, 0.0), np.where(ok, dw, 0.0)
    idx = np.clip(idx, 0, m - 1)
    mass, stiff = np.zeros((m, m)), np.zeros((m, m))
    rows = np.repeat(idx[:, :, None], 4, axis=2)
    cols = np.repeat(idx[:, None, :], 4, axis=1)
    np.add.at(mass, (rows, cols), wt[:, None, None] * w[:, :, None] * w[:, None, :])
    np.add.at(stiff, (rows, cols), wt[:, None, None] * dw[:, :, None] * dw[:, None, :])
    return mass, stiff


def _mode(mat, y, axis):
    """``mat`` applied along ``axis`` of ``y``."""
    return np.moveaxis(np.tensordot(mat, y, axes=(1, axis)), 0, axis)


class HelmholtzFilter:
    """``F`` and ``F^T`` of a coefficient grid ``shape`` (origin ``origin``, spacing ``h``) on the
    box ``[lo, hi]`` with filter radius ``radius`` (same length unit as ``h``)."""

    def __init__(self, shape, origin, h, lo, hi, radius):
        self.shape, self.radius = tuple(shape), float(radius)
        rho2 = (float(radius) / float(h)) ** 2  # the Gram matrices are in knot units
        self.slices, self.u, self.w, lams = [], [], [], []
        for a, m in enumerate(self.shape):
            u_lo = (float(lo[a]) - float(origin[a])) / h
            u_hi = (float(hi[a]) - float(origin[a])) / h
            mass, stiff = gram_1d(m, u_lo, u_hi)
            active = np.flatnonzero(np.diag(mass) > 1e-12)
            sl = slice(int(active[0]), int(active[-1]) + 1)
            mass, stiff = mass[sl, sl], stiff[sl, sl]
            lam, vec = eigh(stiff, mass)  # vec^T mass vec = I, vec^T stiff vec = diag(lam)
            self.slices.append(sl)
            self.u.append(vec)
            self.w.append(vec.T @ mass)  # = vec^{-1}
            lams.append(np.clip(lam, 0.0, None))
        total = functools.reduce(np.add.outer, lams) if len(lams) > 1 else lams[0]
        self.inv_d = 1.0 / (1.0 + rho2 * total)
        self.mask = np.zeros(self.shape, bool)
        self.mask[tuple(self.slices)] = True

    def _apply(self, x, first, second):
        y = np.asarray(x, dtype=np.float64)[tuple(self.slices)]
        for a, mat in enumerate(first):
            y = _mode(mat, y, a)
        y = y * self.inv_d
        for a, mat in enumerate(second):
            y = _mode(mat, y, a)
        out = np.zeros(self.shape)
        out[tuple(self.slices)] = y
        return out

    def forward(self, s):
        """``phi = F s = A^{-1} M s`` (coefficients)."""
        return self._apply(s, self.w, self.u)

    def adjoint(self, g):
        """``F^T g = M A^{-1} g``: the gradient with respect to ``s`` of a gradient ``g`` with
        respect to ``phi``'s coefficients."""
        return self._apply(g, [u.T for u in self.u], [w.T for w in self.w])

    def riesz(self, g):
        """``A^{-1} g``, the coefficients of the H1 Riesz representer of the functional
        ``g . dphi`` (the Hilbertian velocity extension), scaled by the knot volume ``h^D``."""
        return self._apply(g, [u.T for u in self.u], self.u)


@functools.lru_cache(maxsize=4)
def _cached(shape, origin, h, lo, hi, radius):
    return HelmholtzFilter(shape, origin, h, lo, hi, radius)


def filter_for(ls, lo, hi, radius: float) -> HelmholtzFilter:
    """The (cached) filter of the level set ``ls``'s coefficient grid on the box ``[lo, hi]``."""
    key = (
        tuple(np.asarray(ls.coef).shape),
        tuple(float(v) for v in np.asarray(ls.lo)),
        float(ls.h),
        tuple(float(v) for v in np.asarray(lo)),
        tuple(float(v) for v in np.asarray(hi)),
        float(radius),
    )
    return _cached(*key)
