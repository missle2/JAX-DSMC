"""Round-1 collision operators by name: the coarse baselines and the trained corrections.

Baselines (forward, ``theta`` at the physical coefficients): ``ntc``, ``ntc_adjacent``,
``esfp``, ``esfp_recon``, ``bgk``, ``esbgk``, ``bgk_recon``, ``esbgk_recon`` (``_recon``: the
reconstructed velocity target, :func:`jax_dsmc.collide.relax.collide`).

Trained models (on the differentiable ES-FP + reconstruction base): ``scalar`` (the two
coefficients), ``linear`` (the theory basis), ``mlp`` -- see :mod:`jax_dsmc.collide.relax`.

The collide functions are cached ``functools.partial`` objects: the jitted runners key their
compile cache on ``collide_fn``, which a ``partial`` hashes by identity (a fresh one per call
recompiled every call until the JIT ran out of code memory, 2026-11-04).
"""

from __future__ import annotations

import functools

import jax.numpy as jnp

from jax_dsmc.collide import relax

BASELINES = ("ntc", "ntc_adjacent", "esfp", "esfp_recon", "bgk", "esbgk", "bgk_recon",
             "esbgk_recon")  # fmt: skip
MODELS = ("scalar", "linear", "mlp")
PERIODIC = {"shear": (True, True), "shock": (False, True), "circle": (False, False)}
"""The axes a flow's cell gradients wrap around (the reconstruction and the features)."""


@functools.lru_cache(maxsize=64)
def collide_fn(base, periodic, corrected=None):
    """The (cached) collide function of a relaxation operator ``base``."""
    recon = base.endswith("_recon")
    kind = base.removesuffix("_recon")
    if kind == "esfp":
        return functools.partial(relax.collide, periodic=periodic if recon else None,
                                 corrected=corrected)  # fmt: skip
    if kind in ("bgk", "esbgk"):
        return functools.partial(relax.bgk_collide, es=kind == "esbgk",
                                 periodic=periodic if recon else None)  # fmt: skip
    raise ValueError(f"unknown operator {base}")


def operator(name, flow, key=None):
    """``(collide_fn, theta, pairing)`` of a baseline or model ``name`` on a flow kind
    (``None`` collide_fn = NTC with partner selection ``pairing``)."""
    if name == "ntc":
        return None, None, "uniform"
    if name == "ntc_adjacent":
        return None, None, "adjacent"
    periodic = PERIODIC[flow]
    if name in MODELS:
        corrected = None if name == "scalar" else name
        return (collide_fn("esfp_recon", periodic, corrected), relax.init_correction(
            corrected, key), "uniform")  # fmt: skip
    theta = {k: jnp.float64(v) for k, v in relax.BASELINE.items()}
    return collide_fn(name, periodic), theta, "uniform"
