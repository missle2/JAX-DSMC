"""sparta_jax: SPARTA's DSMC algorithm for the 2-D circle deck, one to one, in JAX (forward only)."""

import jax

jax.config.update("jax_enable_x64", True)

from . import geometry, sim  # noqa: E402

__all__ = ["geometry", "sim"]
