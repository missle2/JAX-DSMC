"""Environment check for the cluster jobs: JAX version, backend and devices, float64, and
a small level-set circle forward + shape terms on the accelerator (a minute).

    python3 -m examples.mesh.cluster.check_env

Exit status 1 with a message if JAX is older than the code needs (``jax.tree.map``,
``jax.random.key``, ``jax.tree_util.register_dataclass`` with ``data_fields`` /
``meta_fields``: 0.4.36) or if no accelerator is visible.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

MIN_JAX = (0, 4, 36)


def main() -> int:
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)  # what `import jax_dsmc` does; env.sh exports it too
    version = tuple(int(p) for p in jax.__version__.split(".")[:3])
    print(f"jax {jax.__version__}  backend {jax.default_backend()}  devices {jax.devices()}")
    if version < MIN_JAX:
        print(
            f"jax {jax.__version__} is older than {'.'.join(map(str, MIN_JAX))}: use the CUDA venv route (env.sh, GPU_KIND=a100); the rocm/5.3.0 jax module is 0.4.7"
        )
        return 1
    if jax.default_backend() == "cpu":
        print("no accelerator visible: check the module / gres lines")
        return 1
    import jax.numpy as jnp  # noqa: PLC0415

    x = jnp.ones(4, dtype=jnp.float64)
    if x.dtype != jnp.float64:
        print("float64 is disabled: JAX_ENABLE_X64 must be 1")
        return 1
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    from examples import ensemble_mesh as EM  # noqa: PLC0415
    from jax_dsmc import ensemble_backward  # noqa: PLC0415

    ensemble_backward.configure("virtual", 1, 2, "staged")
    cfg = EM.default_config()
    cfg.update({"body": "levelset", "warmup": 50, "steps": 40, "window": 20, "fnum_scale": 16.0})
    t0 = time.time()
    all_terms, info, _ = EM.shape_terms(
        cfg, jax.random.key(0), None, "radius", 32, 0.2, ("along_pool",)
    )
    terms = all_terms["along_pool"]
    print(
        f"level-set circle, tiny run: J {info['objective']:.4g}, hits {info['hits']}, "
        f"|dJ/dcoef| {float(jnp.linalg.norm(terms.pathwise + terms.score)):.3g}, {time.time() - t0:.0f}s"
    )
    print("environment ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
