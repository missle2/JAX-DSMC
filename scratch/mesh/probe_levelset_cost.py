"""Where does the level-set gradient run spend its time?  Forward / backward / shape terms
for the analytic and the level-set circle at a reduced step count, compiled call timed."""

import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
import jax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from examples import ensemble_mesh as EM  # noqa: E402

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
for body in ("analytic", "levelset"):
    cfg = EM.default_config()
    cfg.update({"body": body, "warmup": 200, "steps": steps, "window": steps * 2 // 3})
    for i in range(2):
        t0 = time.time()
        _terms, info, _s = EM.shape_terms(cfg, jax.random.key(100 + i))
        total = time.time() - t0
        print(
            f"{body} call {i}: total {total:.1f}s backward {info['seconds_backward']:.1f}s "
            f"shape {info['seconds_shape']:.1f}s forward {total - info['seconds_backward'] - info['seconds_shape']:.1f}s "
            f"hits {info['hits']}",
            flush=True,
        )
