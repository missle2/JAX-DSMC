import sys; sys.path.insert(0, ".")
import time
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from examples import shock_adjoint as sa

th = sa.reference_theta()
for dim in (1, 2, 3):
    cs = sa.case(dim)
    key = jax.random.PRNGKey(3)
    t0 = time.time()
    j, g = sa.value_and_grad(th, cs, key, False)      # pathwise
    _, gs = sa.value_and_grad(th, cs, key, True)      # full
    jv, diag = sa.evaluate(th, cs, key, True)
    print(f"\n=== D={dim}  C={cs.n_cells} N={cs.n_part} M={cs.n_steps} "
          f"work={cs.work:.1e}   J={float(j):.6e}  ({time.time()-t0:.1f}s)")
    print(f"    clipped {int(jnp.sum(diag.attempts_clipped))}  "
          f"violations {int(jnp.sum(diag.majorant_violations))}  "
          f"min margin {float(jnp.min(diag.min_majorant_margin)):.2f}  "
          f"max attempts {int(jnp.max(diag.max_attempts))}/{cs.a_max}")
    print(f"{'param':>7} {'h*':>9} {'FD (fixed key)':>16} {'AD pathwise':>16} {'rel err':>10}"
          f" {'AD full':>16}")
    for name in ("temp", "ux", "alpha", "diam", "omega", "nrho"):
        h = sa.largest_flip_free_h(th, cs, key, name)
        if h is None:
            print(f"{name:>7} {'none':>9}  -- no flip-free window --")
            continue
        fd, flipped = sa.fd_central(th, cs, key, name, h)
        ad, adf = float(g[name]), float(gs[name])
        rel = abs(fd - ad) / max(abs(ad), 1e-300) if ad != 0 else (0.0 if fd == 0 else np.inf)
        print(f"{name:>7} {h:9.1e} {fd:16.8e} {ad:16.8e} {rel:10.2e} {adf:16.8e}")
