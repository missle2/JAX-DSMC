"""Why does the tiny D=1 case disagree while D=2 and D=3 agree?"""
import sys; sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from examples import shock_adjoint as sa

th = sa.reference_theta()
for dim in (1, 2, 3):
    cs = sa.case(dim, npc=20, n_steps=4, n_x=8, transverse=2, a_max=64)
    key = jax.random.PRNGKey(0)
    j, g = sa.value_and_grad(th, cs, key, False)
    print(f"\n--- D={dim}  C={cs.n_cells} N={cs.n_part} work={cs.work:.1e}  J={float(j):.8e}")
    for name in ("temp",):
        print(f"    AD pathwise d/d{name} = {float(g[name]):.10e}")
        print(f"    {'h':>10} {'FD':>18} {'flip':>5} {'floor':>10} {'rel':>10}")
        h = 1e-3
        for _ in range(14):
            fd, flip, floor = sa.fd_central(th, cs, key, name, h)
            ad = float(g[name])
            print(f"    {h:10.2e} {fd:18.10e} {str(flip):>5} {floor:10.2e} "
                  f"{abs(fd-ad)/abs(ad):10.2e}")
            h /= 3.0
