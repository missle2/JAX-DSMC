p = 'varB/jax_dsmc/mesh/levelset.py'
s = open(p).read()
patch = '''

def _sample_step_b(i, carry, ls, x, vhat, ds):
    found, s_lo, s_hi, phi_prev = carry
    s = i * ds
    phi = value(ls, x + s[:, None] * vhat)
    cross = ~found & (phi_prev > 0.0) & (phi <= 0.0)
    s_lo = jnp.where(cross, s - ds, s_lo)
    s_hi = jnp.where(cross, s, s_hi)
    return found | cross, s_lo, s_hi, phi


def _bisect_step_b(_i, carry, ls, x, vhat):
    s_lo, s_hi = carry
    mid = 0.5 * (s_lo + s_hi)
    phi = value(ls, x + mid[:, None] * vhat)
    positive = phi > 0.0
    return jnp.where(positive, mid, s_lo), jnp.where(positive, s_hi, mid)


def _samples(ls, x, vhat, ds, phi_start):
    n = x.shape[0]
    carry = (jnp.zeros(n, bool), jnp.zeros(n), jnp.zeros(n), phi_start)
    found, s_lo, s_hi, _ = jax.lax.fori_loop(
        1, ls.n_sample + 1, functools.partial(_sample_step_b, ls=ls, x=x, vhat=vhat, ds=ds), carry
    )
    return found, s_lo, s_hi, s_lo, s_hi


def _refine(ls, x, vhat, found, s_lo, s_hi, phi_lo, phi_hi, tol):
    _s_lo, s_hi = jax.lax.fori_loop(
        0, ls.n_bisect, functools.partial(_bisect_step_b, ls=ls, x=x, vhat=vhat), (s_lo, s_hi)
    )
    return s_hi
'''
open(p, 'w').write(s + patch)
