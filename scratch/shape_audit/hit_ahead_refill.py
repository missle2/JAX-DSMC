"""How often does shape.hit_ahead class a slot by a LATER occupant's hit?  The class of
(step k, slot i) should be "the particle in slot i at the end of step k hits the body at a
later captured step"; hit_ahead uses any later hit of the slot, whoever occupies it then.

    PYTHONPATH=. python3 scratch/shape_audit/hit_ahead_refill.py CASE [knudsen mach dt slots a_max]
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from examples import ensemble_mesh as EM
from jax_dsmc import shape
from jax_dsmc.wall_adjoint import SplitEvents, particle_sum, RecordEvents


def per_step_hits(records):
    dim = records.x.shape[-1]
    walls = records.walls
    if isinstance(walls, RecordEvents):
        head = jnp.any(walls.head.face >= 2 * dim, axis=1)
        n = head.shape[1]
        body = (walls.face >= 2 * dim) & (walls.slot >= 0)
        target = jnp.where(body, walls.slot, n)
        steps = jnp.arange(head.shape[0])[:, None]
        return head.at[steps, target].set(True, mode="drop")
    if isinstance(walls, SplitEvents):
        head = jnp.any(walls.head.face >= 2 * dim, axis=1)
        tail = jnp.any(walls.tail.face >= 2 * dim, axis=1).astype(jnp.float64)
        folded = jax.vmap(particle_sum)(
            jnp.zeros_like(head, jnp.float64)[:, None], tail[:, None], walls.slot
        )
        return head | (folded > 0)
    return jnp.any(walls.face >= 2 * dim, axis=1)


def own_class(hit, born):
    """later[k] = hit[k+1] | (later[k+1] & ~born[k+1]): stop at the slot's next birth."""
    hit, born = np.asarray(hit), np.asarray(born)
    later = np.zeros_like(hit)
    for k in range(hit.shape[0] - 2, -1, -1):
        later[k] = hit[k + 1] | (later[k + 1] & ~born[k + 1])
    return later


case = sys.argv[1]
cfg = EM.default_config()
cfg.update(case=case, body="levelset")
param = "squash_y" if case in ("ellipse", "spheroid") else "radius"
if len(sys.argv) > 2:
    kn, mach, dt, slots, a_max = sys.argv[2:7]
    cfg.update(knudsen=float(kn), mach=float(mach), dt=float(dt), slots=int(slots), a_max=int(a_max))
state0, final, diag, rec, tallies, s = EM.forward(
    cfg, param, EM.nominal(cfg, param), jax.random.key(1), True
)
hit = per_step_hits(rec)
alive = np.asarray(rec.alive)
born = np.asarray(rec.born_face) >= 0
cur = np.asarray(shape.hit_ahead(rec, s["mesh"]))
own = own_class(hit, born)
assert not np.any(own & ~cur), "the own class must be a subset"
wrong = cur & ~own & alive
hits_now = np.asarray(hit) & alive  # the queries of hit_fits (body event this step)
print(f"case {case} cfg kn={cfg['knudsen']} mach={cfg['mach']}")
print(f"alive entries {alive.sum()}, classed hit-ahead {(cur & alive).sum()}, "
      f"of which by a later occupant {wrong.sum()} ({wrong.sum() / max((cur & alive).sum(), 1):.1%})")
print(f"slots reborn in the window: {np.any(born, axis=0).mean():.1%} of slots; "
      f"births per slot {born.sum(0).mean():.2f}")
print(f"body-hit queries {hits_now.sum()}, misclassed {(wrong & hits_now).sum()} "
      f"({(wrong & hits_now).sum() / max(hits_now.sum(), 1):.1%})")
