"""Domain decomposition vs single device, on forced-host devices."""
import sys; sys.path.insert(0, ".")
import dataclasses, math
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from examples.turbulence.case import setup, initial_particles
from jax_dsmc import decomp, parallel
from jax_dsmc.boundary import BC
from jax_dsmc.simulate import run
from jax_dsmc.state import ParticleState

P = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DIM = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 32
NPC = 40
STEPS = 12

sp, grid, static, params, cstate, info = setup(dim=DIM, cells_per_side=SIDE, npc=NPC, n_steps=STEPS)
static = dataclasses.replace(static, n_steps=STEPS)
key = jax.random.PRNGKey(0)
st0 = initial_particles(key, grid, static, info)
print(f"D={DIM} side={SIDE} C={info['C']} N={info['N']:,} P={P} steps={STEPS}")

# --- single device reference ---
(s1, c1), (m1, d1) = jax.jit(lambda s, c: run(s, c, grid, sp, params, static, key, bc=BC.PERIODIC))(st0, cstate)
n_alive1 = int(jnp.sum(s1.alive))
print(f"single : alive {n_alive1:,}  nrho mean {float(jnp.mean(m1.nrho[-1])):.6e}  "
      f"T mean {float(jnp.mean(m1.temp[-1])):.4f}")

# --- decomposed ---
mesh = parallel.device_mesh(P)
per = decomp.suggest_capacity(info["N"], P, headroom=1.6)   # PER DEVICE
m_max = decomp.suggest_migration_buffer(per, SIDE // P)
print(f"decomp : capacity/device {per:,}  migration buffer {m_max}")

# pad the state out to P * cap_per_dev, particles grouped by slab
slab = np.asarray(decomp.slab_of(st0.x, grid, P))
xs, vs, sps, al, lg = [], [], [], [], []
for r in range(P):
    idx = np.where(slab == r)[0]
    if idx.size > per:
        raise SystemExit(f"slab {r} holds {idx.size} > capacity {per}")
    pad = per - idx.size
    xs.append(np.concatenate([np.asarray(st0.x)[idx], np.zeros((pad, DIM))]))
    vs.append(np.concatenate([np.asarray(st0.v)[idx], np.zeros((pad, 3))]))
    sps.append(np.zeros(per, np.int32))
    al.append(np.concatenate([np.ones(idx.size, bool), np.zeros(pad, bool)]))
    lg.append(np.zeros(per))
st_d = ParticleState(
    x=jnp.asarray(np.concatenate(xs)), v=jnp.asarray(np.concatenate(vs)),
    species=jnp.asarray(np.concatenate(sps)), alive=jnp.asarray(np.concatenate(al)),
    logh_acc=jnp.asarray(np.concatenate(lg)))
with mesh:
    st_d = parallel.shard_state(st_d, mesh)
    (s2, c2), (m2, d2, mig) = decomp.run_decomposed(
        st_d, cstate, grid, sp, params, static, key, mesh, BC.PERIODIC, STEPS, m_max)
n_alive2 = int(jnp.sum(s2.alive))
print(f"decomp : alive {n_alive2:,}  nrho mean {float(jnp.mean(m2.nrho[-1])):.6e}  "
      f"T mean {float(jnp.mean(m2.temp[-1])):.4f}")
print(f"  migration: sent/step {float(jnp.mean(mig['sent'])):.1f}  "
      f"overflow_send {int(jnp.sum(mig['overflow_send']))}  "
      f"overflow_recv {int(jnp.sum(mig['overflow_recv']))}  "
      f"far_jumps {int(jnp.sum(mig['far_jumps']))}")
print(f"  PARTICLE CONSERVATION: {n_alive2:,} vs {info['N']:,}  "
      f"({'OK' if n_alive2 == info['N'] else 'LOST ' + str(info['N'] - n_alive2)})")
r1, r2 = float(jnp.mean(m1.nrho[-1])), float(jnp.mean(m2.nrho[-1]))
t1, t2 = float(jnp.mean(m1.temp[-1])), float(jnp.mean(m2.temp[-1]))
print(f"  nrho agree to {abs(r2/r1-1)*100:.4f} %   temp agree to {abs(t2/t1-1)*100:.4f} %")
