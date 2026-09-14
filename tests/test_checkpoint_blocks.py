"""Block replay preserves values, event streams and full-horizon gradients."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import adjoint, simulate
from tests.test_ntc import build


def test_padded_checkpoint_blocks_preserve_full_horizon_gradient():
    state, cs, grid, sp, params, static, _, key = build(N=16,A_max=32,dt=1e-5)
    def value(dt, block):
        st=dataclasses.replace(static,n_steps=7,checkpoint_block_size=block)
        (out,_),(_,diag)=simulate.run(state,cs,grid,sp,dataclasses.replace(params,dt=dt),st,key,
                                      collide_only=True,emit_moments='none',score_mode='trajectory')
        return adjoint.trajectory_surrogate(jnp.sum(out.v**4),out.logh_acc[0]),diag
    reference=jax.value_and_grad(lambda dt:value(dt,0),has_aux=True)(params.dt)
    got=jax.value_and_grad(lambda dt:value(dt,3),has_aux=True)(params.dt)
    for a,b in zip(jax.tree.leaves(reference),jax.tree.leaves(got),strict=True):
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
