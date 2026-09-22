import time, math
import numpy as np, jax.numpy as jnp
from jax_dsmc.state import GridSpec
from jax_dsmc.mesh import surface as S, cut
from jax_dsmc.mesh.types import CUT, INSIDE, OUTSIDE, AnalyticBody
s = S.read_sparta_surf('/home/aakash/sparta/sparta/examples/sphere/data.sphere', 3)
grid = GridSpec(lo=jnp.array([-2.,-2.,-2.]), hi=jnp.array([2.,2.,2.]), n=(20,20,20))
t0=time.time(); cs = cut.overlaps(s, grid); print('overlaps', time.time()-t0, cs.shape)
t0=time.time(); st = cut.stencil(s, grid, 1); print('stencil', time.time()-t0, st.shape)
t0=time.time(); ct = cut.classify(s, grid, cs); print('classify', time.time()-t0, [(ct==k).sum() for k in (OUTSIDE, INSIDE, CUT)])
t0=time.time(); oc = cut.outside_corner(s, grid, ct, cs); print('oc', time.time()-t0, ((oc<0)&(ct==CUT)).sum())
for sub in (4, 8, 16):
    t0=time.time(); fr = cut.volume_fraction(s, grid, ct, cs, oc, sub=sub); vol = fr*0.008
    print(sub, time.time()-t0, vol.sum(), cut.exact_flow_volume(s, grid), vol.sum()-cut.exact_flow_volume(s, grid))
t0=time.time(); print('split', cut.split_check(s, grid, ct, cs, oc, raise_on_split=False), time.time()-t0)
ab = AnalyticBody(centre=jnp.array([0.,0.,0.]), radius=jnp.float64(1.0))
cs2 = cut.overlaps(ab, grid); ct2 = cut.classify(ab, grid, cs2); oc2 = cut.outside_corner(ab, grid, ct2, cs2)
print('analytic counts', [(ct2==k).sum() for k in (OUTSIDE, INSIDE, CUT)])
t0=time.time(); fr2 = cut.volume_fraction(ab, grid, ct2, cs2, oc2); print((fr2*0.008).sum(), 64-4*math.pi/3, time.time()-t0)
t0=time.time(); print('split an', cut.split_check(ab, grid, ct2, cs2, oc2, raise_on_split=False), time.time()-t0)
