import time, math
import numpy as np, jax.numpy as jnp
from jax_dsmc.state import GridSpec
from jax_dsmc.mesh import surface as S, cut
from jax_dsmc.mesh.types import CUT, INSIDE, OUTSIDE, AnalyticBody
c = S.read_sparta_surf('/home/aakash/sparta/sparta/examples/circle/data.circle', 2)
grid = GridSpec(lo=jnp.array([0.,0.]), hi=jnp.array([10.,10.]), n=(20,20))
t0=time.time()
cs = cut.overlaps(c, grid); st = cut.stencil(c, grid, 1); ct = cut.classify(c, grid, cs); oc = cut.outside_corner(c, grid, ct, cs)
print('counts', [(ct==k).sum() for k in (OUTSIDE, INSIDE, CUT)], 'S', cs.shape, st.shape, 'oc -1 in cut:', ((oc<0)&(ct==CUT)).sum(), time.time()-t0)
for sub in (8,16,32,64):
    fr = cut.volume_fraction(c, grid, ct, cs, oc, sub=sub)
    vol = fr*0.25
    print(sub, vol.sum(), cut.exact_flow_volume(c, grid), vol.sum()-cut.exact_flow_volume(c, grid))
t0=time.time(); print('split', cut.split_check(c, grid, ct, cs, oc, raise_on_split=False), time.time()-t0)
ab = AnalyticBody(centre=jnp.array([5.,5.]), radius=jnp.float64(3.0))
cs2 = cut.overlaps(ab, grid); ct2 = cut.classify(ab, grid, cs2); oc2 = cut.outside_corner(ab, grid, ct2, cs2)
print('analytic counts', [(ct2==k).sum() for k in (OUTSIDE, INSIDE, CUT)])
fr2 = cut.volume_fraction(ab, grid, ct2, cs2, oc2); print((fr2*0.25).sum(), 100-9*math.pi)
t0=time.time(); m = cut.build_mesh(c, grid); print('build', time.time()-t0, float(m.volume.sum()))
