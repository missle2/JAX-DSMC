"""Per-cell flow volumes vs SPARTA's `compute property/grid vol` dumps."""
import sys, time
import numpy as np, jax.numpy as jnp
from jax_dsmc.state import GridSpec
from jax_dsmc.grid import cell_of
from jax_dsmc.mesh import surface as S, cut
from jax_dsmc.mesh.types import CUT
sys.path.insert(0, "examples/mesh")
from sparta_io import read_grid_dump

def compare(name, surf, grid, dump, sub):
    m = cut.build_mesh(surf, grid, sub=sub)
    d = read_grid_dump(dump, snapshot=0)
    D = len(grid.n)
    xc = np.stack([d["xc"], d["yc"]] + ([d["zc"]] if D == 3 else []), axis=1)
    cid = np.asarray(cell_of(jnp.asarray(xc), grid))
    ours = np.asarray(m.volume)
    # SPARTA may list a cut cell several times (split); sum rows per id
    ref = np.zeros_like(ours); np.add.at(ref, cid, d["vol"])
    ct = np.asarray(m.cell_type)
    err = ours - ref
    print(name, "sub", sub, "rows", len(cid), "unique", len(np.unique(cid)),
          "max|err| cut", np.abs(err[ct == CUT]).max(), "max|err| other", np.abs(err[ct != CUT]).max(),
          "sum ours", ours.sum(), "sum ref", ref.sum())
    full = np.prod((np.asarray(grid.hi) - np.asarray(grid.lo)) / np.asarray(grid.n))
    cutset_ref = (ref > 1e-12) & (ref < full - 1e-12)
    print("  cut-set identical:", np.array_equal(cutset_ref, ct == CUT), "n", cutset_ref.sum())
    return err

g2 = GridSpec(lo=jnp.array([0.,0.]), hi=jnp.array([10.,10.]), n=(20,20))
c = S.read_sparta_surf('examples/mesh/sparta/data.circle', 2)
for sub in (8, 32, 64):
    compare("circle", c, g2, "examples/mesh/results/sparta/grid_vol.circle_ar.12345.txt", sub)
g3 = GridSpec(lo=jnp.array([-2.,-2.,-2.]), hi=jnp.array([2.,2.,2.]), n=(20,20,20))
s = S.read_sparta_surf('examples/mesh/sparta/data.sphere', 3)
for sub in (8, 16):
    t0 = time.time(); compare("sphere", s, g3, "examples/mesh/results/sparta/grid_vol.sphere_ar.12345.txt", sub); print("  t", time.time()-t0)

# which cells differ in the cut-set test?
m = cut.build_mesh(c, g2, sub=32); d = read_grid_dump("examples/mesh/results/sparta/grid_vol.circle_ar.12345.txt", snapshot=0)
cid = np.asarray(cell_of(jnp.asarray(np.stack([d["xc"], d["yc"]], 1)), g2)); ref = np.zeros(400); ref[cid] = d["vol"]
ct = np.asarray(m.cell_type); ours = np.asarray(m.volume)
partial = (ref > 1e-12) & (ref < 0.25 - 1e-12)
for i in np.where(partial != (ct == CUT))[0]:
    print("cell", i, "xc", d["xc"][cid == i], d["yc"][cid == i], "type", ct[i], "ours", ours[i], "ref", ref[i], "csurfs", np.asarray(m.csurfs)[i])
