"""Where after_step's 6.5 s goes (2026-10-17): its pieces in order, on the same synthetic step
as time_update.py, after one warm-up call.  Also: does project_area recompile every call (its
jit closes over a new partial each time)?"""
import dataclasses, json, sys, time
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "examples/mesh")
import optimize as O
from jax_dsmc.mesh import levelset as L
run, it = sys.argv[1], int(sys.argv[2])
d = f"examples/mesh/results/optimize/{run}"
cfg = json.load(open(f"{d}/history.json"))["config"]
grid, ls0 = O.initial_levelset(cfg)
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{d}/coef_{it}.npy")))
stepped = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{d}/coef_{it + 1}.npy")))  # the real next iterate
area0 = float(L.body_volume(ls0, lo, hi, 4))
def t(name, fn, *a):
    t0 = time.perf_counter(); r = fn(*a); jax.block_until_ready(jax.tree_util.tree_leaves(r)); print(f"{name:40s} {time.perf_counter() - t0:6.2f} s", flush=True); return r
O.after_step(stepped, "tangent", area0, lo, hi, 1.0)  # warm-up
for rep in range(2):
    print("pass", rep)
    new = t("reinitialise", L.reinitialise, stepped)
    new2, shift = t("project_area", O.project_area, new, area0, lo, hi)
    new3, moved = t("keep_inside", O.keep_inside, new2, lo, hi, 1.0)
    print("   knots moved by the floor:", moved)
    t("after_step", O.after_step, stepped, "tangent", area0, lo, hi, 1.0)
# compile count of project_area's jit: time 3 calls on the same ls
for i in range(3):
    t(f"project_area same ls #{i}", O.project_area, new, area0, lo, hi)
vol = jax.jit(lambda c, coef: L.body_volume(dataclasses.replace(ls, coef=coef + c), lo, hi, 4))
t("jitted body_volume (compile)", vol, 0.0, new.coef)
t("jitted body_volume (cached)", vol, 0.0, new.coef)
