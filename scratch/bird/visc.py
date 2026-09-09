"""2-D Taylor-Green: measure the viscosity against Chapman-Enskog."""
import argparse, pickle, sys, time
sys.path.insert(0, ".")
import numpy as np
from examples.turbulence.case import run, setup
from examples.turbulence.validate import energy_history, fit_decay, report_2d

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--side", type=int, default=128)
ap.add_argument("--npc", type=int, default=100)
ap.add_argument("--mach", type=float, default=0.3)
ap.add_argument("--cells-per-mfp", type=float, default=1.0)
ap.add_argument("--efolds", type=float, default=2.5)
ap.add_argument("--stride", type=int, default=8)
ap.add_argument("--dim", type=int, default=2)
ap.add_argument("--frames", type=int, default=0)
ap.add_argument("--movie-dir", default=None)
a = ap.parse_args()

kw = dict(cells_per_side=a.side, npc=a.npc, mach=a.mach, cells_per_mfp=a.cells_per_mfp)
_, _, _, _, _, info0 = setup(dim=a.dim, **kw)
rate = 4 * info0["nu"] * info0["kbox"] ** 2
n_steps = int(a.efolds / rate / info0["dt"])
print(f"target {a.efolds} e-folds -> {n_steps} steps  (rate {rate:.4e} 1/s)")

t0 = time.time()
mom, info = run(dim=a.dim, n_steps=n_steps, sample_stride=a.stride,
                frame_every=a.frames or None, movie_dir=a.movie_dir, **kw)
info["wall"] = time.time() - t0
print(f"  wall {info['wall']:.0f}s   clipped {info['clipped']}  violations {info['violations']}")
np.savez_compressed(a.out + ".npz", u=np.asarray(mom.u), nrho=np.asarray(mom.nrho),
                    temp=np.asarray(mom.temp), count=np.asarray(mom.count))
pickle.dump(info, open(a.out + ".pkl", "wb"))
if a.dim == 2:
    report_2d(mom, info)
