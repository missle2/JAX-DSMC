"""Per-cell correction features of all twelve cases, as the learned operator sees them (thinned
fine snapshots on the coarse grid; cells with two or more particles) -- the raw input of the
unsupervised analysis (scratch/lc_unsup.py). Same sampling as examples/learned_collision/features.py
(32 snapshots per case, seed 0), but keeps every cell instead of quantiles.

    JAX_ENABLE_X64=1 PYTHONPATH=. python scratch/lc_cellfeat.py

Writes examples/learned_collision/results/cellfeat_v1.npz (one f64[cells, 11] array per case).
"""

from __future__ import annotations

import numpy as np

from examples.learned_collision import flows
from examples.learned_collision.evaluate import load_ref
from examples.learned_collision.features import case_features
from examples.learned_collision.shear_wave import RESULTS


def main():
    rng = np.random.default_rng(0)
    refs, out = {}, {}
    for name, (ref_name, _, role) in flows.CASES.items():
        if ref_name not in refs:
            refs = {ref_name: load_ref(ref_name, "v1")}  # one reference in memory
        out[name] = case_features(name, *refs[ref_name], 32, rng)
        print(f"{name} ({role}): {out[name].shape[0]} cells", flush=True)
    np.savez(RESULTS / "cellfeat_v1.npz", **out)


if __name__ == "__main__":
    main()
