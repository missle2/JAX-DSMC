"""Would the topology guard have kept the DSMC runs' bodies whole? (2026-10-27)  Per run, the
knot lattice's face-connected inside pieces at every saved iterate, and for every step
``coef_k -> coef_{k+1}`` the guard's verdict against ``coef_k``: knots held inside / outside
and the pieces after.  A static replay (the run's later steps came from the unguarded body),
so it measures how often and how much the guard intervenes, not the guarded trajectory.

    PYTHONPATH=. python3 scratch/topo/replay.py <tag> [<tag> ...]
"""

import dataclasses
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

from jax_dsmc.mesh import topology as T  # noqa: E402

ROOT = Path(O.RESULTS)


def main():
    jax.config.update("jax_enable_x64", True)
    for tag in sys.argv[1:]:
        out = ROOT / tag
        cfg = dict(O.default_config(), **json.loads((out / "history.json").read_text())["config"])
        _grid, ls0 = O.initial_levelset(cfg)
        ks = sorted(int(p.stem[5:]) for p in out.glob("coef_*.npy"))
        prev = None
        rows = []
        for k in ks:
            ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(out / f"coef_{k}.npy")))
            v = O.knot_values(ls)
            if not np.isfinite(v).all():
                break
            row = {"k": k, "pieces": T.pieces(v)}
            if prev is not None:
                _out, counts = T.guard(prev, v, O.GUARD_MARGIN * float(ls.h))
                moved = int(np.sum((prev < 0) != (v < 0)))
                row.update(guarded=counts["pieces"], held_in=counts["held_inside"],
                           held_out=counts["held_outside"], moved=moved)  # fmt: skip
            rows.append(row)
            prev = v
        split = [r for r in rows if r.get("held_in") or r.get("held_out")]
        rises = [r["k"] for r, q in zip(rows[1:], rows[:-1], strict=False)
                 if r["pieces"] > q["pieces"]]  # fmt: skip
        print(f"{tag}: {len(rows)} iterates, max pieces {max(r['pieces'] for r in rows)}, "
              f"piece-count rises at {rises}")  # fmt: skip
        print(f"  guard intervened on {len(split)} steps; max pieces after guard "
              f"{max((r.get('guarded', 1) for r in rows), default=1)}")  # fmt: skip
        for r in split[:40]:
            print(f"    step {r['k'] - 1}->{r['k']}: unguarded pieces {r['pieces']}, held "
                  f"inside {r['held_in']}, held outside {r['held_out']} of {r['moved']} sign "
                  "changes")  # fmt: skip
        Path(f"scratch/topo/replay_{tag}.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
