"""Unit check of emit/face against kinetic theory (drifting Maxwellian through the xlo face).

Theory (Bird 1994 sec. 4.3, 12.1), s = U / vscale, vscale = sqrt(2 k T / m):
  * insertions per task per step: ntarget = nrho dy dt / fnum * vscale (e^{-s^2} + sqrt(pi) s
    (1 + erf s)) / (2 sqrt(pi))  (the random rounding int(ntarget + U) is unbiased);
  * normal velocity of an entrant: density proportional to vx exp(-(vx - U)^2 / vscale^2), vx > 0;
  * tangential components: Gaussian, mean vstream, variance vscale^2 / 2;
  * dtremain: uniform on [0, dt); y: uniform over the task face.

    run_locked.sh cpu python3 examples/sparta_jax/check_emit.py
"""

import functools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from sparta_jax import sim  # noqa: E402


def ks_stat(samples, cdf):
    x = np.sort(samples)
    n = len(x)
    f = cdf(x)
    return float(max(np.max(np.arange(1, n + 1) / n - f), np.max(f - np.arange(n) / n)))


def normal_cdf_table(U, vs):
    """Numerical CDF of p(vx) ~ vx exp(-(vx - U)^2 / vs^2) on vx > 0."""
    v = np.linspace(0.0, U + 8 * vs, 200001)
    p = v * np.exp(-((v - U) ** 2) / vs**2)
    c = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(v))])
    c /= c[-1]
    return functools.partial(np.interp, xp=v, fp=c)


@functools.partial(jax.jit, static_argnames="cfg")
def _emit_many(steps, consts, cfg):
    st = sim.empty_state(cfg)
    fn = functools.partial(sim.emit, consts=consts, cfg=cfg)
    out = jax.vmap(lambda k: fn(st, 99, k))(steps)
    s, (slot, dtr), n_in, fail, _ov = out
    take = jax.vmap(lambda a, i: a[jnp.minimum(i, cfg.capacity - 1)])
    return {
        "valid": slot < cfg.capacity,
        "vx": take(s.vx, slot),
        "vy": take(s.vy, slot),
        "vz": take(s.vz, slot),
        "y": take(s.y, slot),
        "dtr": dtr,
        "cell": take(s.cell, slot) - (cfg.ncell + 1),  # emit codes inserts near
        "n_in": n_in,
        "fail": fail,
    }


def main():
    cfg, consts, geo = sim.make(capacity=4096)
    nstep = 4000
    out = {}
    for c0 in range(0, nstep, 500):
        r = _emit_many(jnp.arange(c0, c0 + 500, dtype=jnp.int32), consts, cfg)
        for k, v in r.items():
            out.setdefault(k, []).append(np.asarray(v))
    out = {k: np.concatenate(v) for k, v in out.items()}
    m = out["valid"]
    vs, U = cfg.vscale, cfg.vstream[0]
    nt = float(geo.task_ntarget.sum())
    counts = out["n_in"].astype(float)
    vx, vy, vz = out["vx"][m], out["vy"][m], out["vz"][m]
    # per-task position check: y relative to its task face
    cell = out["cell"][m]
    j = cell // cfg.nx
    yrel = (out["y"][m] - (cfg.ylo + j * cfg.dy)) / cfg.dy
    vv = np.linspace(0.0, U + 8 * vs, 200001)
    pv = vv * np.exp(-((vv - U) ** 2) / vs**2)
    mean_vx_theory = float(np.sum(vv * pv) / np.sum(pv))
    n = len(vx)
    res = {
        "steps": nstep,
        "particles": int(n),
        "emit_fail": int(out["fail"].sum()),
        "count_per_step": [float(counts.mean()), float(counts.std() / math.sqrt(nstep))],
        "count_theory": nt,
        "count_z": float((counts.mean() - nt) / (counts.std() / math.sqrt(nstep))),
        "mean_vx": [float(vx.mean()), float(vx.std() / math.sqrt(n))],
        "mean_vx_theory": mean_vx_theory,
        "ks_vx": ks_stat(vx, normal_cdf_table(U, vs)),
        "var_vy_over_theory": float(vy.var() / (vs * vs / 2)),
        "var_vz_over_theory": float(vz.var() / (vs * vs / 2)),
        "mean_vy_over_vscale": float(vy.mean() / vs),
        "ks_vy": ks_stat(vy / (vs / math.sqrt(2)), _std_normal_cdf),
        "ks_y_in_face": ks_stat(yrel, np.asarray),
        "ks_dtremain": ks_stat(out["dtr"][m] / cfg.dt, np.asarray),
        "ks_5sigma": 5.0 * 0.87 / math.sqrt(n),  # ~5 sigma band of the KS statistic
    }
    with open(os.path.join(HERE, "results", "check_emit.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


def _std_normal_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


if __name__ == "__main__":
    main()
