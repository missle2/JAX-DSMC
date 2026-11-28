"""Run a 3-D benchmark deck the way SPARTA's input runs it, and log it the way SPARTA logs it.

Each ``run`` of the deck executes in chunks of ``stats_every`` steps (the deck's ``stats N``):
one compiled chunk (``sim3d.run`` with traced dt and step0) serves every chunk of every run, and
the deck's stats columns (``sim3d.diagnostics``) are evaluated at the chunk boundaries, as SPARTA
evaluates them at its output steps.  The loop time of a run is the wall time from its first
chunk's dispatch to the device finishing its last, with the executable already compiled -- the
same span SPARTA's ``Loop time`` measures.

The log has SPARTA's stats block and ``Loop time of T on P procs for N steps with M particles``
line, so one parser reads both codes' logs (``bench/parse_logs.py``).
"""

from __future__ import annotations

import dataclasses
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import geometry3d as G3
from . import sim3d as S


@dataclasses.dataclass
class RunResult:
    dt: float
    nsteps: int
    timed: bool
    loop_time: float
    nparticles_end: int
    stats: dict  # per-step counters (numpy)
    rows: list  # stats rows at the output steps


def _stats_row(step, cpu, diag, last):
    return {
        "step": int(step),
        "cpu": float(cpu),
        "np": int(diag["np"]),
        "nattempt": int(last["nattempt"]),
        "ncoll": int(last["ncoll"]),
        "nscoll": int(last["nscoll"]),
        "temp": float(diag["temp"]),
        "maxpercell": int(diag["maxpercell"]),
    }


def setup(deck, seed=12345, geo=None, **cfg_overrides):
    """Geometry, config and the initial (created, sorted) state on one device."""
    t0 = time.perf_counter()
    geo = geo or G3.build_cached(deck)
    t_geo = time.perf_counter() - t0
    cfg, consts, geo = S.make(deck, geo, **cfg_overrides)
    t1 = time.perf_counter()
    state = S.create_particles(cfg, consts, geo, deck.temp, deck.vstream, seed)
    state = jax.block_until_ready(state)
    t_create = time.perf_counter() - t1
    return geo, cfg, consts, state, {"geometry_s": t_geo, "create_s": t_create}


# Compiled chunk executables, keyed by everything that fixes their shapes (the static Config,
# which holds the capacity, cell counts and table sizes, and the chunk length).  The runs CALL
# these executables, so a run compiles once -- before, the ahead-of-time compile used for the
# timing was thrown away and the first jitted call compiled again -- and further reps of the
# same size in the process (other seeds) compile nothing.
_EXECUTABLES = {}


def compile_chunk(state, consts, cfg, deck):
    """The chunk executable for (cfg, stats_every), compiled once per process; returns
    (executable, compile seconds -- 0 when it was already compiled)."""
    key = ("single", cfg, deck.stats_every)
    if key in _EXECUTABLES:
        return _EXECUTABLES[key], 0.0
    t0 = time.perf_counter()
    exe = S.run.lower(state, 0, deck.runs[0].dt, 0, consts, cfg, deck.stats_every).compile()
    _EXECUTABLES[key] = exe
    return exe, time.perf_counter() - t0


def run_deck(deck, seed=12345, log=None, **cfg_overrides):
    """All runs of the deck.  Returns a dict: timings, per-run results, config summary."""
    geo, cfg, consts, state, setup_t = setup(deck, seed, **cfg_overrides)
    exe, t_compile = compile_chunk(state, consts, cfg, deck)
    every = deck.stats_every
    out = {
        "deck": deck.name,
        "x": deck.x,
        "y": deck.y,
        "z": deck.z,
        "n_create": geo.n_create,
        "capacity": cfg.capacity,
        "lane": cfg.lane,
        "devices": 1,
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "compile_s": t_compile,
        **setup_t,
        "runs": [],
    }
    w = log.write if log is not None else (lambda s: None)
    w(f"sparta_jax 3-D deck {deck.name} {deck.x} {deck.y} {deck.z} on {out['device_kind']}\n")
    w(f"Created {geo.n_create} particles\n")
    step0 = 0
    for r in deck.runs:
        if r.vre_start and cfg.collide:
            state = S.reset_vremax(state, cfg)
        if r.nsteps % every:
            raise ValueError(f"run length {r.nsteps} is not a multiple of stats {every}")
        nchunk = r.nsteps // every
        diag0 = jax.device_get(S.diagnostics(state, cfg))
        jax.block_until_ready(state)
        chunks, diags = [], []
        t_start = time.perf_counter()
        for c in range(nchunk):
            state, st = exe(state, step0 + c * every, r.dt, seed, consts)
            chunks.append(st)
            diags.append(S.diagnostics(state, cfg))
        jax.block_until_ready((state, diags))
        loop = time.perf_counter() - t_start
        stats = {k: np.concatenate([np.asarray(getattr(st, k)) for st in chunks]) for k in S.Stats._fields}
        diags = jax.device_get(diags)
        # stats rows (cpu column: the wall time is only known at the end; spread uniformly)
        rows = [_stats_row(step0, 0.0, diag0, {k: 0 for k in stats})]
        for c in range(nchunk):
            last = {k: stats[k][(c + 1) * every - 1] for k in stats}
            rows.append(_stats_row(step0 + (c + 1) * every, loop * (c + 1) / nchunk, diags[c], last))
        w("Step CPU Np Natt Ncoll Nscoll temp maxpercell\n")
        for row in rows:
            w(
                f"{row['step']:8d} {row['cpu']:12.6g} {row['np']:9d} {row['nattempt']:9d} "
                f"{row['ncoll']:9d} {row['nscoll']:7d} {row['temp']:12.8g} {row['maxpercell']:6d}\n"
            )
        nend = int(rows[-1]["np"])
        w(f"Loop time of {loop:g} on 1 procs for {r.nsteps} steps with {nend} particles\n\n")
        bad = int(stats["overflow"].sum()), int(stats["emit_fail"].sum()), int(stats["nstuck"].sum())
        if bad[0]:
            raise RuntimeError(f"capacity/buffer overflow: {bad[0]} particles lost")
        out["runs"].append(
            {
                "dt": r.dt,
                "nsteps": r.nsteps,
                "timed": r.timed,
                "loop_time": loop,
                "np_end": nend,
                "particle_moves": S.moves(S.Stats(**{k: stats[k] for k in S.Stats._fields})),
                "totals": {k: int(v.sum()) for k, v in stats.items() if k != "np"},
                "max_nlane": int(stats["nlane"].max()),
                "max_nwalk": int(stats["nwalk"].max()),
                "rows": rows,
            }
        )
        step0 += r.nsteps
    timed = [r for r in out["runs"] if r["timed"]][-1]
    out["loop_time"] = timed["loop_time"]
    out["mpsps"] = timed["particle_moves"] / timed["loop_time"] / 1e6
    w(f"Particle moves = {timed['particle_moves']}\n")
    w(f"Particle-moves/CPUsec/proc: {timed['particle_moves'] / timed['loop_time']:.4g}\n")
    return out


def dump_json(result, path):
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ---------------------------------------------------------------------------------------------
# decomposed (multi-device / multi-process) runs
# ---------------------------------------------------------------------------------------------


def _to_host(tree):
    """Host copies of (possibly multi-process) arrays.  Per-rank stats are sharded over ranks,
    so with several processes each holds only its own ranks' part and np.asarray refuses
    ("spans non-addressable devices": every multi-process run, Perlmutter CPU tune 62133655):
    gather them (outside the timed loop)."""
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        return multihost_utils.process_allgather(tree, tiled=True)
    return jax.device_get(tree)


def run_deck_decomposed(deck, nrank, seed=12345, log=None, layout=None, headroom=None, **cfg_overrides):
    """All runs of the deck over ``nrank`` devices (``decomp3d``).  Same outputs as
    ``run_deck``, plus the layout and the per-rank particle-count imbalance."""
    from . import decomp3d as DD

    t0 = time.perf_counter()
    geo = G3.build_cached(deck)
    t_geo = time.perf_counter() - t0
    t0 = time.perf_counter()
    dec = DD.build(deck, nrank, geo=geo, layout=layout, headroom=headroom, **cfg_overrides)
    state = DD.create_particles(dec, seed)
    seeds = DD.rank_seeds(dec, seed)
    jax.block_until_ready(state)
    t_create = time.perf_counter() - t0
    every = deck.stats_every
    key = ("decomp", dec.cfg, dec.layout, dec.L, dec.M, every, nrank)
    t_compile = 0.0
    if key not in _EXECUTABLES:
        t0 = time.perf_counter()
        fn = DD._run_fn(dec.cfg, dec.layout, dec.L, dec.M, every, dec.mesh, DD._spec_key(dec.consts))
        _EXECUTABLES[key] = fn.lower(state, dec.consts, seeds, jnp.int32(0), jnp.float64(deck.runs[0].dt)).compile()
        t_compile = time.perf_counter() - t0
    exe = _EXECUTABLES[key]
    out = {
        "deck": deck.name,
        "x": deck.x,
        "y": deck.y,
        "z": deck.z,
        "n_create": geo.n_create,
        "capacity": dec.cfg.capacity,
        "lane": dec.cfg.lane,
        "devices": nrank,
        "processes": jax.process_count(),
        "layout": dec.layout.P,
        "bdim": dec.layout.bdim,
        "L": dec.L,
        "M": dec.M,
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "compile_s": t_compile,
        "geometry_s": t_geo,
        "create_s": t_create,
        "runs": [],
    }
    w = log.write if log is not None else (lambda s: None)
    w(f"sparta_jax 3-D deck {deck.name} {deck.x} {deck.y} {deck.z} on {nrank} x {out['device_kind']} layout {dec.layout.P}\n")
    w(f"Created {geo.n_create} particles\n")
    step0 = 0
    for r in deck.runs:
        if r.vre_start and dec.cfg.collide:
            state = DD.reset_vremax(dec, state)
        nchunk = r.nsteps // every
        diag0 = jax.device_get(DD.diagnostics(dec, state))
        jax.block_until_ready(state)
        chunks, diags = [], []
        t_start = time.perf_counter()
        for c in range(nchunk):
            state, st = exe(state, dec.consts, seeds, jnp.int32(step0 + c * every), jnp.float64(r.dt))
            chunks.append(st)
            diags.append(DD.diagnostics(dec, state))
        jax.block_until_ready((state, diags))
        loop = time.perf_counter() - t_start
        chunks = _to_host(chunks)
        per_rank = {k: np.concatenate([np.asarray(getattr(st, k)) for st in chunks]) for k in S.Stats._fields}
        stats = {k: (v.max(axis=1) if k in ("nlane", "nwalk") else v.sum(axis=1)) for k, v in per_rank.items()}
        diags = jax.device_get(diags)
        rows = [_stats_row(step0, 0.0, diag0, {k: 0 for k in stats})]
        for c in range(nchunk):
            last = {k: stats[k][(c + 1) * every - 1] for k in stats}
            rows.append(_stats_row(step0 + (c + 1) * every, loop * (c + 1) / nchunk, diags[c], last))
        w("Step CPU Np Natt Ncoll Nscoll temp maxpercell\n")
        for row in rows:
            w(
                f"{row['step']:8d} {row['cpu']:12.6g} {row['np']:9d} {row['nattempt']:9d} "
                f"{row['ncoll']:9d} {row['nscoll']:7d} {row['temp']:12.8g} {row['maxpercell']:6d}\n"
            )
        nend = int(rows[-1]["np"])
        w(f"Loop time of {loop:g} on {nrank} procs for {r.nsteps} steps with {nend} particles\n\n")
        if int(stats["overflow"].sum()):
            raise RuntimeError(f"capacity/buffer overflow: {int(stats['overflow'].sum())} particles lost")
        np_rank = per_rank["np"][-1]
        out["runs"].append(
            {
                "dt": r.dt,
                "nsteps": r.nsteps,
                "timed": r.timed,
                "loop_time": loop,
                "np_end": nend,
                "particle_moves": S.moves(S.Stats(**{k: stats[k] for k in S.Stats._fields})),
                "totals": {k: int(v.sum()) for k, v in stats.items() if k != "np"},
                "max_nlane": int(stats["nlane"].max()),
                "imbalance": float(np_rank.max() / max(np_rank.mean(), 1)),
                "np_rank_max": int(np_rank.max()),
                "rows": rows,
            }
        )
        step0 += r.nsteps
    timed = [r for r in out["runs"] if r["timed"]][-1]
    out["loop_time"] = timed["loop_time"]
    out["mpsps"] = timed["particle_moves"] / timed["loop_time"] / 1e6
    w(f"Particle moves = {timed['particle_moves']}\n")
    w(f"Particle-moves/CPUsec/proc: {timed['particle_moves'] / timed['loop_time'] / nrank:.4g}\n")
    return out
