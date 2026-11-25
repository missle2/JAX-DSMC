#!/usr/bin/env python3
"""One benchmark point, either code: SPARTA (Kokkos or CPU) or sparta_jax.

    # SPARTA: run by this script (it builds the launch line and parses the log)
    python3 bench/run_bench.py sparta --exe $SPA --deck collide --particles 8M \\
        --launch "srun -n 16 --gpus-per-task 1" --accel "-k on g 1 -sf kk -pk kokkos gpu/aware yes" \\
        --machine pm-gpu --variant kk_cuda --nodes 4 --reps 3 --out results/pm-gpu

    # sparta_jax: launched BY srun, one process per GPU (jax.distributed when > 1 process)
    srun -n 16 --gpus-per-task 1 python3 bench/run_bench.py jax --deck collide --particles 8M \\
        --machine pm-gpu --variant jax --nodes 4 --reps 3 --out results/pm-gpu

Every rep writes a SPARTA-format log ``log.<date>.<deck>.<machine>_<variant>.<size>.<P>[.rN]``
(the naming of SPARTA's bench/ logs) and appends one JSON row to ``<out>/rows.jsonl``:
code, machine, variant, deck, particles, grid, nodes, procs, steps, loop time, Mpsps
(million particle-steps per second = particle moves / loop time) and, for sparta_jax, compile
and setup times and the per-rank imbalance.  ``bench/parse_logs.py`` rebuilds the same rows
from the logs alone.

Problem sizes: ``--particles`` is the nominal count (10 per cell, as the bench README); the grid
is ``sparta_jax.decks.size_xyz`` (the README's own grids for 10K..10M), or ``--xyz``.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMB]?)$", re.I)


def parse_size(s):
    m = SIZE_RE.match(s)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size {s!r}")
    mult = {"": 1, "K": 10**3, "M": 10**6, "B": 10**9}[m.group(2).upper()]
    return int(float(m.group(1)) * mult)


def size_label(n):
    for unit, m in (("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if n >= m and n % m == 0:
            return f"{n // m}{unit}"
    return str(n)


def log_name(deck, machine, variant, n, procs, rep):
    date = datetime.date.today().strftime("%d%b%y")
    return f"log.{date}.{deck}.{machine}_{variant}.{size_label(n)}.{procs}" + (f".r{rep}" if rep else "")


def git_rev(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def append_row(out, row):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "rows.jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------------------------
# SPARTA
# ---------------------------------------------------------------------------------------------

LOOP_RE = re.compile(r"Loop time of ([\d.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) particles")
MOVES_RE = re.compile(r"Particle moves\s+=\s+(\d+)")


def sparta_deck_file(deck):
    """bench/decks/in.<deck>: SPARTA's deck with only the timed run's length parametrised
    (``variable steps index <default>``), as SPARTA's own *.gpu.steps decks do."""
    return os.path.join(HERE, "decks", f"in.{deck}")


TOTALS = {
    "Particle moves": "particle_moves",
    "Boundary collides": "nbound",
    "Boundary exits": "nexit",
    "SurfColl checks": "nscheck",
    "SurfColl occurs": "nscoll",
    "Collide attempts": "nattempt",
    "Collide occurs": "ncoll",
    "Particles stuck": "nstuck",
}


def sparta_totals(text):
    """The last run's event totals ('Collide attempts  = 15417194 (15.4M)' ...)."""
    tail = text[text.rfind("Loop time of"):]
    out = {}
    for label_, key in TOTALS.items():
        m = re.search(re.escape(label_) + r"\s*=\s*(\d+)", tail)
        if m:
            out[key] = int(m.group(1))
    return out


def parse_sparta_log(text):
    loops = LOOP_RE.findall(text)
    moves = MOVES_RE.findall(text)
    if not loops:
        raise RuntimeError("no 'Loop time' line in the log")
    t, procs, steps, npart = loops[-1]
    pm = int(moves[-1]) if moves else None
    return {"loop_time": float(t), "procs": int(procs), "steps": int(steps), "np_end": int(npart), "particle_moves": pm,
            "totals": sparta_totals(text)}


def run_sparta(a):
    from sparta_jax import decks as D

    x, y, z = a.xyz or D.size_xyz(a.particles)
    n = 10 * x * y * z
    deckfile = sparta_deck_file(a.deck)
    for rep in range(a.reps):
        name = log_name(a.deck, a.machine, a.variant, n, a.procs, rep)
        path = os.path.join(a.out, "logs", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        steps = [] if a.steps is None else ["-v", "steps", str(a.steps)]
        cmd = (
            shlex.split(a.launch)
            + [a.exe, "-in", deckfile, "-log", path, "-screen", "none", "-v", "x", str(x), "-v", "y", str(y), "-v", "z", str(z)]
            + steps
            + ["-v", "seed", str(12345 + rep)]
            + shlex.split(a.accel)
        )
        print("+", " ".join(shlex.quote(c) for c in cmd), flush=True)
        if a.dry_run:
            continue
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=os.path.dirname(deckfile))
        wall = time.perf_counter() - t0
        if p.returncode != 0:
            print(f"SPARTA failed (exit {p.returncode}); see {path}", flush=True)
            continue
        with open(path) as fh:
            res = parse_sparta_log(fh.read())
        row = {
            "code": "sparta",
            "machine": a.machine,
            "variant": a.variant,
            "deck": a.deck,
            "particles": n,
            "grid": [x, y, z],
            "nodes": a.nodes,
            "procs": res["procs"],
            "steps": res["steps"],
            "loop_time": res["loop_time"],
            "particle_moves": res["particle_moves"],
            "mpsps": (res["particle_moves"] or n * res["steps"]) / res["loop_time"] / 1e6,
            "wall_s": wall,
            "totals": res["totals"],
            "rep": rep,
            "log": name,
            "command": " ".join(shlex.quote(c) for c in cmd),
            "host": socket.gethostname(),
            "rev": git_rev(os.path.dirname(os.path.dirname(a.exe))),
        }
        append_row(a.out, row)
        print(json.dumps({k: row[k] for k in ("deck", "particles", "procs", "loop_time", "mpsps")}), flush=True)


# ---------------------------------------------------------------------------------------------
# sparta_jax
# ---------------------------------------------------------------------------------------------


def init_jax(a):
    """jax.distributed when launched as several processes (srun); CPU collectives on CPU."""
    import jax

    nproc = int(os.environ.get("SLURM_NTASKS", "1"))
    if a.platform == "cpu" and nproc > 1:
        jax.config.update("jax_cpu_collectives_implementation", a.cpu_collectives)
    if nproc > 1:
        jax.distributed.initialize()
    if a.cache:
        jax.config.update("jax_compilation_cache_dir", a.cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return jax


def run_jax(a):
    jax = init_jax(a)
    import dataclasses

    import sparta_jax  # noqa: F401
    from sparta_jax import decks as D
    from sparta_jax import driver

    x, y, z = a.xyz or D.size_xyz(a.particles)
    deck = D.make(a.deck, x, y, z, steps=a.steps)
    n = deck.n_particles
    nrank = a.ranks or len(jax.devices())
    overrides = {}
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        overrides[k] = type_guess(v)
    me = jax.process_index()
    for rep in range(a.reps):
        buf = io.StringIO()
        t0 = time.perf_counter()
        if nrank == 1 and jax.process_count() == 1 and not a.force_decomp:
            res = driver.run_deck(deck, seed=12345 + rep, log=buf, **overrides)
        else:
            res = driver.run_deck_decomposed(deck, nrank, seed=12345 + rep, log=buf, headroom=a.headroom, **overrides)
        wall = time.perf_counter() - t0
        if me != 0:
            continue
        name = log_name(a.deck, a.machine, a.variant, n, nrank, rep)
        path = os.path.join(a.out, "logs", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(f"sparta_jax {git_rev(ROOT)} jax {jax.__version__} {jax.default_backend()} "
                     f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r} overrides={overrides}\n")
            fh.write(buf.getvalue())
        timed = [r for r in res["runs"] if r["timed"]][-1]
        row = {
            "code": "jax",
            "machine": a.machine,
            "variant": a.variant,
            "deck": a.deck,
            "particles": n,
            "grid": [x, y, z],
            "nodes": a.nodes,
            "procs": nrank,
            "processes": jax.process_count(),
            "steps": timed["nsteps"],
            "loop_time": timed["loop_time"],
            "particle_moves": timed["particle_moves"],
            "mpsps": res["mpsps"],
            "compile_s": res["compile_s"],
            "geometry_s": res["geometry_s"],
            "create_s": res["create_s"],
            "wall_s": wall,
            "imbalance": timed.get("imbalance", 1.0),
            "totals": timed["totals"],
            "layout": res.get("layout", [1, 1, 1]),
            "capacity": res["capacity"],
            "max_nlane": timed["max_nlane"],
            "device_kind": res["device_kind"],
            "backend": res["backend"],
            "jax": jax.__version__,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "overrides": overrides,
            "rep": rep,
            "log": name,
            "host": socket.gethostname(),
            "rev": git_rev(ROOT),
        }
        append_row(a.out, row)
        print(json.dumps({k: row[k] for k in ("deck", "particles", "procs", "loop_time", "mpsps", "compile_s")}), flush=True)


def type_guess(v):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("code", choices=("sparta", "jax"))
    p.add_argument("--deck", required=True, choices=("free", "collide", "sphere"))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--particles", type=parse_size, help="nominal count, e.g. 8M")
    g.add_argument("--xyz", type=int, nargs=3)
    p.add_argument("--steps", type=int, default=None, help="timed run length (default: the deck's)")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--machine", required=True, help="platform tag, e.g. pm-gpu, alp-h200")
    p.add_argument("--variant", default="default", help="build/runtime variant tag")
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--out", required=True)
    # sparta
    p.add_argument("--exe")
    p.add_argument("--procs", type=int, default=1, help="MPI ranks (SPARTA)")
    p.add_argument("--launch", default="", help="launcher prefix, e.g. 'srun -n 8'")
    p.add_argument("--accel", default="", help="SPARTA accelerator switches, e.g. '-k on g 1 -sf kk'")
    p.add_argument("--dry-run", action="store_true")
    # jax
    p.add_argument("--ranks", type=int, default=0, help="devices to decompose over (0 = all)")
    p.add_argument("--force-decomp", action="store_true", help="use decomp3d even on 1 device")
    p.add_argument("--headroom", type=float, default=None)
    p.add_argument("--set", action="append", help="sim3d.Config override key=value")
    p.add_argument("--platform", default=os.environ.get("JAX_PLATFORMS", "gpu"))
    p.add_argument("--cpu-collectives", default="gloo", choices=("gloo", "mpi"))
    p.add_argument("--cache", default=os.environ.get("JAX_COMPILATION_CACHE_DIR"))
    a = p.parse_args(argv)
    if a.code == "sparta":
        if not a.exe:
            p.error("--exe is required for sparta")
        run_sparta(a)
    else:
        run_jax(a)


if __name__ == "__main__":
    main()
