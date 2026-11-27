#!/usr/bin/env python3
"""The scaling study as data: phases -> jobs -> plan files, with node-hour estimates.

    python3 bench/matrix.py strong --platform pm-gpu --nodes 1 2 4          # print the jobs + cost
    python3 bench/matrix.py strong --platform pm-gpu --nodes 1 2 4 --submit # write plans, sbatch
    python3 bench/matrix.py all --platform pm-gpu --budget                  # the whole study's cost

Phases:
  check     env check job (1 node): check_env.py on every device, multi-process
  validate  sparta_jax vs SPARTA at small size (8 seeds each), 1 GPU
  tune      runtime (and build-variant) sweeps on 1 node; bench/pick_best.py turns the results
            into bench/configs/<platform>.json, which later phases read
  single    per-device / per-node size sweep (32K .. memory), every deck, both codes
  strong    fixed totals (8M, 64M; + 512M on GPU) over --nodes
  weak      fixed per node (1M, 16M; + 128M on GPU) over --nodes

Estimates: per-run time = setup + reps * steps * particles / throughput, with the throughput
(million particle-steps / s per device) taken from measured rows in <results>/rows.jsonl when
there are any for the platform/code/deck (nearest size), else from DEFAULT_MPSPS.  Strong-scaling
points whose predicted time per rep exceeds --prune-seconds are dropped (the series then starts
at the smallest kept node count, as bench.html's series have missing points).  Every plan line
is wrapped in `timeout` at 3x its estimate + 5 min, so a hung run cannot eat a 32-node job.

Grouping: one job per (platform, phase, node count); runs inside a job go smallest first.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

NODES = (1, 2, 4, 8, 16, 32)
DECKS = ("free", "collide", "sphere")
STEPS_TOTAL = {"free": 130, "collide": 130, "sphere": 2000}  # equilibration + timed, as in the decks

# platform facts used by the planner (the scripts read platforms.sh; keep these in step)
PLATFORMS = {
    "pm-gpu": {"site": "nersc", "devices": 4, "cores": 64, "mem_gb": 40, "gpu": True},
    "pm-cpu": {"site": "nersc", "devices": 0, "cores": 128, "mem_gb": 512, "gpu": False},
    "alp-h200": {"site": "curc", "devices": 4, "cores": 64, "mem_gb": 141, "gpu": True},
    "alp-a100": {"site": "curc", "devices": 3, "cores": 64, "mem_gb": 40, "gpu": True},
    "alp-l40": {"site": "curc", "devices": 3, "cores": 64, "mem_gb": 48, "gpu": True},
    "alp-rtx6000": {"site": "curc", "devices": 4, "cores": 64, "mem_gb": 96, "gpu": True},
    "alp-mi100": {"site": "curc", "devices": 3, "cores": 48, "mem_gb": 32, "gpu": True},
}

# guesses until measured (million particle-steps / s per device; per NODE for CPU platforms)
# (2026-11-25 laptop RTX 3050 Ti, 1M particles: sparta_jax free 462, collide 72, sphere 24 --
# an A100 has ~8x its memory bandwidth and 30x its FP64 rate)
DEFAULT_MPSPS = {
    ("gpu", "sparta"): {"free": 3000.0, "collide": 800.0, "sphere": 500.0},
    ("gpu", "jax"): {"free": 2000.0, "collide": 350.0, "sphere": 150.0},
    ("cpu", "sparta"): {"free": 1200.0, "collide": 600.0, "sphere": 400.0},
    ("cpu", "jax"): {"free": 150.0, "collide": 40.0, "sphere": 20.0},
}
SETUP_S = {"sparta": 10.0, "jax": 90.0}  # per run: launch, geometry, creation, compile (first rep)
BYTES_PER_PARTICLE = {"sparta": 120.0, "jax": 260.0}  # device memory incl. temporaries (to be measured)


@dataclasses.dataclass
class Run:
    code: str
    deck: str
    particles: int
    nodes: int
    ranks_per_node: int
    variant: str = "default"
    env: dict = dataclasses.field(default_factory=dict)
    accel: str = ""
    extra: str = ""
    reps: int = 3
    steps: int | None = None
    sparta_build: str = "default"
    threads: int = 1  # OpenMP threads per rank (SPARTA Kokkos-OMP) / host devices per process (JAX CPU)
    wrap: str = ""  # command prefix: bench/env/with_mps.sh <nodes> --, bench/env/with_modules.sh <mods> --
    est_s: float = 0.0


@dataclasses.dataclass
class Job:
    platform: str
    phase: str
    nodes: int
    runs: list
    shared_gpus: int = 0  # > 0: a fraction of one node (NERSC shared QOS / one Alpine GPU)

    @property
    def name(self):
        return f"{self.phase}-{self.platform}-n{self.nodes}"

    @property
    def est_s(self):
        return sum(r.est_s for r in self.runs)

    @property
    def walltime_s(self):
        return min(self.est_s * 1.5 + 600, 12 * 3600)

    @property
    def node_hours(self):
        """Charged node-hours: NERSC charges the elapsed time (estimate + 2 min of job start),
        not the limit; a shared-QOS job is charged its fraction of the node."""
        frac = self.shared_gpus / PLATFORMS[self.platform]["devices"] if self.shared_gpus else 1.0
        return self.nodes * frac * (self.est_s + 120.0) / 3600.0


def label(n):
    for unit, m in (("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if n >= m and n % m == 0:
            return f"{n // m}{unit}"
    return str(n)


# ---------------------------------------------------------------------------------------------
# estimates
# ---------------------------------------------------------------------------------------------


def load_rows(results_dir):
    path = os.path.join(results_dir, "rows.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def throughput(platform, code, deck, particles_per_device, rows):
    """Mpsps per device (per node on CPU platforms), measured if possible."""
    pf = PLATFORMS[platform]
    kind = "gpu" if pf["gpu"] else "cpu"
    cand = [r for r in rows if r["code"] == code and r["deck"] == deck and r.get("machine", "").startswith(platform)]
    if cand:
        def per_dev(r):
            ndev = max(1, r["procs"] if pf["gpu"] else r.get("nodes", 1))
            return r["particles"] / ndev, r["mpsps"] / ndev

        best = min(cand, key=lambda r: abs(math.log(max(1.0, per_dev(r)[0]) / max(1.0, particles_per_device))))
        return per_dev(best)[1]
    return DEFAULT_MPSPS[(kind, code)][deck]


def estimate(run: Run, platform, rows):
    pf = PLATFORMS[platform]
    ndev = run.nodes * run.ranks_per_node if pf["gpu"] else run.nodes
    per_dev = run.particles / max(1, ndev)
    mpsps = throughput(platform, run.code, run.deck, per_dev, rows) * ndev
    steps = STEPS_TOTAL[run.deck] if run.steps is None else STEPS_TOTAL[run.deck] - (100 if run.deck != "sphere" else 1000) + run.steps
    per_rep = steps * run.particles / (mpsps * 1e6)
    setup = SETUP_S[run.code] + (run.particles / 1e8) * (20.0 if run.code == "jax" else 5.0)
    run.est_s = setup + run.reps * per_rep + (run.reps - 1) * (10.0 if run.code == "jax" else SETUP_S["sparta"])
    return per_rep


def fits_memory(run: Run, platform):
    pf = PLATFORMS[platform]
    if not pf["gpu"]:
        per = run.particles / run.nodes
        return per * BYTES_PER_PARTICLE[run.code] * 1.5 < pf["mem_gb"] * 1e9 * 0.8
    per = run.particles / (run.nodes * run.ranks_per_node)
    return per * BYTES_PER_PARTICLE[run.code] < pf["mem_gb"] * 1e9 * 0.85


# ---------------------------------------------------------------------------------------------
# plan lines
# ---------------------------------------------------------------------------------------------


def best_config(platform):
    path = os.path.join(HERE, "configs", f"{platform}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def default_accel(platform, cfg):
    """SPARTA/Kokkos switches.  Ranks see every GPU of their node (pf_launch nobind) and Kokkos
    picks one by local rank: `-k on g <devices per node>`."""
    pf = PLATFORMS[platform]
    if pf["gpu"]:
        return cfg.get("sparta", {}).get("accel", f"-k on g {pf['devices']} -sf kk")
    return cfg.get("sparta", {}).get("accel", "")


def plan_line(run: Run, platform):
    out = '"${RESULTS_DIR}"'
    env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in run.env.items())
    steps = "" if run.steps is None else f" --steps {run.steps}"
    common = (
        f"--deck {run.deck} --particles {label(run.particles)} --machine \"${{MACHINE}}\" "
        f"--variant {shlex.quote(run.variant)} --nodes {run.nodes} --reps {run.reps}{steps} --out {out}"
    )
    tmo = int(run.est_s * 3 + 300)
    gpu = PLATFORMS[platform]["gpu"]
    bind = "nobind" if (run.code == "sparta" and gpu) else ("node" if "_pernode" in run.variant else "bind")
    accel = run.accel
    if gpu and run.ranks_per_node == 1:  # one GPU (possibly a shared-QOS fraction of a node)
        bind = "bind"
        accel = re.sub(r"-k on g \d+", "-k on g 1", accel)
    launch = f"$(pf_launch {run.nodes} {run.ranks_per_node} {bind})"
    if run.code == "sparta":
        exe = f'"${{SPARTA_BUILD_DIR}}/sparta-${{PLATFORM}}-{run.sparta_build}/spa"'
        procs = run.nodes * run.ranks_per_node
        line = (
            f"{env} timeout {tmo} {run.wrap} python3 bench/run_bench.py sparta --exe {exe} {common} --procs {procs} "
            f"--launch \"{launch}\" --accel {shlex.quote(accel)}"
        )
    else:
        line = f"{env} timeout {tmo} {run.wrap} {launch} python3 bench/run_bench.py jax {common} {run.extra}"
        if not PLATFORMS[platform]["gpu"]:
            line += " --platform cpu"
    return line.strip()


# ---------------------------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------------------------


def reps_for(nodes):
    """3 reps up to 4 nodes (run-to-run spread), 2 at 8, 1 from 16 (the bench.html series are
    single runs)."""
    return 3 if nodes <= 4 else (2 if nodes <= 8 else 1)


def runs_strong(platform, nodes, rows, prune_s, codes, cfg, big=False):
    pf = PLATFORMS[platform]
    sizes = [8_000_000, 64_000_000] + ([512_000_000] if pf["gpu"] and big and nodes >= 8 else [])
    rpn = pf["devices"] if pf["gpu"] else cfg.get("cpu_ranks_per_node", pf["cores"])
    out = []
    for code in codes:
        for deck in DECKS:
            for n in sizes:
                if code == "jax" and not pf["gpu"] and n > 8_000_000:
                    continue  # JAX on CPU nodes: the 8M series only
                r = mk_run(platform, code, deck, n, nodes, rpn, cfg)
                r.reps = reps_for(nodes)
                if not fits_memory(r, platform):
                    continue
                per_rep = estimate(r, platform, rows)
                if per_rep > prune_s:
                    continue
                out.append(r)
    return out


def runs_weak(platform, nodes, rows, prune_s, codes, cfg, big=False):
    pf = PLATFORMS[platform]
    per_node = [1_000_000, 16_000_000] + ([128_000_000] if pf["gpu"] and big and nodes <= 8 else [])
    rpn = pf["devices"] if pf["gpu"] else cfg.get("cpu_ranks_per_node", pf["cores"])
    out = []
    for code in codes:
        for deck in DECKS:
            for pn in per_node:
                r = mk_run(platform, code, deck, pn * nodes, nodes, rpn, cfg)
                r.reps = reps_for(nodes)
                if fits_memory(r, platform) and estimate(r, platform, rows) <= 2 * prune_s:
                    out.append(r)
    return out


def runs_single(platform, rows, codes, cfg, per_device):
    """Per-device sweep (1 rank: ``per_device``) or per-node sweep (all devices): sizes doubling
    from 32K (per-device) / 1M (per-node) to the memory limit, reps 2, at most 300 s a rep."""
    pf = PLATFORMS[platform]
    out = []
    rpn = 1 if per_device else (pf["devices"] if pf["gpu"] else cfg.get("cpu_ranks_per_node", pf["cores"]))
    first = 32_000 if per_device else 1_000_000
    sizes = [first * 2**k for k in range(0, 16)]
    for code in codes:
        for deck in DECKS:
            for n in sizes:
                r = mk_run(platform, code, deck, n, 1, rpn, cfg)
                r.reps = 2
                if not fits_memory(r, platform):
                    break
                if estimate(r, platform, rows) > 300:
                    break
                out.append(r)
    return out


GPU_STEPS = {"free": 1000, "collide": 1000, "sphere": None}  # SPARTA's *.gpu.steps practice


def mk_run(platform, code, deck, n, nodes, rpn, cfg):
    r = _mk_run(platform, code, deck, n, nodes, rpn, cfg)
    if PLATFORMS[platform]["gpu"]:
        r.steps = GPU_STEPS[deck]
    return r


def _mk_run(platform, code, deck, n, nodes, rpn, cfg):
    pf = PLATFORMS[platform]
    if code == "sparta":
        sc = cfg.get("sparta", {})
        # tuned: ranks per GPU (MPS) on multi-GPU runs, and a wrapper (MPS daemon / PrgEnv)
        if pf["gpu"] and rpn > 1:
            rpn = rpn * int(sc.get("ranks_per_device", 1))
        return Run(
            code="sparta",
            deck=deck,
            particles=n,
            nodes=nodes,
            ranks_per_node=rpn,
            variant=sc.get("tag", "kk" if pf["gpu"] else "mpi"),
            accel=default_accel(platform, cfg),
            sparta_build=sc.get("build", "default" if pf["gpu"] else "mpi"),
            env=dict(sc.get("env", {})),
            wrap=sc.get("wrap", "").replace("{nodes}", str(nodes)),
        )
    jc = cfg.get("jax", {})
    if pf["gpu"]:
        return Run(code="jax", deck=deck, particles=n, nodes=nodes, ranks_per_node=rpn, variant=jc.get("tag", "jax"),
                   env=dict(jc.get("env", {})), extra=jc.get("extra", ""))
    # CPU: processes per node x host devices per process (the tuned layout)
    ppn, dpp = jc.get("cpu_layout", [8, 16])
    env = dict(jc.get("env", {}))
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={dpp}").strip()
    return Run(code="jax", deck=deck, particles=n, nodes=nodes, ranks_per_node=ppn, variant=jc.get("tag", "jaxcpu"),
               env=env, extra=jc.get("extra", ""), threads=dpp)


def runs_tune(platform, rows, cfg):
    """One node: the knobs of bench/README.md section 'Tuning', at a size that fills a device."""
    pf = PLATFORMS[platform]
    out = []
    # 16.384M = 32K x 2^9, a size of the single-device sweep: the tuned variant's rows double as
    # that sweep point (dedupe in build_jobs), and the step counts are the sweep's (GPU_STEPS)
    n_dev = 16_384_000 if pf["gpu"] else 32_000_000
    rpn = pf["devices"] if pf["gpu"] else pf["cores"]
    decks = ("collide", "sphere")
    if pf["gpu"]:
        # --- SPARTA/Kokkos.  (a) build variants, baseline switches, 1 GPU and the full node:
        #   default      Kokkos 5 defaults (CUDA malloc-async OFF: "CUDA malloc async support
        #                disabled" in the build logs)
        #   mallocasync  Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON
        #   march        host code -march=native
        #   bigbig       SPARTA_SIZES=bigbig (64-bit ids)
        # (b) runtime, default build: comm serial, gpu/aware no (between ranks: node only),
        #     particle/reorder 10 and 100 (a deck setting: a labelled variant), 2 ranks per GPU
        #     under MPS (node).  Single-GPU platforms (Alpine) skip it: the sweep has the baseline.
        if not single_gpu_only(platform):
            dev = pf["devices"]
            base = f"-k on g {dev} -sf kk"
            for build in ("default", "mallocasync", "march", "bigbig"):
                for deck in DECKS:
                    for r_ in (1, rpn):
                        out.append(Run("sparta", deck, n_dev * r_, 1, r_, variant=f"kk_{build}", accel=base,
                                       sparta_build=build, reps=2))
            for tag, accel, gpus in (
                ("kk_commserial_default", base + " -pk kokkos comm serial", (rpn,)),
                ("kk_noaware_default", base + " -pk kokkos gpu/aware no", (rpn,)),
                ("kk_reorder10_default", base + " -v reorder 10", (1, rpn)),
                ("kk_reorder100_default", base + " -v reorder 100", (1, rpn)),
            ):
                for deck in DECKS:
                    for r_ in gpus:
                        out.append(Run("sparta", deck, n_dev * r_, 1, r_, variant=tag, accel=accel, reps=2))
            for deck in DECKS:
                out.append(Run("sparta", deck, n_dev * rpn, 1, 2 * rpn, variant="kk_mps2_default", accel=base,
                               reps=2, wrap="bash bench/env/with_mps.sh 1 --"))
            for r in out:
                estimate(r, platform, rows)
        # --- sparta_jax: XLA flags, allocator, sort, lanes (1 GPU), then process layout (node)
        xla = {
            "jax": "",
            "jax_nocmdbuf": "--xla_gpu_enable_command_buffer=",
            "jax_cmdbuf_min1": "--xla_gpu_graph_min_graph_size=1",
            "jax_autotune0": "--xla_gpu_autotune_level=0",
            "jax_wlunroll": "--xla_gpu_enable_while_loop_unrolling=WHILE_LOOP_UNROLLING_AUTO_UNROLL",
            "jax_nocub": "--xla_gpu_enable_cub_radix_sort=false",
        }
        for tag, flags in xla.items():
            for deck in decks:
                r = Run("jax", deck, n_dev, 1, 1, variant=tag, env={"XLA_FLAGS": flags} if flags else {}, reps=2)
                estimate(r, platform, rows)
                out.append(r)
        for tag, env, extra in (
            ("jax_platformalloc", {"XLA_PYTHON_CLIENT_ALLOCATOR": "platform", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, ""),
            ("jax_mallocasync", {"TF_GPU_ALLOCATOR": "cuda_malloc_async", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, ""),
            ("jax_lane2x", {}, "--set lane_scale=2"),
            ("jax_lanehalf", {}, "--set lane_scale=0.5"),
            ("jax_trials16", {}, "--set trials=16"),
            ("jax_sortpairs", {}, "--set sort=full"),
            ("jax_startsearch", {}, "--set starts=search"),
            ("jax_cprefix", {}, "--set collide_mode=prefix"),
            ("jax_cunroll4", {}, "--set collide_unroll=4"),
            ("jax_cunroll8", {}, "--set collide_unroll=8"),
        ):
            for deck in decks:
                r = Run("jax", deck, n_dev, 1, 1, variant=tag, env=env, extra=extra, reps=2)
                estimate(r, platform, rows)
                out.append(r)
        # multi-GPU on one node: NCCL settings and the latency-hiding scheduler
        for tag, env in (
            ("jax_node", {}),
            ("jax_node_lhs", {"XLA_FLAGS": "--xla_gpu_enable_latency_hiding_scheduler=true"}),
            ("jax_node_pernode", {}),  # one process driving every GPU of the node
        ):
            for deck in decks:
                r = Run("jax", deck, n_dev * rpn, 1, rpn, variant=tag, env=env, reps=2)
                estimate(r, platform, rows)
                out.append(r)
    else:
        n = 32_000_000
        # --- SPARTA CPU: MPI-only vs Kokkos OMP layouts vs Kokkos serial, with/without SMT
        cce = "bash bench/env/with_modules.sh PrgEnv-cray --"
        for build, tag, ranks, threads, accel, wrap in (
            ("mpi", "mpi128", 128, 1, "", ""),
            ("mpi", "mpi256smt", 256, 1, "", ""),
            ("mpi_march", "mpi128_march", 128, 1, "", ""),
            ("mpi_cce", "mpi128_cce", 128, 1, "", cce),
            ("mpi", "mpi128_reorder100", 128, 1, "-v reorder 100", ""),
            ("kokkos_serial", "kkser128", 128, 1, "-k on -sf kk", ""),
            ("kokkos_omp", "kkomp64x2", 64, 2, "-k on t 2 -sf kk", ""),
            ("kokkos_omp", "kkomp32x4", 32, 4, "-k on t 4 -sf kk", ""),
            ("kokkos_omp", "kkomp16x8", 16, 8, "-k on t 8 -sf kk", ""),
            ("kokkos_omp", "kkomp8x16", 8, 16, "-k on t 16 -sf kk", ""),
        ):
            for deck in DECKS:
                r = Run("sparta", deck, n, 1, ranks, variant=tag, accel=accel, sparta_build=build, reps=2, threads=threads,
                        env={"OMP_NUM_THREADS": threads}, wrap=wrap)
                estimate(r, platform, rows)
                out.append(r)
        # --- sparta_jax CPU: processes x host devices per process, XLA:CPU flags
        nj = 8_000_000
        for ppn, dpp in ((1, 128), (2, 64), (16, 8), (8, 16), (128, 1), (1, 1)):
            for deck in decks:
                env = {"XLA_FLAGS": f"--xla_force_host_platform_device_count={dpp}"}
                size = nj if ppn * dpp > 1 else 1_000_000
                r = Run("jax", deck, size, 1, ppn, variant=f"jaxcpu_{ppn}x{dpp}", env=env, reps=1, threads=dpp)
                estimate(r, platform, rows)
                out.append(r)
        for tag, flags in (
            ("jaxcpu_nothunk", "--xla_cpu_use_thunk_runtime=false"),
            ("jaxcpu_noeigen", "--xla_cpu_multi_thread_eigen=false"),
            ("jaxcpu_concsched", "--xla_cpu_enable_concurrency_optimized_scheduler=true"),
            ("jaxcpu_split8", "--xla_cpu_parallel_codegen_split_count=8"),
        ):
            for deck in decks:
                env = {"XLA_FLAGS": f"--xla_force_host_platform_device_count=16 {flags}"}
                r = Run("jax", deck, nj, 1, 8, variant=tag, env=env, reps=1, threads=16)
                estimate(r, platform, rows)
                out.append(r)
    if pf["gpu"]:
        for r in out:
            r.steps = GPU_STEPS[r.deck]
            estimate(r, platform, rows)
    return out


def runs_validate(platform):
    """sparta_jax at the bench README sizes 10K and 100K, 8 seeds, for validate.py (SPARTA's
    side runs on any CPU: bench/validate.py --sparta)."""
    out = []
    for deck in DECKS:
        for n in (10_000, 100_000):
            out.append(Run("jax", deck, n, 1, 1, variant="validate", reps=8, est_s=0.0))
    return out


def single_gpu_only(platform):
    """Alpine: one GPU per job, never more (2026-11-27: multi-GPU jobs do not get through the
    gpu-normal queue, and Alpine's role in the study is the single-GPU comparison across
    architectures; multi-GPU and multi-node scaling is Perlmutter's)."""
    return PLATFORMS[platform]["site"] == "curc"


DECK_STEPS = {"free": 100, "collide": 100, "sphere": 1000}  # the decks' timed run length


# The JAX version the study's sparta_jax rows are for (2026-11-27: 0.12.1; the MI100 is held at
# AMD's ROCm 6.4.4 build).  Rows of another JAX version do not count as done.
JAX_HEADLINE = {"alp-mi100": "0.4.35"}


def jax_headline(platform):
    return os.environ.get("BENCH_JAX_VERSION") or JAX_HEADLINE.get(platform, "0.12.1")


def done_reps(run: Run, platform, rows):
    """Rows already recorded for this exact run (code, variant, deck, size, procs, steps; for
    sparta_jax also the JAX version)."""
    steps = run.steps or DECK_STEPS[run.deck]
    procs = run.nodes * run.ranks_per_node
    jv = jax_headline(platform)
    return sum(
        1
        for r in rows
        if r.get("machine", "").startswith(platform)
        and (run.code != "jax" or r.get("jax") == jv)
        and r["code"] == run.code
        and r["variant"] == run.variant
        and r["deck"] == run.deck
        and r["particles"] == run.particles
        and r["procs"] == procs
        and r["steps"] == steps
    )


def dedupe(jobs, platform, rows):
    """Drop the runs whose reps are all recorded already (a resubmitted or overlapping phase)."""
    skipped = 0
    for j in jobs:
        keep = []
        for r in j.runs:
            if done_reps(r, platform, rows) >= r.reps:
                skipped += 1
            else:
                keep.append(r)
        j.runs = keep
    if skipped:
        print(f"{skipped} run(s) already recorded in the results: skipped (--redo runs them again)")
    return [j for j in jobs if j.runs or j.phase == "check"]


def build_jobs(phase, platform, nodes, rows, prune_s, codes, big=False):
    cfg = best_config(platform)
    jobs = []
    one_gpu = single_gpu_only(platform)
    if phase == "check":
        jobs.append(Job(platform, "check", 1, [], shared_gpus=1 if one_gpu else 0))
    elif phase == "validate":
        runs = runs_validate(platform)
        for r in runs:
            estimate(r, platform, rows)
        jobs.append(Job(platform, "validate", 1, runs, shared_gpus=1 if PLATFORMS[platform]["gpu"] else 0))
    elif phase == "tune":
        runs = runs_tune(platform, rows, cfg)
        one = [r for r in runs if r.ranks_per_node == 1 and PLATFORMS[platform]["gpu"]]
        node = [r for r in runs if r not in one]
        if one:
            jobs.append(Job(platform, "tune1", 1, one, shared_gpus=1))
        if not one_gpu:
            jobs.append(Job(platform, "tune", 1, node))
    elif phase == "single":
        pf = PLATFORMS[platform]
        jobs.append(Job(platform, "single1", 1, runs_single(platform, rows, codes, cfg, True), shared_gpus=1 if pf["gpu"] else 0))
        if not one_gpu:
            jobs.append(Job(platform, "singlenode", 1, runs_single(platform, rows, codes, cfg, False)))
    elif phase in ("strong", "weak") and one_gpu:
        print(f"{platform}: single-GPU only (no {phase} scaling on Alpine)")
    elif phase in ("strong", "weak"):
        for n in nodes:
            fn = runs_strong if phase == "strong" else runs_weak
            runs = fn(platform, n, rows, prune_s, codes, cfg, big=big)
            if runs:
                jobs.append(Job(platform, phase, n, sorted(runs, key=lambda r: (r.particles, r.code))))
    return jobs


# ---------------------------------------------------------------------------------------------
# output / submission
# ---------------------------------------------------------------------------------------------


def write_plan(job: Job, plan_dir):
    os.makedirs(plan_dir, exist_ok=True)
    path = os.path.join(plan_dir, job.name + ".plan")
    with open(path, "w") as fh:
        fh.write(f"# {job.name}: {len(job.runs)} runs, estimate {job.est_s / 60:.1f} min\n")
        if job.phase == "check":
            fh.write("python3 bench/env/check_env.py\n")
            # SPARTA: the binary finds its libraries (gcc/openmpi modules) and runs on the GPU
            spa = '"${SPARTA_BUILD_DIR}/sparta-${PLATFORM}-default/spa"'
            fh.write(f'echo "ldd spa:"; ldd {spa} | grep -E "not found|libmpi|libstdc" || true\n')
            if PLATFORMS[job.platform]["gpu"]:
                fh.write(
                    f'python3 bench/run_bench.py sparta --exe {spa} --deck collide --particles 10K --steps 20 --reps 1 '
                    f'--machine "${{MACHINE}}" --variant check --out "${{RESULTS_DIR}}/check" '
                    f'--launch "$(pf_launch 1 1 bind)" --accel "-k on g 1 -sf kk"\n'
                )
            if not single_gpu_only(job.platform):
                fh.write(f"NCCL_DEBUG=INFO $(pf_launch 1 {max(1, PLATFORMS[job.platform]['devices'])}) python3 bench/env/check_env.py --distributed\n")
        for r in job.runs:
            fh.write(f"# est {r.est_s:.0f} s\n{plan_line(r, job.platform)}\n")
    return path


def hms(s):
    s = int(math.ceil(s / 60.0)) * 60
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:00"


def sbatch_args(platform, nodes, shared_gpus=0):
    env = dict(os.environ, PLATFORM=platform)
    fn = f"pf_resources_shared {shared_gpus}" if shared_gpus else f"pf_resources {nodes}"
    cmd = f"source {HERE}/env/platforms.sh >/dev/null && {fn}"
    return subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True, check=True).stdout.strip()


def default_results():
    """RESULTS_DIR, else the cluster's results root (Alpine /projects, NERSC $PSCRATCH), else
    bench/results.  The planner must see the study's rows to skip finished runs."""
    if os.environ.get("RESULTS_DIR"):
        return os.environ["RESULTS_DIR"]
    for base in (f"/projects/{os.environ.get('USER', '')}", os.environ.get("PSCRATCH", "")):
        d = os.path.join(base, "sparta-scaling", "results")
        if base and os.path.isdir(d):
            return d
    return os.path.join(ROOT, "bench", "results")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("check", "validate", "tune", "single", "strong", "weak", "all"))
    p.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    p.add_argument("--nodes", type=int, nargs="+", default=list(NODES))
    p.add_argument("--codes", nargs="+", default=["sparta", "jax"])
    p.add_argument("--prune-seconds", type=float, default=300.0)
    p.add_argument("--results", default=default_results(),
                   help="results root (rows of <root>/<platform>/rows.jsonl are read: estimates and dedupe)")
    p.add_argument("--plans", default=os.path.join(ROOT, "bench", "plans"))
    p.add_argument("--big", action="store_true", help="add the 512M strong / 128M-per-node weak GPU series")
    p.add_argument("--redo", action="store_true", help="also run what the results already hold")
    p.add_argument("--after", default="", help="sbatch --dependency=afterok:<jobid>[:<jobid>...]")
    p.add_argument("--only", default="", help="regex: keep only the jobs whose name matches (e.g. '^tune-')")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--list", action="store_true", help="print every run")
    a = p.parse_args()
    rows = []
    for d in {a.results, os.path.join(a.results, a.platform)}:
        rows += load_rows(d)
    phases = ("check", "validate", "tune", "single", "strong", "weak") if a.phase == "all" else (a.phase,)
    if single_gpu_only(a.platform) and a.phase == "all":
        phases = ("check", "validate", "tune", "single")
    jobs = []
    for ph in phases:
        jobs += build_jobs(ph, a.platform, a.nodes, rows, a.prune_seconds, a.codes, big=a.big)
    if not a.redo:
        jobs = dedupe(jobs, a.platform, rows)
    if a.only:
        jobs = [j for j in jobs if re.search(a.only, j.name)]
    total = 0.0
    print(f"{'job':34s} {'nodes':>5s} {'runs':>5s} {'est':>8s} {'limit':>8s} {'node-h':>8s}")
    for j in jobs:
        total += j.node_hours
        print(f"{j.name:34s} {j.nodes:5d} {len(j.runs):5d} {j.est_s / 60:7.1f}m {hms(j.walltime_s):>8s} {j.node_hours:8.2f}")
        if a.list:
            for r in j.runs:
                print(f"    {r.code:6s} {r.deck:7s} {label(r.particles):>5s} rpn {r.ranks_per_node:3d} {r.variant:22s} {r.est_s:7.0f}s")
    print(f"{'total':34s} {'':5s} {sum(len(j.runs) for j in jobs):5d} {'':8s} {'':8s} {total:8.2f}")
    if a.submit:
        for j in jobs:
            plan = write_plan(j, os.path.join(a.plans, a.platform))
            res = sbatch_args(a.platform, j.nodes, j.shared_gpus)
            dep = [f"--dependency=afterok:{a.after}"] if a.after else []
            cmd = ["sbatch", *shlex.split(res), *dep, f"--time={hms(j.walltime_s)}", f"--job-name={j.name}",
                   f"--export=ALL,PLATFORM={a.platform},PLAN={plan}", os.path.join(HERE, "jobs", "bench.sbatch")]
            print("+", " ".join(shlex.quote(c) for c in cmd))
            subprocess.run(cmd, check=True)
    elif jobs:
        for j in jobs:
            write_plan(j, os.path.join(a.plans, a.platform))
        print(f"plans written to {os.path.join(a.plans, a.platform)} (not submitted; --submit to sbatch)")


if __name__ == "__main__":
    main()
