"""Test-suite configuration.  Three decisions, all measured rather than assumed.

**1. The suite runs on the CPU by default.**  Every test here is a correctness test on
a deliberately tiny case, and for those XLA's GPU compilation dominates: the whole suite
takes 599 s on the GPU against 329 s on the CPU (1.8x), and a single three-case test that
bills 0.0 s inside a full run costs 39 s when it is the only thing that pays the compile.
Nothing in the suite asserts device throughput -- `test_perf.py` inspects jaxprs and
recompilation counts, which are platform-independent -- so the CPU loses no coverage.
Run `pytest --gpu` to check that the same tests compile and pass on the accelerator; that
is the only thing the GPU run adds, and it belongs before a milestone, not in the loop.

**This applies to the test suite only -- production runs belong on the GPU.**  The finding
above is about tiny cases where XLA compilation dominates; once a run is long enough to
amortise the compile, the ordering reverses hard.  Measured on the Couette gate
configuration (80 000 particles, 200 cells): **1.0 M particle-steps/s on the CPU against
5.5 M on the GPU**, a 5.3x gap, with the GPU compiling faster too (13.7 s vs 23.3 s).
`examples/couette` and `examples/shock` runners therefore set `JAX_PLATFORMS=cuda`.

**2. Four forced host devices.**  `test_decomp.py` needs `>= 2` devices and skips
otherwise.  On this box the GPU is a single device, so the GPU run *skips* the six
sharding tests; the CPU run with `--xla_force_host_platform_device_count=4` runs them.
The CPU default is therefore strictly more coverage, not less.

**3. A persistent XLA compilation cache.**  Measured on one test: 40.3 s cold, 8.1 s warm
(5x), for 1.8 MB of cache.  Compilation is the suite's dominant cost and it is pure
overhead on the second and later runs, which is the case that matters in a dev loop.
`--no-jax-cache` disables it; the cache is keyed on the jaxpr, so a code change
invalidates its own entries and a stale cache cannot mask a real change.

The environment must be set before *any* test module imports jax, which is why this
happens in `pytest_configure` and why this file imports no jax at module scope.
"""

from __future__ import annotations

import os

CACHE_DIR = ".pytest_cache/jax"


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="run on the accelerator instead of the CPU (pre-milestone check)",
    )
    parser.addoption(
        "--no-jax-cache",
        action="store_true",
        default=False,
        help="disable the persistent XLA compilation cache (forces cold compiles)",
    )


def pytest_configure(config):
    """Set the JAX environment before any test module imports jax."""
    if not config.getoption("--gpu"):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        flags = os.environ.get("XLA_FLAGS", "")
        if "xla_force_host_platform_device_count" not in flags:
            os.environ["XLA_FLAGS"] = (flags + " --xla_force_host_platform_device_count=4").strip()

    if not config.getoption("--no-jax-cache"):
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.abspath(CACHE_DIR))
        # default thresholds skip small/fast entries, which is most of this suite
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")


def pytest_report_header(config):
    where = "gpu" if config.getoption("--gpu") else "cpu (4 host devices)"
    cache = "off" if config.getoption("--no-jax-cache") else CACHE_DIR
    return f"jax-dsmc: platform={where}, compilation-cache={cache}"
