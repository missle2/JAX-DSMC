"""Cost budgets: would we notice if a change made the solver dramatically slower?

`test_perf.py` asserts the *shape* of the compiled program -- no `while_loop`, one
scan, no RNG inside it, no recompilation.  Those catch known causes.  This file catches
the symptom: a change that leaves the structure intact and still does far more work.

**Why not wall-clock.**  `test_perf.py`'s docstring is right that a timing threshold is
flaky on shared hardware -- this box shows ~1.6x run-to-run variance on the full suite.
So the primary budgets here are on XLA's own `cost_analysis()` and `memory_analysis()`,
which describe the compiled HLO and are **bit-identical across runs**: same flops, same
bytes, same temp allocation, every time.  A 5x slower kernel that kept the same jaxpr
shape would move these; noise does not.

**Baselines are per backend** and differ a lot (the forward solve is 9.6e5 flops on CPU
and 2.3e6 on GPU -- XLA counts a fused GPU kernel differently), so `BUDGETS` is keyed on
`jax.default_backend()`.  Bands are deliberately wide: this is a "did something dramatic
happen" gate, not a micro-benchmark.  Refresh with

    python3 -m tests.test_perf_budget --print-baseline        # on each backend

The ratio tests below are backend-independent and are the more durable of the two: they
pin the reverse-pass claims of `PSEUDOCODE.md` section 23.4 directly.

**Compile time is deliberately NOT asserted here -- check it by hand instead:**

    python3 -m tests.test_perf_budget --compile-time      # add JAX_PLATFORMS=cuda for GPU

Two reasons.  It is the one quantity that cannot be measured reproducibly in a test: JAX
caches compiled programs *in process*, so only the first `compile()` is real (5.28 s, then
0.00 s three times running), and the persistent cache in `conftest.py` makes even that a
deserialisation on a warm run.  And it does not need a gate -- a compile-time blow-up
shows up immediately in any real run, especially one that sweeps `A_max`, because that is
what recompiles.  Worth running the command above occasionally, and after any change to
the attempt-slot scan; for reference, drawing RNG per slot once cost 7x in compile time
(spec section 8.6).
"""

from __future__ import annotations

import sys
import time

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from examples import shock_adjoint as sa  # noqa: E402

TOL = 0.35
"""Fractional band on the absolute budgets.  Wide on purpose -- see the module docstring."""

BUDGETS = {
    # backend: {label: (flops, bytes_accessed, temp_bytes)}
    "cpu": {
        "forward": (9.607770e05, 1.580740e06, 0.200e6),
        "grad": (1.882729e06, 5.314315e06, 1.314e6),
    },
    "gpu": {
        "forward": (2.286232e06, 5.276180e05, 0.158e6),
        "grad": (4.139877e06, 2.937167e06, 1.182e6),
    },
}


def _case():
    """The same tiny D=2 case `test_adjoint.py` uses, so the budgets track that config."""
    return sa.case(2, npc=20, n_steps=4, n_x=8, transverse=2, a_max=128)


def _forward(theta, key):
    return sa.value_and_grad(theta, _case(), key, False)[0]


def _grad(theta, key):
    """Value **and** gradient: returning only the value lets XLA delete the backward pass."""
    return sa.value_and_grad(theta, _case(), key, True)


BUILDERS = {"forward": _forward, "grad": _grad}


def measure(label):
    """``(flops, bytes_accessed, temp_bytes)`` of the compiled program."""
    f = jax.jit(BUILDERS[label])
    compiled = f.lower(sa.reference_theta(), jax.random.PRNGKey(0)).compile()
    cost = compiled.cost_analysis()
    return (
        float(cost["flops"]),
        float(cost["bytes accessed"]),
        float(compiled.memory_analysis().temp_size_in_bytes),
    )


@pytest.mark.parametrize("label", sorted(BUILDERS))
@pytest.mark.parametrize("metric", (0, 1, 2))
def test_compiled_cost_is_within_budget(label, metric):
    """Flops, bytes accessed and peak temporary allocation, against a recorded baseline."""
    backend = jax.default_backend()
    if backend not in BUDGETS:
        pytest.skip(f"no baseline recorded for backend {backend!r}")
    name = ("flops", "bytes accessed", "temp bytes")[metric]
    got = measure(label)[metric]
    want = BUDGETS[backend][label][metric]
    assert got == pytest.approx(want, rel=TOL), (
        f"{backend}/{label} {name}: {got:.6e} vs baseline {want:.6e} "
        f"({got / want:.2f}x). If this is an intended change, refresh with "
        f"`python3 -m tests.test_perf_budget --print-baseline` on this backend."
    )


def test_reverse_pass_costs_about_twice_the_forward_in_flops():
    """Section 23.4: the reverse pass is a small multiple of the forward, not an order.

    Backend-independent, so this survives a baseline refresh and a hardware change.
    """
    fwd, grad = measure("forward")[0], measure("grad")[0]
    ratio = grad / fwd
    assert 1.2 < ratio < 4.0, f"reverse/forward flops = {ratio:.2f}x, expected ~2x"


def test_reverse_pass_memory_multiple_is_bounded():
    """Section 23.4.1: memory, not time, is what caps an adjoint run -- so bound it."""
    fwd, grad = measure("forward")[2], measure("grad")[2]
    ratio = grad / fwd
    assert ratio < 15.0, f"reverse/forward temp memory = {ratio:.1f}x, expected ~7x"


def test_cost_analysis_is_deterministic():
    """The whole approach rests on this: recompiling gives bit-identical numbers."""
    assert measure("grad") == measure("grad")


def _print_compile_time():
    """Report, do not assert.  A fresh process per backend is what makes it meaningful."""
    print(f"  backend={jax.default_backend()}")
    for label in sorted(BUILDERS):
        f = jax.jit(BUILDERS[label])
        lowered = f.lower(sa.reference_theta(), jax.random.PRNGKey(0))
        t0 = time.perf_counter()
        lowered.compile()
        print(f"  {label:>8}: first compile {time.perf_counter() - t0:.2f} s")
    print("  (only the first compile of a program is real; JAX caches in process)")


def _print_baseline():
    backend = jax.default_backend()
    print(f'    "{backend}": {{')
    for label in sorted(BUILDERS):
        f, b, t = measure(label)
        print(f'        "{label}": ({f:.6e}, {b:.6e}, {t / 1e6:.3f}e6),')
    print("    },")


if __name__ == "__main__":
    if "--compile-time" in sys.argv:
        _print_compile_time()
    else:
        _print_baseline()
