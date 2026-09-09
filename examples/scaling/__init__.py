"""Weak- and strong-scaling measurement over 1, 2 and 4 devices.

Run it on the node you care about::

    python3 -m examples.scaling --devices 1 2 4 --mode both

**What is being measured.**  The figure of merit for a particle code is
*particle-steps per second*, not wall time, because the two scaling questions ask
different things of it:

**strong scaling**
    fixed total problem, more devices.  Ideal is wall time ``~ 1/P``.  This is the
    hard one, and DSMC's global sort is what limits it (see
    :mod:`jax_dsmc.parallel`).

**weak scaling**
    fixed work *per device*, more devices.  Ideal is constant wall time.  This is
    the regime a 10^10-particle run actually lives in -- nobody buys more GPUs to
    run the same small problem faster, they buy them to run a bigger one.

Both are reported, along with a per-stage breakdown (move / sort / collide /
sample), because an aggregate speed-up number does not say what to fix and the
per-stage split does.

**Benchmarking discipline** (spec section 8.6, learned the hard way -- a phantom
36x cliff turned out to be a concurrent job on the same GPU):

* every timing is preceded by a warm-up call and a ``block_until_ready``; without
  the block, JAX's async dispatch times the *enqueue*, not the work;
* compilation is excluded by construction -- the warm-up pays it;
* the machine must be otherwise idle.  :func:`check_idle` warns if it is not.
"""
