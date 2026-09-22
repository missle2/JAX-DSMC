"""How much does the host backward's gradient move with the ridge constant?  (provenance check)"""
import sys

import jax

from examples import ensemble_inlet_shock as eis
from jax_dsmc import ensemble_backward as eb

particles, steps = int(sys.argv[1]), int(sys.argv[2])
geometry = eis.inlet_shock_geometry(particles, 8, 0.05)
key = jax.random.key(32)
for ridge in (1e-10, 1e-8, 1e-6):
    eb.RIDGE_REL = ridge
    j, g, _ = eis.ensemble_gradient("temp", key, geometry, 8, 512, steps, "certified", particles, 32, False, True)
    print(f"RIDGE_REL={ridge:g} J={j:.10g} g={g:.10g}", flush=True)
