"""Physical and numerical constants.

Spec: PSEUDOCODE.md section 2.2 (clamp constants) and section 7.1 (K_BOLTZ).

All values are plain Python floats so they can be used both as static Python
scalars and as weakly-typed operands inside traced expressions.
"""

# --- physical -------------------------------------------------------------

K_BOLTZ = 1.380649e-23
"""Boltzmann constant, J/K (SI, exact by the 2019 SI redefinition).

Matches SPARTA's ``update.cpp`` ``boltz`` for the ``si`` unit system.
"""

# --- numerical clamps (section 2.2) ---------------------------------------

EPS = 1e-30
"""Denominator clamp.

Harmless wherever the composite expression is bounded: section 2.3 measures
``safe_norm``'s derivative as exactly ``1.0`` across 14 decades of ``|v|``
*independently of* ``eps``, because the chain rule cancels ``1/(2 sqrt(x))``
against ``2 v``. Do NOT raise this to "fix" a gradient -- if a derivative is
unbounded, the clamp that bounds it is a *relative*, physically-scaled one
(``EPS_REL_VR2``), not a bigger absolute floor.
"""

EPS_D = 1e-6
"""RELATIVE frame-degeneracy threshold for the VSS scattering frame.

Always used as ``d > EPS_D * vrm``, **never** as an absolute cut. SPARTA's
``collide_vss.cpp:429`` writes ``if (d > 1.0e-6)``, which is *dimensional* and
only works because SPARTA runs in SI with thermal speeds ~1e2 m/s. Section 7.4
measures an order-unity difference in post-collision velocity from a change of
units alone under the absolute form. The relative form is exactly
scale-invariant and caps the scatter Jacobian at ``1 / EPS_D`` in every unit
system.
"""

EPS_REL_VR2 = 1e-8
"""RELATIVE clamp for ``safe_pow`` applied to ``vr2``.

The absolute cut handed to ``safe_pow`` is

    eps = EPS_REL_VR2 * (2 * K_BOLTZ * tref / mr)

i.e. a fixed fraction of the pair's reference thermal energy scale, so the
clamp sits at ``|vr| = sqrt(EPS_REL_VR2) * v_thermal`` in any unit system.
An absolute ``1e-30`` here leaves ``d q / d|vr|`` unbounded at ~2.5e16
(section 2.3). See ``tests/test_safe.py::test_u1_*`` for the validated bound.
"""

EPS_U = 1e-12
"""Clamp for ``safe_pow`` applied to a dimensionless uniform on ``[0, 1)``.

Used by the VSS angular law ``cosX = 2 * u_chi**(1/alpha) - 1``
(``collide_vss.cpp:417``). ``u_chi`` is already dimensionless, so the
relative-to-thermal-scale form of ``EPS_REL_VR2`` does not apply and a plain
small absolute clamp is correct (section 7.4).
"""
