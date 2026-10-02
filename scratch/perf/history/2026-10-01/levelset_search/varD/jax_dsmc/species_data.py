"""Reference VSS parameters for real gases, from Bird's tables.

Source: ``papers_reference/bird_molecular_gas_dynamics/appendices.pdf``.  Table A1
(page 408) gives mass and the viscosity index ``omega``; Table A3 (page 410) gives the
VSS scattering exponent ``alpha`` and the VSS reference diameter at 273 K.

**On trusting the transcription.**  The PDF text extractor scrambles Table A3's *name*
column (it prints ``He`` and ``CH4`` swapped) while preserving the numeric columns, so
rows were reassigned by matching the VSS diameters against Table A2's VHS diameters,
which are systematically ~1 % larger.  Two independent anchors confirm the result:
``Ar`` comes out at ``alpha = 1.40, d = 4.11e-10``, which is exactly SPARTA's
``data/ar.vss`` (``Ar 4.11e-10 0.81 273.15 1.4``), and every other value matches the
widely cited Bird VSS set.

Bird notes (page 410) that ``alpha`` is temperature dependent -- ``Kr`` at 473 K is 1.29
and ``Xe`` at 378 K is 1.40 -- and that "it is not clear whether ``a`` can reasonably be
regarded as a constant in applications that involve large changes in temperature".
These values are for temperatures of order 273 K.
"""

from __future__ import annotations

import dataclasses
import math

ALPHA_STATIONARY = math.sqrt(2.0)
"""``sqrt(2)``: the exact stationary point of ``<cos^2 chi>`` for the VSS deflection law.

``<cos^2 chi> = 4a/(a+2) - 4a/(a+1) + 1`` has derivative ``8/(a+2)^2 - 4/(a+1)^2``, which
vanishes at ``a = sqrt(2)``.  Anisotropy relaxation is driven by that moment, so **a
temperature-like objective has near-zero sensitivity to ``alpha`` near ``sqrt(2)``**, and
an ``alpha`` gradient measured there is ill-conditioned no matter how good the estimator
is.  Measured and confirmed in ``examples/p2_replication`` (sign change bracketing
``sqrt(2)``, Pearson ``r = 0.992`` against this curve).
"""


@dataclasses.dataclass(frozen=True)
class Species:
    """One gas.  ``mass`` kg, ``diam`` m at ``tref``, ``omega`` and ``alpha`` unitless."""

    symbol: str
    name: str
    mass: float
    diam: float
    omega: float
    alpha: float
    monatomic: bool
    tref: float = 273.0

    @property
    def d_legendre_d_alpha(self) -> float:
        """``d<cos^2 chi>/d(alpha)`` at this gas's ``alpha``.

        The **conditioning metric for an alpha sensitivity study**: its magnitude is
        proportional to ``dJ/d(alpha)`` for a temperature-like objective, so a gas whose
        ``alpha`` sits near ``sqrt(2)`` has structurally no signal to measure.
        """
        a = self.alpha
        return 8.0 / (a + 2.0) ** 2 - 4.0 / (a + 1.0) ** 2

    @property
    def alpha_conditioning(self) -> float:
        """``|d<cos^2 chi>/d(alpha)|`` relative to argon's.  Bigger is better."""
        return abs(self.d_legendre_d_alpha) / abs(ARGON.d_legendre_d_alpha)


# Table A1 masses are printed as `m x 10^27 kg`; Table A3 diameters as `d x 10^10 m`.
HELIUM = Species("He", "helium", 6.65e-27, 2.30e-10, 0.66, 1.26, True)
NEON = Species("Ne", "neon", 33.5e-27, 2.72e-10, 0.66, 1.31, True)
ARGON = Species("Ar", "argon", 66.3e-27, 4.11e-10, 0.81, 1.40, True)
KRYPTON = Species("Kr", "krypton", 139.1e-27, 4.70e-10, 0.80, 1.32, True)
XENON = Species("Xe", "xenon", 218.0e-27, 5.65e-10, 0.85, 1.44, True)

NITROGEN = Species("N2", "nitrogen", 46.5e-27, 4.11e-10, 0.74, 1.36, False)
OXYGEN = Species("O2", "oxygen", 53.12e-27, 4.01e-10, 0.77, 1.40, False)

MONATOMIC = (HELIUM, NEON, ARGON, KRYPTON, XENON)
"""The five noble gases Bird tabulates -- the ones with 3 degrees of freedom (Table A1),
so no internal energy and no Larsen-Borgnakke, which is what our solver models."""

ALL = (*MONATOMIC, NITROGEN, OXYGEN)

BY_SYMBOL = {s.symbol: s for s in ALL}


def best_for_alpha_study(candidates=MONATOMIC) -> Species:
    """The monatomic gas whose ``alpha`` sits furthest from the stationary point.

    Helium, by a wide margin: ``alpha = 1.26`` gives ~12.6x argon's signal.  Argon is the
    *worst* available choice, sitting 1.0 % from ``sqrt(2)``.
    """
    return max(candidates, key=lambda s: abs(s.d_legendre_d_alpha))
