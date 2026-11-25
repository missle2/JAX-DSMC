"""SPARTA's three benchmark decks (``bench/in.free``, ``in.collide``, ``in.sphere``) as data.

Not an input parser: each function returns the numbers its ``.in`` file sets, with the size
variables ``x y z`` (``-v x 100 ...``) as arguments.  Copies of the decks are in
``sparta_jax/deck/bench`` for reference; the formulas below are copied from them.

The benchmark is the LAST run of each deck; the earlier runs equilibrate (``Run.timed``).
"""

from __future__ import annotations

import dataclasses
import os

from . import geometry3d as G3


@dataclasses.dataclass(frozen=True)
class Run:
    """One ``run`` command: timestep, steps, and the collide_modify vremax state in force."""

    dt: float
    nsteps: int
    timed: bool = False
    vre_start: bool = True  # reset vremax/remain at the start of this run (collide_modify)


@dataclasses.dataclass(frozen=True)
class Deck:
    name: str
    x: int
    y: int
    z: int
    lo: tuple
    hi: tuple
    boundary: str  # "r" (reflect, all faces) or "o" (outflow, all faces)
    nrho: float
    fnum: float
    temp: float
    vstream: tuple
    n_create: int  # create_particles n (0 = from nrho and the flow volume)
    collide: bool
    emit: bool  # fix emit/face on all faces
    surf_file: str | None
    tsurf: float
    acc: float  # surf_collide diffuse accommodation
    runs: tuple
    vre_every: int  # collide_modify vremax N (0 = never)
    stats_every: int
    balance_every: int  # fix balance N (0 = none)
    species: G3.Species = G3.Species()

    @property
    def grid(self) -> G3.Grid:
        return G3.Grid(lo=self.lo, hi=self.hi, n=(self.x, self.y, self.z))

    @property
    def n_particles(self):
        """Nominal particle count (the benchmark's "size"): 10 per grid cell."""
        return 10 * self.x * self.y * self.z


def _species():
    d = G3.BENCH_DIR
    return G3.read_species(os.path.join(d, "ar.species"), os.path.join(d, "ar.vss"), "Ar")


def free(x=10, y=10, z=10, steps=100) -> Deck:
    """in.free: box with reflecting walls, no collisions."""
    return Deck(
        name="free",
        x=x,
        y=y,
        z=z,
        lo=(0.0, 0.0, 0.0),
        hi=(x * 1.0e-5, y * 1.0e-5, z * 1.0e-5),
        boundary="r",
        nrho=7.07043e22,
        fnum=7.07043e6,
        temp=273.15,
        vstream=(0.0, 0.0, 0.0),
        n_create=10 * x * y * z,
        collide=False,
        emit=False,
        surf_file=None,
        tsurf=0.0,
        acc=0.0,
        runs=(Run(7.0e-8, 30), Run(7.0e-9, steps, timed=True)),
        vre_every=0,
        stats_every=10,
        balance_every=0,
        species=_species(),
    )


def collide(x=10, y=10, z=10, steps=100) -> Deck:
    """in.collide: in.free plus VSS collisions (and hence the per-step sort)."""
    d = free(x, y, z, steps)
    return dataclasses.replace(d, name="collide", collide=True)


def sphere(x=10, y=10, z=10, steps=1000) -> Deck:
    """in.sphere: flow at 2500 m/s past a sphere of radius 1 in a 10^3 box, outflow faces with
    emit/face on all of them, diffuse surface (acc 1), vremax reset every 100 steps."""
    n = 10 * x * y * z
    return Deck(
        name="sphere",
        x=x,
        y=y,
        z=z,
        lo=(-5.0, -5.0, -5.0),
        hi=(5.0, 5.0, 5.0),
        boundary="o",
        nrho=7.03e18,
        fnum=7.33e21 / n,
        temp=300.0,
        vstream=(2500.0, 0.0, 0.0),
        n_create=0,
        collide=True,
        emit=True,
        surf_file=os.path.join(G3.BENCH_DIR, "data.sphere"),
        tsurf=300.0,
        acc=1.0,
        # collide_modify vremax 100 yes ; run 1000 ; collide_modify vremax 100 no ; run 1000
        runs=(Run(1.0e-5, 1000, vre_start=True), Run(1.0e-5, steps, timed=True, vre_start=False)),
        vre_every=100,
        stats_every=100,
        balance_every=100,  # fix 3 balance 100 1.1 rcb part
        species=_species(),
    )


DECKS = {"free": free, "collide": collide, "sphere": sphere}


def make(name, x, y=None, z=None, steps=None) -> Deck:
    y = x if y is None else y
    z = x if z is None else z
    kw = {} if steps is None else {"steps": steps}
    return DECKS[name](x, y, z, **kw)


def size_xyz(n_particles: int):
    """Grid (x, y, z) for a nominal particle count, as the bench README does for its log sizes
    (10K = 10^3, 100K = 20x20x25, 1M = 40x50x50, 10M = 100^3): the most cubic factorisation of
    n/10 cells with x <= y <= z."""
    cells = n_particles // 10
    best = None
    for x in range(1, int(round(cells ** (1 / 3))) + 2):
        if cells % x:
            continue
        rest = cells // x
        for y in range(x, int(rest**0.5) + 2):
            if rest % y:
                continue
            z = rest // y
            if z < y:
                continue
            score = max(x, y, z) / min(x, y, z)
            if best is None or score < best[0]:
                best = (score, (x, y, z))
    if best is None:
        raise ValueError(f"{n_particles}: no grid with 10 particles per cell")
    return best[1]
