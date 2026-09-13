"""Reaction network specification and vectorised propensity evaluation.

A network is a set of species and reaction channels.  Each channel ``j`` has a
propensity of the separable form

    a_j(x) = c_j * MA_j(x) * prod_{m in mods(j)} f_m(x)

where ``MA_j`` is a mass-action factor over reactant species (orders 0, 1 or 2)
and each ``f_m`` is a Hill activation or repression factor in ``(0, 1]``:

    activation:  f = b + (1 - b) * x^h / (K^h + x^h)
    repression:  f = b + (1 - b) * K^h / (K^h + x^h)

``b`` is a leak (basal) term that keeps promoters from shutting off completely,
which matters both biologically and numerically.

This separable form is deliberately restrictive.  It covers essentially every
gene-regulatory model in the transcription-factor literature, and it compiles to
flat arrays that a single CUDA thread can evaluate from registers without any
branching, which is the property the whole framework depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACTIVATION = 1
REPRESSION = -1


@dataclass
class Modifier:
    """A Hill regulatory factor applied to one reaction channel."""

    species: str
    kind: int  # ACTIVATION or REPRESSION
    K: float
    h: float = 2.0
    basal: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in (ACTIVATION, REPRESSION):
            raise ValueError("kind must be ACTIVATION or REPRESSION")
        if self.K <= 0:
            raise ValueError("Hill constant K must be positive")
        if not 0.0 <= self.basal < 1.0:
            raise ValueError("basal must lie in [0, 1)")


@dataclass
class Reaction:
    """One reaction channel.

    Parameters
    ----------
    name
        Human-readable label, used in diagnostics only.
    rate
        Deterministic rate constant ``c_j``.
    reactants
        Mapping from species name to stoichiometric order consumed.  Orders
        above 2 are rejected; trimolecular reactions are not physical at the
        level of description used here.
    products
        Mapping from species name to stoichiometric count produced.
    modifiers
        Hill regulatory factors multiplying the propensity.
    """

    name: str
    rate: float
    reactants: dict[str, int] = field(default_factory=dict)
    products: dict[str, int] = field(default_factory=dict)
    modifiers: list[Modifier] = field(default_factory=list)


@dataclass
class CompiledNetwork:
    """Flat-array form of a network, ready for any backend.

    All arrays are NumPy; backends copy them to the device once at start-up.
    """

    species: list[str]
    reaction_names: list[str]
    rate: np.ndarray  # (R,)   float64
    stoich: np.ndarray  # (R, S) int32, net change in species per firing
    factor_rxn: np.ndarray  # (F,)   int32, reaction each factor multiplies
    factor_species: np.ndarray  # (F,)   int32
    factor_kind: np.ndarray  # (F,)   int32; 0 mass-action, 1 Hill act, -1 Hill rep
    factor_order: np.ndarray  # (F,)   int32, mass-action order (0 for Hill)
    factor_K: np.ndarray  # (F,)   float64
    factor_h: np.ndarray  # (F,)   float64
    factor_basal: np.ndarray  # (F,)   float64

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        return len(self.reaction_names)

    @property
    def n_factors(self) -> int:
        return int(self.factor_rxn.size)

    def index(self, name: str) -> int:
        return self.species.index(name)


class ReactionNetwork:
    """A named collection of species and reaction channels."""

    def __init__(self, name: str, species: list[str], reactions: list[Reaction]):
        self.name = name
        self.species = list(species)
        self.reactions = list(reactions)
        seen = set()
        for s in self.species:
            if s in seen:
                raise ValueError(f"duplicate species {s!r}")
            seen.add(s)
        for r in self.reactions:
            for group in (r.reactants, r.products):
                for s in group:
                    if s not in seen:
                        raise ValueError(f"reaction {r.name!r} refers to unknown species {s!r}")
            for m in r.modifiers:
                if m.species not in seen:
                    raise ValueError(f"modifier of {r.name!r} refers to unknown species {m.species!r}")

    def compile(self) -> CompiledNetwork:
        s_index = {s: i for i, s in enumerate(self.species)}
        n_s, n_r = len(self.species), len(self.reactions)

        rate = np.zeros(n_r, dtype=np.float64)
        stoich = np.zeros((n_r, n_s), dtype=np.int32)

        f_rxn: list[int] = []
        f_sp: list[int] = []
        f_kind: list[int] = []
        f_order: list[int] = []
        f_K: list[float] = []
        f_h: list[float] = []
        f_basal: list[float] = []

        for j, rxn in enumerate(self.reactions):
            rate[j] = rxn.rate
            for s, n in rxn.reactants.items():
                if n not in (1, 2):
                    raise ValueError(f"reactant order {n} unsupported in {rxn.name!r}; use 1 or 2")
                stoich[j, s_index[s]] -= n
                f_rxn.append(j)
                f_sp.append(s_index[s])
                f_kind.append(0)
                f_order.append(n)
                f_K.append(1.0)
                f_h.append(1.0)
                f_basal.append(0.0)
            for s, n in rxn.products.items():
                stoich[j, s_index[s]] += n
            for m in rxn.modifiers:
                f_rxn.append(j)
                f_sp.append(s_index[m.species])
                f_kind.append(int(m.kind))
                f_order.append(0)
                f_K.append(float(m.K))
                f_h.append(float(m.h))
                f_basal.append(float(m.basal))

        return CompiledNetwork(
            species=list(self.species),
            reaction_names=[r.name for r in self.reactions],
            rate=rate,
            stoich=stoich,
            factor_rxn=np.asarray(f_rxn, dtype=np.int32),
            factor_species=np.asarray(f_sp, dtype=np.int32),
            factor_kind=np.asarray(f_kind, dtype=np.int32),
            factor_order=np.asarray(f_order, dtype=np.int32),
            factor_K=np.asarray(f_K, dtype=np.float64),
            factor_h=np.asarray(f_h, dtype=np.float64),
            factor_basal=np.asarray(f_basal, dtype=np.float64),
        )


def propensities_numpy(net: CompiledNetwork, x: np.ndarray) -> np.ndarray:
    """Reference propensity evaluation, ``x`` of shape (W, S) -> (W, R).

    This is the definition against which every backend is validated.
    """
    x = np.atleast_2d(x).astype(np.float64)
    w = x.shape[0]
    a = np.repeat(net.rate[None, :], w, axis=0)
    if net.n_factors == 0:
        return a

    xs = x[:, net.factor_species]  # (W, F)
    kind = net.factor_kind
    order = net.factor_order

    fac = np.ones_like(xs)

    ma1 = kind == 0
    if ma1.any():
        o1 = ma1 & (order == 1)
        o2 = ma1 & (order == 2)
        fac[:, o1] = xs[:, o1]
        fac[:, o2] = 0.5 * xs[:, o2] * (xs[:, o2] - 1.0)

    hill = kind != 0
    if hill.any():
        xh = np.power(np.maximum(xs[:, hill], 0.0), net.factor_h[hill])
        kh = np.power(net.factor_K[hill], net.factor_h[hill])
        frac = xh / (kh + xh)
        is_rep = net.factor_kind[hill] == REPRESSION
        frac = np.where(is_rep, 1.0 - frac, frac)
        b = net.factor_basal[hill]
        fac[:, hill] = b + (1.0 - b) * frac

    np.multiply.at(a.T, net.factor_rxn, fac.T)  # scatter-product into channels
    return np.maximum(a, 0.0)
