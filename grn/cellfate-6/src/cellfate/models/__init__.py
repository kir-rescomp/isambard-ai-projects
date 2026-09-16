"""Reference models.

``birth_death``
    One species with constant birth and linear death.  The stationary
    distribution is Poisson with mean ``k/d``, so it gives an analytic check on
    the leaping backends.

``toggle``
    Two mutually repressing proteins with self-activation, produced directly.
    A minimal bistable switch, cheap enough to run to convergence on CPU.  Used
    to validate the weighted-ensemble rate estimate against brute force.

``th17_treg``
    The production model.  RORgt and FOXP3 with explicit mRNA, mutual
    repression, self-activation, and cytokine inputs entering through STAT3
    (IL-6) and STAT5 (IL-2 with TGF-beta).  Bursty transcription at low mRNA
    copy number is the noise source that drives the fate transition, which is
    why the mRNA layer is modelled explicitly rather than adiabatically
    eliminated.
"""

from __future__ import annotations

import numpy as np
import torch

from ..model import ACTIVATION, REPRESSION, Modifier, Reaction, ReactionNetwork

__all__ = ["birth_death", "toggle", "th17_treg", "ratio_pcoord", "difference_pcoord"]


def birth_death(k: float = 20.0, d: float = 1.0) -> ReactionNetwork:
    return ReactionNetwork(
        name="birth_death",
        species=["X"],
        reactions=[
            Reaction("birth", rate=k, products={"X": 1}),
            Reaction("death", rate=d, reactants={"X": 1}),
        ],
    )


def toggle(
    k: float = 40.0,
    d: float = 1.0,
    frac_K: float = 0.35,
    h: float = 4.0,
    h_self: float = 2.0,
    leak: float = 0.005,
    self_basal: float = 0.5,
    scale: float = 1.0,
) -> ReactionNetwork:
    """Symmetric two-protein toggle.  ``scale`` raises copy numbers and so the
    barrier height; ``scale=1`` is crossable by brute force, ``scale>=3`` is not.
    """
    kp = k * scale
    kk = frac_K * kp / d
    rxns = []
    for a, b in (("A", "B"), ("B", "A")):
        rxns.append(
            Reaction(
                f"make_{a}",
                rate=kp,
                products={a: 1},
                modifiers=[
                    Modifier(b, REPRESSION, K=kk, h=h, basal=leak),
                    Modifier(a, ACTIVATION, K=kk, h=h_self, basal=self_basal),
                ],
            )
        )
        rxns.append(Reaction(f"decay_{a}", rate=d, reactants={a: 1}))
    return ReactionNetwork("toggle", ["A", "B"], rxns)


def th17_treg(
    il6: float = 1.0,
    tgfb_il2: float = 1.0,
    k_tx: float = 4.0,
    d_m: float = 1.0,
    k_tl: float = 10.0,
    d_p: float = 0.1,
    frac_rep: float = 0.35,
    frac_self: float = 0.35,
    h_rep: float = 4.0,
    h_self: float = 2.0,
    leak: float = 0.005,
    self_basal: float = 0.5,
    K_cyt: float = 0.5,
    scale: float = 1.0,
) -> ReactionNetwork:
    """RORgt / FOXP3 mutual-repression switch with explicit mRNA.

    Parameters
    ----------
    il6, tgfb_il2
        Cytokine inputs in arbitrary units, entering as Hill activation of
        RORgt and FOXP3 transcription respectively.  Sweeping these two is the
        main scientific axis: the transition rate as a function of the cytokine
        milieu.  Equal values give a symmetric switch.
    frac_rep, frac_self
        Hill constants expressed as a fraction of the maximal protein level
        ``P_max = k_tx * k_tl / (d_m * d_p) * scale``.  Expressing them
        relatively keeps the switch topology invariant when ``scale`` changes,
        so that ``scale`` moves only the barrier height.
    scale
        System-size parameter.  Multiplies transcription rate and both Hill
        constants together, raising molecule numbers without moving the
        deterministic steady states, and therefore raising the barrier roughly
        exponentially.  Use ``scale=1`` for validation against brute force and
        ``scale>=3`` for the production rare-event regime.
    """
    p_max = k_tx * k_tl / (d_m * d_p) * scale
    kr = frac_rep * p_max
    ks = frac_self * p_max
    ktx = k_tx * scale

    def cyt(v: float) -> float:
        return v**2 / (K_cyt**2 + v**2)

    s3 = cyt(il6)
    s5 = cyt(tgfb_il2)

    rxns = [
        Reaction(
            "tx_RORgt",
            rate=ktx * max(s3, 1e-9),
            products={"mRORgt": 1},
            modifiers=[
                Modifier("FOXP3", REPRESSION, K=kr, h=h_rep, basal=leak),
                Modifier("RORgt", ACTIVATION, K=ks, h=h_self, basal=self_basal),
            ],
        ),
        Reaction("deg_mRORgt", rate=d_m, reactants={"mRORgt": 1}),
        Reaction("tl_RORgt", rate=k_tl, reactants={"mRORgt": 1}, products={"mRORgt": 1, "RORgt": 1}),
        Reaction("deg_RORgt", rate=d_p, reactants={"RORgt": 1}),
        Reaction(
            "tx_FOXP3",
            rate=ktx * max(s5, 1e-9),
            products={"mFOXP3": 1},
            modifiers=[
                Modifier("RORgt", REPRESSION, K=kr, h=h_rep, basal=leak),
                Modifier("FOXP3", ACTIVATION, K=ks, h=h_self, basal=self_basal),
            ],
        ),
        Reaction("deg_mFOXP3", rate=d_m, reactants={"mFOXP3": 1}),
        Reaction("tl_FOXP3", rate=k_tl, reactants={"mFOXP3": 1}, products={"mFOXP3": 1, "FOXP3": 1}),
        Reaction("deg_FOXP3", rate=d_p, reactants={"FOXP3": 1}),
    ]
    net = ReactionNetwork("th17_treg", ["mRORgt", "RORgt", "mFOXP3", "FOXP3"], rxns)
    net.p_max = p_max
    net.meta = {"il6": il6, "tgfb_il2": tgfb_il2, "scale": scale, "p_max": p_max}
    return net


# ---------------------------------------------------------------------- #
# progress coordinates


def ratio_pcoord(net, pos: str, neg: str, pseudocount: float = 1.0):
    """phi = (pos - neg) / (pos + neg + c), bounded in (-1, 1).

    Bounded coordinates keep the outermost bins from becoming unreachable when
    copy numbers drift, which is a common failure mode with unbounded ones.
    """
    i, j = net.index(pos), net.index(neg)

    def f(x: torch.Tensor) -> torch.Tensor:
        a = x[:, i].to(torch.float64)
        b = x[:, j].to(torch.float64)
        return (a - b) / (a + b + pseudocount)

    return f


def difference_pcoord(net, pos: str, neg: str):
    i, j = net.index(pos), net.index(neg)

    def f(x: torch.Tensor) -> torch.Tensor:
        return (x[:, i] - x[:, j]).to(torch.float64)

    return f


def equilibrate_basin(engine, x0: torch.Tensor, tau: float, n_steps: int, seed: int = 0):
    """Relax a population into a basin to build a recycling reservoir."""
    gen = torch.Generator(device=engine.device)
    gen.manual_seed(seed)
    x, _ = engine.advance(x0, tau, n_steps, generator=gen)
    return x


def counts_to_state(net, counts: dict, n: int, device) -> torch.Tensor:
    x0 = np.zeros(net.n_species, dtype=np.int64)
    for s, v in counts.items():
        x0[net.index(s)] = v
    return torch.as_tensor(np.tile(x0, (n, 1)), device=device)
