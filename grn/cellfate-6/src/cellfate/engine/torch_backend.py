"""Vectorised tau-leaping on any Torch device (CPU, CUDA, GH200).

One row of the state tensor is one cell.  Every operation is a fused elementwise
or matmul kernel over the walker dimension, so a single GH200 sustains on the
order of 10^5-10^6 walkers concurrently for a network of this size.

The stepper is *synchronised*: every walker takes the same leap ``tau`` so that
lanes in a warp stay in lockstep.  Walkers whose step would drive a species
negative are recursively subdivided, which keeps the divergence confined to a
small minority of walkers rather than making every step data-dependent.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

from ..model import CompiledNetwork

# index_reduce_ is marked beta and warns on every call, once per rank per
# process.  With hundreds of ranks that buries the actual log output.  The API
# is used in exactly one place below; if a future torch release changes it, the
# propensity tests fail loudly, which is the real safeguard.
warnings.filterwarnings(
    "ignore", message=".*index_reduce.*is in beta.*", category=UserWarning
)


@dataclass
class LeapDiagnostics:
    """Counters accumulated across a call to :meth:`TorchEngine.advance`."""

    subdivisions: int = 0
    clamped: int = 0
    steps: int = 0

    def merge(self, other: "LeapDiagnostics") -> None:
        self.subdivisions += other.subdivisions
        self.clamped += other.clamped
        self.steps += other.steps


class TorchEngine:
    """Tau-leaping integrator for a compiled network.

    Parameters
    ----------
    net
        Compiled network.
    device
        Torch device string, e.g. ``"cuda:0"``.
    dtype
        Floating dtype for propensities.  ``float32`` is roughly twice the
        throughput; ``float64`` is affordable on GH200 and is the default
        because propensity sums over many channels are the accuracy-critical
        step.  Validate any switch to ``float32`` with ``cellfate selftest``.
    max_subdivisions
        Depth limit for recursive tau halving before a step is clamped at zero.
        Clamping is a bias, so the count is reported and should stay negligible.
    """

    def __init__(
        self,
        net: CompiledNetwork,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        max_subdivisions: int = 6,
    ):
        self.net = net
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_subdivisions = max_subdivisions

        d, f = self.device, dtype
        self.rate = torch.as_tensor(net.rate, dtype=f, device=d)
        self.stoich = torch.as_tensor(net.stoich, dtype=f, device=d)  # (R, S)
        self.stoich_i = torch.as_tensor(net.stoich, dtype=torch.int64, device=d)
        self.f_rxn = torch.as_tensor(net.factor_rxn, dtype=torch.int64, device=d)
        self.f_sp = torch.as_tensor(net.factor_species, dtype=torch.int64, device=d)
        self.f_K = torch.as_tensor(net.factor_K, dtype=f, device=d)
        self.f_h = torch.as_tensor(net.factor_h, dtype=f, device=d)
        self.f_b = torch.as_tensor(net.factor_basal, dtype=f, device=d)
        kind = torch.as_tensor(net.factor_kind, dtype=torch.int64, device=d)
        order = torch.as_tensor(net.factor_order, dtype=torch.int64, device=d)
        self.is_hill = kind != 0
        self.is_rep = kind == -1
        self.is_ma1 = (kind == 0) & (order == 1)
        self.is_ma2 = (kind == 0) & (order == 2)
        self.f_Kh = torch.pow(self.f_K, self.f_h)

    # ------------------------------------------------------------------ #

    def propensities(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` of shape (W, S) integer or float -> (W, R) propensities."""
        xf = x.to(self.dtype)
        w = xf.shape[0]
        a = self.rate.expand(w, -1).clone()
        if self.net.n_factors == 0:
            return a

        xs = xf[:, self.f_sp]  # (W, F)
        fac = torch.ones_like(xs)

        if bool(self.is_ma1.any()):
            fac = torch.where(self.is_ma1, xs, fac)
        if bool(self.is_ma2.any()):
            fac = torch.where(self.is_ma2, 0.5 * xs * (xs - 1.0), fac)
        if bool(self.is_hill.any()):
            xh = torch.pow(xs.clamp(min=0.0), self.f_h)
            frac = xh / (self.f_Kh + xh)
            frac = torch.where(self.is_rep, 1.0 - frac, frac)
            hill = self.f_b + (1.0 - self.f_b) * frac
            fac = torch.where(self.is_hill, hill, fac)

        a = a.index_reduce_(1, self.f_rxn, fac, "prod", include_self=True)
        return a.clamp(min=0.0)

    # ------------------------------------------------------------------ #

    def _leap(
        self,
        x: torch.Tensor,
        tau: float,
        depth: int,
        diag: LeapDiagnostics,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        a = self.propensities(x)
        k = torch.poisson(a * tau, generator=generator)
        nx = x + (k @ self.stoich).round().to(x.dtype)
        diag.steps += 1

        bad = (nx < 0).any(dim=1)
        if not bool(bad.any()):
            return nx

        if depth >= self.max_subdivisions:
            diag.clamped += int(bad.sum())
            return nx.clamp_(min=0)

        idx = bad.nonzero(as_tuple=True)[0]
        diag.subdivisions += int(idx.numel())
        sub = x.index_select(0, idx)
        half = tau * 0.5
        sub = self._leap(sub, half, depth + 1, diag, generator)
        sub = self._leap(sub, half, depth + 1, diag, generator)
        nx.index_copy_(0, idx, sub)
        return nx

    def advance(
        self,
        x: torch.Tensor,
        tau: float,
        n_steps: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, LeapDiagnostics]:
        """Advance every walker by ``n_steps`` leaps of size ``tau``."""
        diag = LeapDiagnostics()
        for _ in range(n_steps):
            x = self._leap(x, tau, 0, diag, generator)
        return x, diag

    # ------------------------------------------------------------------ #

    def suggest_tau(self, x: torch.Tensor, epsilon: float = 0.03) -> float:
        """A conservative leap size from the current walker population.

        Uses a simplified Cao-Gillespie criterion: the leap must not change any
        species by more than a fraction ``epsilon`` of its current count (with a
        floor of one molecule) for the bulk of walkers.  Returned as a scalar so
        that all walkers stay synchronised.
        """
        a = self.propensities(x)  # (W, R)
        xf = x.to(self.dtype)
        mu = a @ self.stoich  # (W, S) expected drift per unit time
        sig2 = a @ (self.stoich**2)  # (W, S) variance rate
        bound = torch.clamp(epsilon * xf, min=1.0)
        with torch.no_grad():
            t1 = bound / mu.abs().clamp(min=1e-30)
            t2 = bound**2 / sig2.clamp(min=1e-30)
            per_walker = torch.minimum(t1, t2).amin(dim=1)
            tau = torch.quantile(per_walker.flatten().to(torch.float64), 0.05)
        return float(tau)


def state_from_counts(counts: dict[str, int], net: CompiledNetwork, n: int, device) -> torch.Tensor:
    """Build an (n, S) int64 state tensor from a species-count dictionary."""
    x0 = np.zeros(net.n_species, dtype=np.int64)
    for s, v in counts.items():
        x0[net.index(s)] = v
    return torch.as_tensor(np.tile(x0, (n, 1)), device=device)
