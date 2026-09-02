"""Weighted-ensemble binning and resampling.

The weighted ensemble method carries a population of walkers, each with a
probability weight.  After every generation the walkers are sorted into bins
along a progress coordinate and the population within each bin is resampled so
that under-populated regions of state space receive computational effort in
proportion to interest rather than in proportion to probability.  Weight is
conserved exactly at every step, which is what makes the resulting rate
estimates unbiased.

Two resampling rules are provided:

``systematic``
    Stratified resampling to exactly ``n_per_bin`` walkers of equal weight
    ``W_bin / n_per_bin``.  Unbiased, O(n), and the default.

``split_merge``
    Classical WESTPA-style splitting of the heaviest walker and merging of the
    two lightest until the target count is reached.  Retains more weight
    diversity within a bin; marginally more expensive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BinScheme:
    """A rectilinear binning of a scalar progress coordinate.

    ``edges`` are the interior boundaries, so ``n_bins = len(edges) + 1``.
    Bin 0 is the region below ``edges[0]`` and bin ``n_bins - 1`` is above
    ``edges[-1]``.  The two states of interest are named by bin index.
    """

    edges: np.ndarray
    state_a: int  # source basin (recycling target)
    state_b: int  # sink basin (flux is measured here)

    @property
    def n_bins(self) -> int:
        return len(self.edges) + 1

    @staticmethod
    def uniform(lo: float, hi: float, n_bins: int, state_a: int = 0, state_b: int = -1) -> "BinScheme":
        edges = np.linspace(lo, hi, n_bins - 1)
        b = state_b if state_b >= 0 else n_bins + state_b
        return BinScheme(edges=edges, state_a=state_a, state_b=b)

    def assign(self, pcoord: torch.Tensor) -> torch.Tensor:
        e = torch.as_tensor(self.edges, dtype=pcoord.dtype, device=pcoord.device)
        return torch.bucketize(pcoord, e)


def systematic_indices(weights: torch.Tensor, target: int, generator=None) -> torch.Tensor:
    """Stratified resampling: parent indices for ``target`` equal-weight children."""
    total = weights.sum()
    if total <= 0:
        return torch.zeros(target, dtype=torch.int64, device=weights.device)
    cdf = torch.cumsum(weights, dim=0) / total
    u = torch.rand(1, device=weights.device, dtype=weights.dtype, generator=generator)
    points = (u + torch.arange(target, device=weights.device, dtype=weights.dtype)) / target
    return torch.searchsorted(cdf, points.clamp(max=1.0)).clamp(max=weights.numel() - 1)


def split_merge_indices(weights: torch.Tensor, target: int, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Classical split/merge.  Returns ``(parent_indices, child_weights)``."""
    w = weights.detach().cpu().numpy().astype(np.float64)
    parents = list(range(len(w)))
    wl = list(w)

    while len(wl) > target:
        order = np.argsort(wl)
        i, j = int(order[0]), int(order[1])
        wi, wj = wl[i], wl[j]
        tot = wi + wj
        keep = i if (tot <= 0 or np.random.rand() < wi / tot) else j
        survivor = parents[keep]
        for k in sorted((i, j), reverse=True):
            del wl[k]
            del parents[k]
        wl.append(tot)
        parents.append(survivor)

    while len(wl) < target:
        i = int(np.argmax(wl))
        wl[i] /= 2.0
        wl.append(wl[i])
        parents.append(parents[i])

    dev = weights.device
    return (
        torch.as_tensor(parents, dtype=torch.int64, device=dev),
        torch.as_tensor(wl, dtype=weights.dtype, device=dev),
    )


def resample(
    x: torch.Tensor,
    w: torch.Tensor,
    bin_id: torch.Tensor,
    n_bins: int,
    n_per_bin: int,
    method: str = "systematic",
    generator=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample a walker population bin by bin.

    Returns ``(new_x, new_w, new_bin_id)``.  Total weight is conserved to
    floating-point precision; empty bins are simply skipped, so the walker
    count is ``n_per_bin * n_occupied_bins``.
    """
    parents_all: list[torch.Tensor] = []
    weights_all: list[torch.Tensor] = []
    bins_all: list[torch.Tensor] = []

    for b in range(n_bins):
        sel = (bin_id == b).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        wb = w.index_select(0, sel)
        total = wb.sum()
        if method == "systematic":
            idx = systematic_indices(wb, n_per_bin, generator=generator)
            child_w = torch.full((n_per_bin,), 1.0, dtype=w.dtype, device=w.device) * (total / n_per_bin)
        elif method == "split_merge":
            idx, child_w = split_merge_indices(wb, n_per_bin, generator=generator)
            child_w = child_w * (total / child_w.sum())
        else:
            raise ValueError(f"unknown resampling method {method!r}")
        parents_all.append(sel.index_select(0, idx))
        weights_all.append(child_w)
        bins_all.append(torch.full((child_w.numel(),), b, dtype=torch.int64, device=w.device))

    if not parents_all:
        return x, w, bin_id

    parents = torch.cat(parents_all)
    return (
        x.index_select(0, parents).clone(),
        torch.cat(weights_all),
        torch.cat(bins_all),
    )
