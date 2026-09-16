"""Estimators built on top of the weighted-ensemble walker population.

Four quantities are accumulated, and they are the scientific output of the
framework:

flux and mean first passage time
    Under recycling boundary conditions the steady-state probability flux into
    the target basin is the reciprocal of the mean first passage time (the Hill
    relation).  Reported with a blocked standard error over generations, since
    successive generations are correlated.

bin-to-bin transition matrix
    Weighted counts of walker movement between bins over one generation lag,
    row-normalised to a Markov state model.

committor
    The probability of reaching B before returning to A, obtained from the
    transition matrix by solving ``(I - T) q = 0`` on the intermediate bins with
    ``q = 0`` on A and ``q = 1`` on B.  The committor is the natural reaction
    coordinate and its sensitivity to individual species is the quantity that
    yields experimentally testable predictions.

quasi-potential
    ``-log`` of the weight-weighted stationary occupancy of each bin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Accumulators:
    """Running estimators over generations of a weighted-ensemble run."""

    n_bins: int
    tau_gen: float
    flux_history: list[float] = field(default_factory=list)
    occupancy: np.ndarray = None  # (n_bins,)
    transitions: np.ndarray = None  # (n_bins, n_bins)
    n_generations: int = 0

    def __post_init__(self) -> None:
        if self.occupancy is None:
            self.occupancy = np.zeros(self.n_bins, dtype=np.float64)
        if self.transitions is None:
            self.transitions = np.zeros((self.n_bins, self.n_bins), dtype=np.float64)

    # ------------------------------------------------------------------ #

    def record(
        self,
        bin_before: np.ndarray,
        bin_after: np.ndarray,
        weights: np.ndarray,
        recycled_weight: float,
    ) -> None:
        np.add.at(self.transitions, (bin_before, bin_after), weights)
        np.add.at(self.occupancy, bin_after, weights)
        self.flux_history.append(recycled_weight / self.tau_gen)
        self.n_generations += 1

    # ------------------------------------------------------------------ #

    def flux(self, burn_in: float = 0.2) -> tuple[float, float]:
        """Mean flux and blocked standard error, discarding a burn-in fraction."""
        f = np.asarray(self.flux_history, dtype=np.float64)
        if f.size == 0:
            return float("nan"), float("nan")
        start = int(burn_in * f.size)
        f = f[start:]
        if f.size < 2:
            return float(f.mean()) if f.size else float("nan"), float("nan")
        n_blocks = max(2, min(20, f.size // 5))
        blocks = np.array_split(f, n_blocks)
        means = np.array([b.mean() for b in blocks if b.size])
        return float(f.mean()), float(means.std(ddof=1) / np.sqrt(means.size))

    def mfpt(self, burn_in: float = 0.2) -> tuple[float, float]:
        """Mean first passage time and its propagated standard error."""
        mu, se = self.flux(burn_in)
        if not np.isfinite(mu) or mu <= 0:
            return float("inf"), float("nan")
        return 1.0 / mu, se / mu**2

    # ------------------------------------------------------------------ #

    def markov_matrix(self, regularise: float = 0.0) -> np.ndarray:
        t = self.transitions + regularise
        rows = t.sum(axis=1, keepdims=True)
        out = np.zeros_like(t)
        nz = rows[:, 0] > 0
        out[nz] = t[nz] / rows[nz]
        out[~nz, np.arange(self.n_bins)[~nz]] = 1.0  # unvisited bins are absorbing
        return out

    def committor(self, state_a: int, state_b: int, regularise: float = 1e-12) -> np.ndarray:
        """Forward committor q_i = P(reach B before A | start in bin i)."""
        t = self.markov_matrix(regularise)
        n = self.n_bins
        q = np.zeros(n, dtype=np.float64)
        q[state_b] = 1.0
        inner = np.array([i for i in range(n) if i not in (state_a, state_b)])
        if inner.size == 0:
            return q
        a_mat = np.eye(inner.size) - t[np.ix_(inner, inner)]
        b_vec = t[np.ix_(inner, [state_b])].ravel()
        try:
            q[inner] = np.linalg.solve(a_mat, b_vec)
        except np.linalg.LinAlgError:
            q[inner] = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
        return np.clip(q, 0.0, 1.0)

    def quasi_potential(self) -> np.ndarray:
        """``-log`` of normalised bin occupancy; ``inf`` where never visited."""
        occ = self.occupancy / max(self.occupancy.sum(), 1e-300)
        with np.errstate(divide="ignore"):
            return -np.log(occ)

    # ------------------------------------------------------------------ #

    def summary(self, state_a: int, state_b: int) -> dict:
        mfpt, mfpt_se = self.mfpt()
        flux, flux_se = self.flux()
        return {
            "generations": self.n_generations,
            "tau_generation": self.tau_gen,
            "flux": flux,
            "flux_stderr": flux_se,
            "mfpt": mfpt,
            "mfpt_stderr": mfpt_se,
            "committor": self.committor(state_a, state_b).tolist(),
            "quasi_potential": np.where(
                np.isfinite(self.quasi_potential()), self.quasi_potential(), None
            ).tolist(),
            "occupancy": self.occupancy.tolist(),
        }
