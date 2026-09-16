"""Exact stochastic simulation algorithm, NumPy, single trajectory at a time.

This exists solely to validate the leaping backends.  It is deliberately slow
and deliberately simple: it is the ground truth, so it must be obviously
correct rather than fast.
"""

from __future__ import annotations

import numpy as np

from ..model import CompiledNetwork, propensities_numpy


def ssa_trajectory(
    net: CompiledNetwork,
    x0: np.ndarray,
    t_end: float,
    rng: np.random.Generator,
    record_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one exact trajectory by Gillespie's direct method.

    Returns ``(times, states)``.  If ``record_times`` is given the trajectory is
    sampled on that grid (piecewise-constant interpolation); otherwise every
    jump is returned.
    """
    x = np.asarray(x0, dtype=np.int64).copy()
    t = 0.0
    stoich = net.stoich.astype(np.int64)

    if record_times is not None:
        out = np.zeros((len(record_times), net.n_species), dtype=np.int64)
        ri = 0
    else:
        ts: list[float] = [0.0]
        xs: list[np.ndarray] = [x.copy()]

    while t < t_end:
        a = propensities_numpy(net, x[None, :])[0]
        a0 = a.sum()
        if a0 <= 0.0:
            break
        t_next = t + rng.exponential(1.0 / a0)
        if record_times is not None:
            while ri < len(record_times) and record_times[ri] < min(t_next, t_end):
                out[ri] = x
                ri += 1
        if t_next > t_end:
            break
        j = int(rng.choice(len(a), p=a / a0))
        x = x + stoich[j]
        t = t_next
        if record_times is None:
            ts.append(t)
            xs.append(x.copy())

    if record_times is not None:
        while ri < len(record_times):
            out[ri] = x
            ri += 1
        return record_times, out
    return np.asarray(ts), np.asarray(xs)


def ssa_ensemble(
    net: CompiledNetwork,
    x0: np.ndarray,
    t_end: float,
    n: int,
    seed: int = 0,
    record_times: np.ndarray | None = None,
) -> np.ndarray:
    """Endpoint (or gridded) states for ``n`` independent exact trajectories."""
    rng = np.random.default_rng(seed)
    if record_times is None:
        out = np.zeros((n, net.n_species), dtype=np.int64)
        for i in range(n):
            _, xs = ssa_trajectory(net, x0, t_end, rng)
            out[i] = xs[-1]
        return out
    out = np.zeros((n, len(record_times), net.n_species), dtype=np.int64)
    for i in range(n):
        _, xs = ssa_trajectory(net, x0, t_end, rng, record_times=record_times)
        out[i] = xs
    return out
