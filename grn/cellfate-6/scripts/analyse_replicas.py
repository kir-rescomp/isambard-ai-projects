#!/usr/bin/env python3
"""Turn a directory of replica runs into the scientific output.

    python scripts/analyse_replicas.py results/step3_multinode/<jobid>

Produces, from the JSON documents and transition matrices already written:

committor
    Pooled across replicas, with a band from the spread between them.  The
    committor is the natural reaction coordinate of the transition, and the bin
    where it crosses 0.5 is the stochastic barrier top, which is generally not
    where the deterministic saddle sits.

quasi-potential
    ``-log`` of the weight-weighted stationary occupancy.  This is the landscape
    figure.

rate summary
    Per-replica MFPT, the combined estimate, and a comparison of the spread
    between replicas against the blocked standard error claimed within each
    one.  If those two disagree, the within-run error bars are not trustworthy
    and should not be quoted.

Writes a CSV of the per-bin quantities and, when matplotlib is available, a
two-panel figure.  No plotting dependency is required for the numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def committor_from_transitions(t: np.ndarray, state_a: int, state_b: int) -> np.ndarray:
    n = t.shape[0]
    rows = t.sum(axis=1, keepdims=True)
    p = np.zeros_like(t)
    nz = rows[:, 0] > 0
    p[nz] = t[nz] / rows[nz]
    p[~nz, np.arange(n)[~nz]] = 1.0

    q = np.zeros(n)
    q[state_b] = 1.0
    inner = np.array([i for i in range(n) if i not in (state_a, state_b)])
    if inner.size == 0:
        return q
    a_mat = np.eye(inner.size) - p[np.ix_(inner, inner)]
    b_vec = p[np.ix_(inner, [state_b])].ravel()
    try:
        q[inner] = np.linalg.solve(a_mat, b_vec)
    except np.linalg.LinAlgError:
        q[inner] = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
    return np.clip(q, 0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory")
    ap.add_argument("--out", default=None, help="output prefix (default: <directory>/analysis)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    d = Path(args.directory)
    docs = sorted(d.glob("*.json"))
    docs = [p for p in docs if not p.name.startswith("analysis")]
    if not docs:
        print(f"no run documents found in {d}", file=sys.stderr)
        return 1

    runs, transitions = [], []
    for p in docs:
        blob = json.loads(p.read_text())
        runs.append(blob)
        tp = p.with_suffix(".transitions.npy")
        if tp.exists():
            transitions.append(np.load(tp))

    mfpt = np.array([r["summary"]["mfpt"] for r in runs], dtype=float)
    flux = np.array([r["summary"]["flux"] for r in runs], dtype=float)
    within = np.array([r["summary"]["mfpt_stderr"] for r in runs], dtype=float)
    good = np.isfinite(flux) & (flux > 0)

    print(f"replicas found          : {good.sum()} of {len(runs)}")
    mean_flux = flux[good].mean()
    combined = 1.0 / mean_flux
    if good.sum() > 1:
        se_flux = flux[good].std(ddof=1) / np.sqrt(good.sum())
        print(f"combined MFPT           : {combined:.4e} +/- {se_flux/mean_flux**2:.3e}")
        spread = mfpt[good].std(ddof=1)
        print(f"spread between replicas : {spread:.3e}")
        print(f"mean within-run stderr  : {within[good].mean():.3e}")
        ratio = spread / within[good].mean()
        verdict = "calibrated" if 0.5 < ratio < 2.0 else "NOT calibrated, do not quote within-run error bars"
        print(f"ratio                   : {ratio:.2f}  ({verdict})")
    else:
        print(f"combined MFPT           : {combined:.4e}")

    # Physical framing: the model has no absolute clock, so report the rate in
    # units of the protein lifetime, which is what makes it comparable to
    # experiment.
    model = runs[0].get("model", {})
    d_p = model.get("d_p", 0.1)
    print(f"\nprotein lifetime        : {1/d_p:g} time units")
    print(f"MFPT in protein lifetimes: {combined * d_p:.3e}")

    if not transitions:
        print("\nno transition matrices found; skipping committor and landscape")
        return 0

    n_bins = transitions[0].shape[0]
    state_a, state_b = 0, n_bins - 1

    per_replica_q = np.array(
        [committor_from_transitions(t, state_a, state_b) for t in transitions]
    )
    pooled_q = committor_from_transitions(np.sum(transitions, axis=0), state_a, state_b)
    q_lo, q_hi = per_replica_q.min(axis=0), per_replica_q.max(axis=0)

    occ = np.array([r["summary"]["occupancy"] for r in runs], dtype=float).sum(axis=0)
    occ = occ / occ.sum()
    with np.errstate(divide="ignore"):
        qp = -np.log(occ)
    qp = qp - np.nanmin(qp[np.isfinite(qp)])

    crossing = int(np.argmin(np.abs(pooled_q - 0.5)))
    print(f"\nbins                    : {n_bins}")
    print(f"committor crosses 0.5 at: bin {crossing}")
    print(f"barrier height          : {np.nanmax(qp[np.isfinite(qp)]):.2f} kT-equivalent units")

    prefix = Path(args.out) if args.out else d / "analysis"
    hdr = "bin,committor,committor_min,committor_max,occupancy,quasi_potential"
    rows = np.column_stack([np.arange(n_bins), pooled_q, q_lo, q_hi, occ, qp])
    np.savetxt(f"{prefix}.csv", rows, delimiter=",", header=hdr, comments="", fmt="%.6g")
    print(f"\nwritten: {prefix}.csv")

    if args.no_plot:
        return 0
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; numbers written, figure skipped")
        return 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.fill_between(np.arange(n_bins), q_lo, q_hi, alpha=0.25, label="replica range")
    ax1.plot(np.arange(n_bins), pooled_q, lw=2, label="pooled")
    ax1.axhline(0.5, ls=":", lw=1, color="k")
    ax1.axvline(crossing, ls=":", lw=1, color="k")
    ax1.set_ylabel("committor  q(x)")
    ax1.set_title("Treg to Th17 transition: committor and landscape")
    ax1.legend(frameon=False)

    finite = np.isfinite(qp)
    ax2.plot(np.arange(n_bins)[finite], qp[finite], lw=2)
    ax2.axvline(crossing, ls=":", lw=1, color="k")
    ax2.set_xlabel("progress coordinate bin   (0 = Treg, %d = Th17)" % (n_bins - 1))
    ax2.set_ylabel("quasi-potential  -log p")
    fig.tight_layout()
    fig.savefig(f"{prefix}.png", dpi=150)
    print(f"written: {prefix}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
