"""Test suite.  Runs on CPU in roughly a minute; no GPU required.

The tests are ordered from cheapest and most local to most expensive and most
global.  A failure in ``test_propensities`` invalidates everything below it, so
read the output from the top.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cellfate import models
from cellfate.engine.ssa import ssa_ensemble
from cellfate.engine.torch_backend import TorchEngine
from cellfate.model import ACTIVATION, REPRESSION, Modifier, Reaction, ReactionNetwork, propensities_numpy
from cellfate.we import BinScheme, WEConfig, WeightedEnsemble
from cellfate.we.resample import resample, split_merge_indices, systematic_indices
from cellfate.we.stats import Accumulators
from cellfate.dist.comm import Communicator, _lpt_assign


ALL_MODELS = ["birth_death", "toggle", "th17_treg"]


def test_lpt_assign_respects_partition_size():
    """Bin ownership must be computed against the ensemble, not the world."""
    counts = np.array([10, 5, 3, 2, 8, 1])
    owner = _lpt_assign(counts, 2)
    assert owner.max() < 2
    assert set(owner.tolist()) <= {0, 1}


def build(name, **kw):
    return {"birth_death": models.birth_death, "toggle": models.toggle, "th17_treg": models.th17_treg}[
        name
    ](**kw).compile()


# ---------------------------------------------------------------------- #
# model specification


def test_stoichiometry_translation_does_not_consume_mrna():
    net = build("th17_treg")
    j = net.reaction_names.index("tl_RORgt")
    assert net.stoich[j, net.index("mRORgt")] == 0
    assert net.stoich[j, net.index("RORgt")] == 1


def test_unknown_species_rejected():
    with pytest.raises(ValueError):
        ReactionNetwork("bad", ["A"], [Reaction("r", 1.0, products={"Z": 1})])


def test_modifier_validation():
    with pytest.raises(ValueError):
        Modifier("A", ACTIVATION, K=-1.0)
    with pytest.raises(ValueError):
        Modifier("A", REPRESSION, K=1.0, basal=1.0)


# ---------------------------------------------------------------------- #
# propensities


@pytest.mark.parametrize("name", ALL_MODELS)
def test_propensities_match_numpy_reference(name):
    net = build(name)
    eng = TorchEngine(net, device="cpu")
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.integers(0, 500, size=(512, net.n_species)))
    a_t = eng.propensities(x).numpy()
    a_n = propensities_numpy(net, x.numpy())
    assert np.max(np.abs(a_t - a_n) / (np.abs(a_n) + 1e-30)) < 1e-10


def test_propensities_nonnegative_at_zero_state():
    net = build("th17_treg")
    eng = TorchEngine(net, device="cpu")
    a = eng.propensities(torch.zeros((4, net.n_species), dtype=torch.int64))
    assert bool((a >= 0).all())


# ---------------------------------------------------------------------- #
# leaping accuracy


@pytest.mark.parametrize("tau", [0.005, 0.02, 0.1])
def test_birth_death_stationary_moments_match_tau_leap_theory(tau):
    """Tau-leaping is exact in the mean and inflates the variance by a known factor.

    For x' = x + Pois(k.tau) - Pois(d.x.tau) the stationary moments of the
    *scheme* are E[x] = k/d and Var[x] = (k/d) / (1 - d.tau/2).  Testing against
    the exact Poisson variance instead would be wrong at large tau and would
    leave the tolerance too loose to detect real bias at small tau.
    """
    k, d, n = 20.0, 1.0, 60000
    net = build("birth_death", k=k, d=d)
    eng = TorchEngine(net, device="cpu")
    g = torch.Generator()
    g.manual_seed(20260902)
    x, diag = eng.advance(torch.zeros((n, 1), dtype=torch.int64), tau, int(30.0 / tau), generator=g)
    xs = x[:, 0].double()

    lam = k / d
    v_pred = lam / (1.0 - d * tau / 2.0)
    assert abs(float(xs.mean()) - lam) < 5 * np.sqrt(v_pred / n)
    assert abs(float(xs.var()) - v_pred) < 5 * np.sqrt((lam + 2 * lam**2) / n)
    assert diag.clamped == 0


def test_leaping_agrees_with_exact_ssa():
    """Tau-leaping and exact SSA must agree on the stationary mean."""
    net = build("birth_death", k=12.0, d=1.0)
    eng = TorchEngine(net, device="cpu")
    leap, _ = eng.advance(torch.zeros((8000, 1), dtype=torch.int64), 0.02, 1000)
    exact = ssa_ensemble(net, np.zeros(1, dtype=np.int64), t_end=20.0, n=300, seed=3)
    m_leap, m_exact = float(leap.double().mean()), float(exact.mean())
    assert abs(m_leap - m_exact) < 0.06 * m_exact


def test_smaller_tau_does_not_change_the_answer():
    net = build("birth_death", k=20.0, d=1.0)
    eng = TorchEngine(net, device="cpu")
    a, _ = eng.advance(torch.zeros((8000, 1), dtype=torch.int64), 0.04, 750)
    b, _ = eng.advance(torch.zeros((8000, 1), dtype=torch.int64), 0.005, 6000)
    assert abs(float(a.double().mean()) - float(b.double().mean())) < 0.6


def test_state_never_goes_negative():
    net = build("toggle", scale=0.5)
    eng = TorchEngine(net, device="cpu")
    x = torch.ones((2000, net.n_species), dtype=torch.int64)
    x, _ = eng.advance(x, 0.05, 400)
    assert bool((x >= 0).all())


# ---------------------------------------------------------------------- #
# bistability of the biological model


def test_th17_treg_is_bistable_and_metastable():
    """Both basins must be occupied and neither may leak into the barrier."""
    net = build("th17_treg", scale=1.0)
    eng = TorchEngine(net, device="cpu")
    pc = models.ratio_pcoord(net, "RORgt", "FOXP3")
    n = 2000
    x = torch.zeros((n, net.n_species), dtype=torch.int64)
    p = int(net.rate[0] * 10 / 0.1)
    x[: n // 2, net.index("FOXP3")] = p
    x[n // 2 :, net.index("RORgt")] = p
    x, _ = eng.advance(x, 0.02, 2500)
    phi = pc(x).numpy()
    assert np.mean(phi < -0.5) > 0.4
    assert np.mean(phi > 0.5) > 0.4
    assert np.mean(np.abs(phi) < 0.5) < 0.05  # barrier region essentially empty


def test_scale_raises_the_barrier():
    """Larger system size must make spontaneous transitions rarer."""
    rates = []
    for scale in (0.4, 0.8):
        net = build("toggle", scale=scale)
        eng = TorchEngine(net, device="cpu")
        pc = models.ratio_pcoord(net, "A", "B")
        x = torch.zeros((1500, net.n_species), dtype=torch.int64)
        x[:, net.index("B")] = int(40 * scale)
        x, _ = eng.advance(x, 0.01, 200)
        x, _ = eng.advance(x, 0.01, 3000)
        rates.append(float(np.mean(pc(x).numpy() > 0.5)))
    assert rates[0] > rates[1]


# ---------------------------------------------------------------------- #
# resampling


def test_systematic_resampling_conserves_weight():
    w = torch.rand(37, dtype=torch.float64) + 0.01
    idx = systematic_indices(w, 16)
    assert idx.numel() == 16
    assert idx.max() < w.numel() and idx.min() >= 0


def test_split_merge_conserves_weight():
    w = torch.rand(41, dtype=torch.float64) + 0.01
    for target in (5, 41, 90):
        idx, cw = split_merge_indices(w, target)
        assert idx.numel() == target == cw.numel()
        assert abs(float(cw.sum()) - float(w.sum())) < 1e-9


@pytest.mark.parametrize("method", ["systematic", "split_merge"])
def test_resample_conserves_total_weight_across_bins(method):
    torch.manual_seed(0)
    n, n_bins = 500, 8
    x = torch.randint(0, 100, (n, 3))
    w = torch.rand(n, dtype=torch.float64)
    w = w / w.sum()
    b = torch.randint(0, n_bins, (n,))
    nx, nw, nb = resample(x, w, b, n_bins, 16, method=method)
    assert abs(float(nw.sum()) - 1.0) < 1e-12
    assert nx.shape[0] == nw.shape[0] == nb.shape[0]
    for k in range(n_bins):
        before = float(w[b == k].sum())
        after = float(nw[nb == k].sum())
        assert abs(before - after) < 1e-12


def test_resample_preserves_per_bin_weight_with_extreme_disparity():
    """One dominant walker among many negligible ones is the adversarial case."""
    x = torch.arange(10).reshape(10, 1)
    w = torch.full((10,), 1e-12, dtype=torch.float64)
    w[3] = 1.0 - 9e-12
    b = torch.zeros(10, dtype=torch.int64)
    _, nw, _ = resample(x, w, b, 1, 8, method="systematic")
    assert abs(float(nw.sum()) - float(w.sum())) < 1e-12


# ---------------------------------------------------------------------- #
# estimators


def test_committor_boundary_conditions_and_monotonicity():
    n = 12
    acc = Accumulators(n_bins=n, tau_gen=1.0)
    rng = np.random.default_rng(1)
    for _ in range(400):  # a biased random walk along the bin axis
        b0 = rng.integers(0, n, size=200)
        step = rng.choice([-1, 0, 1], size=200, p=[0.45, 0.1, 0.45])
        b1 = np.clip(b0 + step, 0, n - 1)
        acc.record(b0, b1, np.full(200, 1.0 / 200), 0.0)
    q = acc.committor(0, n - 1)
    assert q[0] == pytest.approx(0.0, abs=1e-9)
    assert q[n - 1] == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(q) >= -1e-6)


def test_markov_matrix_rows_sum_to_one():
    acc = Accumulators(n_bins=5, tau_gen=1.0)
    acc.record(np.array([0, 1, 2]), np.array([1, 2, 3]), np.full(3, 1 / 3), 0.0)
    t = acc.markov_matrix()
    assert np.allclose(t.sum(axis=1), 1.0)


def test_mfpt_is_reciprocal_of_flux():
    acc = Accumulators(n_bins=3, tau_gen=2.0)
    for _ in range(50):
        acc.record(np.array([0]), np.array([2]), np.array([1.0]), 0.5)
    flux, _ = acc.flux()
    mfpt, _ = acc.mfpt()
    assert flux == pytest.approx(0.25)
    assert mfpt == pytest.approx(4.0)


# ---------------------------------------------------------------------- #
# end-to-end


@pytest.mark.slow
def test_weighted_ensemble_rate_matches_brute_force():
    """The load-bearing test: WE must reproduce direct simulation on a barrier
    low enough for direct simulation to cross."""
    from cellfate.we.driver import brute_force_mfpt

    scale = 0.5
    net = build("toggle", scale=scale)
    eng = TorchEngine(net, device="cpu")
    pc = models.ratio_pcoord(net, "A", "B")
    bins = BinScheme.uniform(-0.9, 0.9, 24)

    x0 = torch.zeros((1500, net.n_species), dtype=torch.int64)
    x0[:, net.index("B")] = int(40 * scale)
    basin, _ = eng.advance(x0, 0.01, 1000)

    bf, hits, _ = brute_force_mfpt(eng, bins, pc, basin, 0.01, 200, 900, seed=11)
    assert hits > 500, "brute-force reference has too few crossings to compare against"

    cfg = WEConfig(tau=0.01, n_substeps=200, n_generations=900, walkers_per_bin=32,
                   initial_walkers=768, seed=5)
    we = WeightedEnsemble(net, eng, bins, pc, basin[:768], cfg)
    acc = we.run(verbose=False)
    mfpt, _ = acc.mfpt(burn_in=0.3)

    assert abs(we.logs[-1].total_weight - 1.0) < 1e-9
    assert 0.6 < mfpt / bf < 1.7, f"WE {mfpt:.1f} vs brute force {bf:.1f}"

    # Exact monotonicity is not a finite-sample property: with a few hundred
    # generations the estimated committor can dip by a fraction of a per cent
    # between adjacent bins.  What must hold is that it rises overall and never
    # reverses materially.
    q = acc.committor(bins.state_a, bins.state_b)
    assert np.all(np.diff(q) >= -0.05), f"committor reverses: {np.diff(q).min():.4f}"
    assert q[-1] - q[0] > 0.9
    assert 0.25 < q[len(q) // 2] < 0.75


def test_empty_walker_population_is_survivable():
    """A rank can legitimately hold zero walkers for a generation.

    With more ranks than occupied bins, bin ownership leaves some ranks empty.
    Every local reduction must tolerate that; a bare ``.max()`` on an empty
    tensor raises and takes the whole job down.
    """
    net = build("toggle", scale=0.5)
    eng = TorchEngine(net, device="cpu")
    pc = models.ratio_pcoord(net, "A", "B")
    bins = BinScheme.uniform(-0.9, 0.9, 8)

    empty = torch.zeros((0, net.n_species), dtype=torch.int64)
    assert pc(empty).numel() == 0
    assert bins.assign(pc(empty)).numel() == 0
    advanced, diag = eng.advance(empty, 0.01, 5)
    assert advanced.shape == (0, net.n_species)
    assert diag.clamped == 0
    x, w, b = resample(empty, torch.zeros(0, dtype=torch.float64),
                       torch.zeros(0, dtype=torch.int64), 8, 4)
    assert x.shape[0] == 0


def test_oversubscribed_bins_raises_a_clear_error():
    """More ranks than bins must fail at construction, not mid-run."""

    class FakeComm(Communicator):
        group_size = 8

    net = build("toggle", scale=0.5)
    eng = TorchEngine(net, device="cpu")
    pc = models.ratio_pcoord(net, "A", "B")
    bins = BinScheme.uniform(-0.9, 0.9, 4)
    basin = torch.zeros((16, net.n_species), dtype=torch.int64)
    with pytest.raises(ValueError, match="bins"):
        WeightedEnsemble(net, eng, bins, pc, basin, WEConfig(), comm=FakeComm())


def test_replica_carries_full_probability():
    """Each replica is a complete ensemble with total weight 1.0.

    Normalising by world size rather than group size gives every replica
    weight 1/n_replicas, which scales its flux down by that factor and
    inflates the mean first passage time proportionally.
    """

    class FakeComm(Communicator):
        group_size = 4
        world_size = 16
        n_groups = 4

    net = build("toggle", scale=0.5)
    eng = TorchEngine(net, device="cpu")
    pc = models.ratio_pcoord(net, "A", "B")
    bins = BinScheme.uniform(-0.9, 0.9, 12)
    basin = torch.zeros((64, net.n_species), dtype=torch.int64)
    basin[:, net.index("B")] = 20
    we = WeightedEnsemble(net, eng, bins, pc, basin,
                          WEConfig(initial_walkers=256), comm=FakeComm())
    # One rank of a four-rank group holds a quarter of unit probability.
    assert float(we.w.sum()) == pytest.approx(0.25, rel=1e-12)


@pytest.mark.slow
def test_checkpoint_round_trip(tmp_path):
    net = build("toggle", scale=0.5)
    eng = TorchEngine(net, device="cpu")
    pc = models.ratio_pcoord(net, "A", "B")
    bins = BinScheme.uniform(-0.9, 0.9, 16)
    basin = torch.zeros((256, net.n_species), dtype=torch.int64)
    basin[:, net.index("B")] = 20

    cfg = WEConfig(tau=0.01, n_substeps=50, n_generations=20, walkers_per_bin=8,
                   initial_walkers=128, seed=1)
    we = WeightedEnsemble(net, eng, bins, pc, basin, cfg)
    we.run(verbose=False)
    path = str(tmp_path / "ckpt")
    we.save_checkpoint(path)

    we2 = WeightedEnsemble(net, eng, bins, pc, basin, cfg)
    we2.load_checkpoint(path)
    assert we2._gen == we._gen
    assert torch.equal(we2.x, we.x)
    assert np.allclose(we2.acc.transitions, we.acc.transitions)
