"""The weighted-ensemble generation loop.

One generation is:

1. propagate every walker for ``tau_gen`` by ``n_substeps`` tau-leaps
   (no communication)
2. evaluate the progress coordinate and assign bins
3. recycle walkers that have entered the target basin, banking their weight as
   probability flux and returning them to the source basin
4. accumulate the bin-to-bin transition matrix
5. redistribute walkers to bin-owning ranks (one all-to-all)
6. resample within each bin to the target walker count
7. reduce flux and diagnostics across ranks

Steps 1 and 2 dominate the wall clock.  Steps 5 to 7 are the only communication
and scale with the number of bins, not the number of walkers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..dist.comm import Communicator, make_communicator
from ..engine.torch_backend import TorchEngine
from ..model import CompiledNetwork
from .resample import BinScheme, resample
from .stats import Accumulators


@dataclass
class WEConfig:
    """Run configuration."""

    tau: float = 0.01  # leap size
    n_substeps: int = 100  # leaps per generation; tau_gen = tau * n_substeps
    n_generations: int = 200
    walkers_per_bin: int = 64
    initial_walkers: int = 4096
    resampling: str = "systematic"
    recycle: bool = True
    seed: int = 0
    replicas: int = 1  # independent ensembles; world_size must be divisible by it
    log_every: int = 10
    checkpoint_every: int = 0  # 0 disables
    checkpoint_path: str | None = None

    @property
    def tau_gen(self) -> float:
        return self.tau * self.n_substeps


@dataclass
class GenerationLog:
    generation: int
    n_walkers: int
    total_weight: float
    occupied_bins: int
    recycled_weight: float
    flux: float
    mfpt: float
    wall_seconds: float
    subdivisions: int
    clamped: int
    max_bin_reached: int = 0
    extra: dict = field(default_factory=dict)


class WeightedEnsemble:
    """Weighted-ensemble sampler for a compiled reaction network."""

    def __init__(
        self,
        net: CompiledNetwork,
        engine: TorchEngine,
        bins: BinScheme,
        pcoord: Callable[[torch.Tensor], torch.Tensor],
        basin_a_states: torch.Tensor,
        config: WEConfig,
        comm: Communicator | None = None,
    ):
        self.net = net
        self.engine = engine
        self.bins = bins
        self.pcoord = pcoord
        self.config = config
        self.comm = comm if comm is not None else make_communicator(device=engine.device)
        self.device = engine.device

        self.generator = torch.Generator(device=self.device)
        # Seed from the GLOBAL rank, not the within-ensemble rank.  Replicas
        # must be statistically independent, and seeding on the group-local
        # rank would give every replica an identical random stream and a
        # spuriously perfect agreement between them.
        self.generator.manual_seed(
            config.seed + 7919 * getattr(self.comm, "global_rank", self.comm.rank)
        )

        self.basin_a = basin_a_states.to(self.device)
        # Each replica is a COMPLETE ensemble carrying total probability 1.0,
        # so the population is divided by the group size, not the world size.
        # Normalising by world_size instead gives every replica total weight
        # 1/n_replicas, which scales its flux down by the same factor and
        # inflates the mean first passage time proportionally.
        n_local = max(1, config.initial_walkers // self.comm.group_size)
        idx = torch.randint(
            0, self.basin_a.shape[0], (n_local,), device=self.device, generator=self.generator
        )
        self.x = self.basin_a.index_select(0, idx).clone()
        self.w = torch.full(
            (n_local,), 1.0 / (n_local * self.comm.group_size), dtype=torch.float64, device=self.device
        )

        if bins.n_bins < self.comm.group_size:
            raise ValueError(
                f"{self.comm.group_size} ranks share one ensemble but the progress "
                f"coordinate has only {bins.n_bins} bins.  Bins are the unit of "
                f"ownership, so ranks beyond the bin count would receive no walkers.  "
                f"Either raise bins.n_bins to at least {self.comm.group_size}, or set "
                f"we.replicas so that world_size / replicas <= n_bins, or use "
                f"'cellfate sweep --shard-by-rank' which needs no collectives at all."
            )

        if self.comm.rank == 0 and self.device.type == "cuda":
            projected = config.walkers_per_bin * bins.n_bins / max(1, self.comm.world_size)
            if projected < 20_000:
                print(
                    f"note: about {projected:,.0f} walkers per GPU. Measured on GH200, a "
                    f"generation is dominated by fixed per-generation cost below roughly "
                    f"20,000 walkers per GPU, so raising walkers_per_bin from "
                    f"{config.walkers_per_bin} costs almost no wall time and shrinks the "
                    f"error bars as its square root.",
                    flush=True,
                )

        self.acc = Accumulators(n_bins=bins.n_bins, tau_gen=config.tau_gen)
        self.logs: list[GenerationLog] = []
        self._gen = 0

    # ------------------------------------------------------------------ #

    def _recycle(self, bin_id: torch.Tensor) -> float:
        """Return walkers in the target basin to the source basin."""
        hit = (bin_id == self.bins.state_b).nonzero(as_tuple=True)[0]
        if hit.numel() == 0:
            return 0.0
        recycled = float(self.w.index_select(0, hit).sum())
        src = torch.randint(
            0, self.basin_a.shape[0], (hit.numel(),), device=self.device, generator=self.generator
        )
        self.x.index_copy_(0, hit, self.basin_a.index_select(0, src))
        bin_id.index_fill_(0, hit, self.bins.state_a)
        return recycled

    # ------------------------------------------------------------------ #

    def step(self) -> GenerationLog:
        cfg = self.config
        t0 = time.perf_counter()

        pc_before = self.pcoord(self.x)
        bin_before = self.bins.assign(pc_before)

        self.x, diag = self.engine.advance(
            self.x, cfg.tau, cfg.n_substeps, generator=self.generator
        )

        pc_after = self.pcoord(self.x)
        bin_after = self.bins.assign(pc_after)
        # A rank may hold no walkers this generation, which is normal early on
        # when few bins are occupied.  Local reductions must tolerate it; the
        # collectives below are still entered by every rank unconditionally,
        # which is what keeps the process group in step.
        max_bin = int(bin_after.max()) if bin_after.numel() else -1

        # The transition matrix must see the pre-recycling destination, or the
        # committor loses every path that actually reaches the target basin.
        bin_observed = bin_after.clone()
        recycled = self._recycle(bin_after) if cfg.recycle else 0.0

        self.acc.record(
            bin_before.detach().cpu().numpy(),
            bin_observed.detach().cpu().numpy(),
            self.w.detach().cpu().numpy(),
            self.comm.all_reduce_scalar(recycled),
        )

        counts = self.comm.bin_counts(bin_after, self.bins.n_bins)
        owned = self.comm.owned_bins(counts)
        if self.comm.is_distributed:
            self.x, self.w, bin_after = self.comm.redistribute(
                self.x, self.w, bin_after, self.bins.n_bins
            )

        keep = torch.zeros(self.bins.n_bins, dtype=torch.bool, device=self.device)
        if len(owned):
            keep[torch.as_tensor(owned, dtype=torch.int64, device=self.device)] = True
        mask = keep[bin_after]
        if not bool(mask.all()):
            sel = mask.nonzero(as_tuple=True)[0]
            self.x, self.w, bin_after = self.x[sel], self.w[sel], bin_after[sel]

        self.x, self.w, _ = resample(
            self.x,
            self.w,
            bin_after,
            self.bins.n_bins,
            cfg.walkers_per_bin,
            method=cfg.resampling,
            generator=self.generator,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        flux, _ = self.acc.flux()
        mfpt, _ = self.acc.mfpt()
        log = GenerationLog(
            generation=self._gen,
            n_walkers=int(self.comm.all_reduce_scalar(float(self.w.numel()))),
            total_weight=self.comm.all_reduce_scalar(float(self.w.sum())),
            occupied_bins=int((counts > 0).sum()),
            recycled_weight=recycled,
            flux=flux,
            mfpt=mfpt,
            wall_seconds=time.perf_counter() - t0,
            subdivisions=diag.subdivisions,
            clamped=diag.clamped,
            max_bin_reached=int(self.comm.all_reduce_max(max_bin)),
        )
        self.logs.append(log)
        self._gen += 1
        return log

    # ------------------------------------------------------------------ #

    def run(self, verbose: bool = True) -> Accumulators:
        cfg = self.config
        for g in range(cfg.n_generations):
            log = self.step()
            if verbose and self.comm.rank == 0 and (g % cfg.log_every == 0 or g == cfg.n_generations - 1):
                print(
                    f"gen {log.generation:5d}  walkers {log.n_walkers:8d}  "
                    f"bins {log.occupied_bins:4d}  weight {log.total_weight:.6e}  "
                    f"max_bin {log.max_bin_reached:3d}  flux {log.flux:.4e}  "
                    f"mfpt {log.mfpt:.4e}  {log.wall_seconds*1e3:7.1f} ms",
                    flush=True,
                )
            if cfg.checkpoint_every and (g + 1) % cfg.checkpoint_every == 0:
                self.save_checkpoint(cfg.checkpoint_path)
        return self.acc

    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: str | None) -> None:
        if not path:
            return
        p = f"{path}.rank{self.comm.rank}.pt"
        torch.save(
            {
                "generation": self._gen,
                "x": self.x.cpu(),
                "w": self.w.cpu(),
                "occupancy": self.acc.occupancy,
                "transitions": self.acc.transitions,
                "flux_history": self.acc.flux_history,
                "rng": self.generator.get_state(),
            },
            p,
        )

    def load_checkpoint(self, path: str) -> None:
        p = f"{path}.rank{self.comm.rank}.pt"
        blob = torch.load(p, map_location=self.device, weights_only=False)
        self._gen = blob["generation"]
        self.x = blob["x"].to(self.device)
        self.w = blob["w"].to(self.device)
        self.acc.occupancy = blob["occupancy"]
        self.acc.transitions = blob["transitions"]
        self.acc.flux_history = blob["flux_history"]
        self.acc.n_generations = len(self.acc.flux_history)
        self.generator.set_state(blob["rng"])


def brute_force_mfpt(
    engine: TorchEngine,
    bins: BinScheme,
    pcoord: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    tau: float,
    n_substeps: int,
    max_generations: int,
    seed: int = 0,
) -> tuple[float, int, int]:
    """Reference MFPT by direct simulation, used to validate the WE estimate.

    Returns ``(mfpt_estimate, n_hits, n_walkers)``.  Only usable when the barrier
    is low enough for direct simulation to see crossings, which is the whole
    reason the weighted ensemble exists.
    """
    gen = torch.Generator(device=engine.device)
    gen.manual_seed(seed)
    x = x0.clone()
    alive = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    first = torch.full((x.shape[0],), float("nan"), dtype=torch.float64, device=x.device)
    for g in range(max_generations):
        if not bool(alive.any()):
            break
        idx = alive.nonzero(as_tuple=True)[0]
        sub, _ = engine.advance(x.index_select(0, idx), tau, n_substeps, generator=gen)
        x.index_copy_(0, idx, sub)
        b = bins.assign(pcoord(x))
        hit = alive & (b == bins.state_b)
        first[hit] = (g + 1) * tau * n_substeps
        alive = alive & ~hit
    hits = torch.isfinite(first)
    n_hits = int(hits.sum())
    if n_hits == 0:
        return float("inf"), 0, int(x.shape[0])
    total_time = float(first[hits].sum()) + float((~hits).sum()) * max_generations * tau * n_substeps
    return total_time / n_hits, n_hits, int(x.shape[0])
