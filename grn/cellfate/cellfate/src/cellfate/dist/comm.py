"""Multi-GPU and multi-node coordination for weighted ensemble.

Design
------
Propagation is embarrassingly parallel: each rank owns a shard of the walker
population and advances it with no communication at all.  This is where more
than 95 per cent of wall time is spent.

Resampling is not embarrassingly parallel: it must be performed over the global
population of each bin, or the weights are wrong.  Rather than gathering every
walker to one rank, each bin is assigned an owner rank and walkers migrate to
their bin's owner with a variable-length all-to-all.  Walker payloads are tiny
(``n_species`` integers plus one double), so this exchange moves kilobytes to
megabytes per generation, which the NVLink and Slingshot fabric absorbs without
appearing in the profile.

Bin ownership is recomputed every generation by greedy longest-processing-time
assignment over the global bin occupancy.  The computation is deterministic and
depends only on the globally reduced occupancy vector, so every rank derives the
identical mapping with no broadcast.
"""

from __future__ import annotations

import os

import numpy as np
import torch


class Communicator:
    """Single-process fallback.  Every method is the identity."""

    rank = 0
    world_size = 1
    is_distributed = False

    def owned_bins(self, global_counts: np.ndarray) -> np.ndarray:
        return np.nonzero(global_counts > 0)[0]

    def bin_counts(self, bin_id: torch.Tensor, n_bins: int) -> np.ndarray:
        return np.bincount(bin_id.detach().cpu().numpy(), minlength=n_bins).astype(np.int64)

    def redistribute(self, x, w, bin_id, n_bins):
        return x, w, bin_id

    def all_reduce_sum(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def all_reduce_scalar(self, v: float) -> float:
        return v

    def barrier(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def _lpt_assign(counts: np.ndarray, world_size: int) -> np.ndarray:
    """Greedy longest-processing-time bin-to-rank assignment.

    Deterministic given ``counts``, so all ranks compute the same mapping.
    """
    owner = np.zeros(len(counts), dtype=np.int64)
    load = np.zeros(world_size, dtype=np.int64)
    order = np.argsort(-counts, kind="stable")
    for b in order:
        r = int(np.argmin(load))
        owner[b] = r
        load[r] += int(counts[b])
    return owner


class TorchDistCommunicator(Communicator):
    """Communicator backed by ``torch.distributed`` (NCCL on GPU)."""

    is_distributed = True

    def __init__(self, backend: str | None = None, device: torch.device | None = None):
        import torch.distributed as dist

        self.dist = dist
        if not dist.is_initialized():
            if backend is None:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = device or (
            torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._owner: np.ndarray | None = None

    # ------------------------------------------------------------------ #

    def bin_counts(self, bin_id: torch.Tensor, n_bins: int) -> np.ndarray:
        local = torch.bincount(bin_id, minlength=n_bins).to(torch.int64).to(self.device)
        self.dist.all_reduce(local, op=self.dist.ReduceOp.SUM)
        return local.cpu().numpy()

    def owned_bins(self, global_counts: np.ndarray) -> np.ndarray:
        self._owner = _lpt_assign(global_counts, self.world_size)
        return np.nonzero((self._owner == self.rank) & (global_counts > 0))[0]

    # ------------------------------------------------------------------ #

    def redistribute(self, x: torch.Tensor, w: torch.Tensor, bin_id: torch.Tensor, n_bins: int):
        """Move every walker to the rank owning its bin."""
        if self._owner is None:
            raise RuntimeError("call owned_bins() before redistribute()")
        owner_t = torch.as_tensor(self._owner, dtype=torch.int64, device=bin_id.device)
        dest = owner_t[bin_id]

        order = torch.argsort(dest, stable=True)
        x, w, bin_id, dest = x[order], w[order], bin_id[order], dest[order]

        send_counts = torch.bincount(dest, minlength=self.world_size).to(torch.int64).to(self.device)
        recv_counts = torch.empty_like(send_counts)
        self.dist.all_to_all_single(recv_counts, send_counts)
        send = send_counts.cpu().tolist()
        recv = recv_counts.cpu().tolist()
        n_recv = int(sum(recv))
        n_sp = x.shape[1]

        x_flat = x.to(torch.int64).contiguous().view(-1)
        x_out = torch.empty(n_recv * n_sp, dtype=torch.int64, device=x.device)
        self.dist.all_to_all_single(
            x_out,
            x_flat,
            output_split_sizes=[c * n_sp for c in recv],
            input_split_sizes=[c * n_sp for c in send],
        )

        w_out = torch.empty(n_recv, dtype=w.dtype, device=w.device)
        self.dist.all_to_all_single(
            w_out, w.contiguous(), output_split_sizes=recv, input_split_sizes=send
        )

        b_out = torch.empty(n_recv, dtype=torch.int64, device=bin_id.device)
        self.dist.all_to_all_single(
            b_out, bin_id.contiguous(), output_split_sizes=recv, input_split_sizes=send
        )

        return x_out.view(n_recv, n_sp), w_out, b_out

    # ------------------------------------------------------------------ #

    def all_reduce_sum(self, arr: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(arr, dtype=torch.float64, device=self.device)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.SUM)
        return t.cpu().numpy().reshape(arr.shape)

    def all_reduce_scalar(self, v: float) -> float:
        t = torch.tensor([float(v)], dtype=torch.float64, device=self.device)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.SUM)
        return float(t.item())

    def barrier(self) -> None:
        self.dist.barrier()

    def shutdown(self) -> None:
        if self.dist.is_initialized():
            self.dist.destroy_process_group()


def make_communicator(distributed: bool | None = None, device=None) -> Communicator:
    """Build a communicator, auto-detecting a ``torchrun`` environment."""
    if distributed is None:
        distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    return TorchDistCommunicator(device=device) if distributed else Communicator()
