#!/usr/bin/env python3
"""Minimal multi-node NCCL check.  Depends on nothing but torch.

Run this before any cellfate multi-node job.  It isolates fabric and launcher
configuration from the application entirely: if this fails, the problem is NCCL
or Slurm, not the simulator; if it passes, the transport is sound and any later
failure is in cellfate.

It exercises exactly the two collectives cellfate uses:

    all_reduce         flux and diagnostic reduction, tiny payload
    all_to_all_single  walker migration to bin-owning ranks, variable splits

Launch with srun, one task per GPU:

    srun --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 \\
         python scripts/nccl_smoke_test.py
"""

from __future__ import annotations

import datetime
import os
import socket
import subprocess
import sys

import torch
import torch.distributed as dist


def bootstrap() -> tuple[int, int, int]:
    """Derive rank, world size and local rank from Slurm, without torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(
            os.environ.get("LOCAL_RANK", 0)
        )
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local = int(os.environ.get("SLURM_LOCALID", 0))
    if "MASTER_ADDR" not in os.environ:
        nodelist = os.environ["SLURM_JOB_NODELIST"]
        head = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist]
        ).decode().split()[0]
        os.environ["MASTER_ADDR"] = head
    os.environ.setdefault(
        "MASTER_PORT", str(20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000)
    )
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(local)
    return rank, world, local


def main() -> int:
    rank, world, local = bootstrap()

    if rank == 0:
        print("=== environment ===", flush=True)
        for k in (
            "MASTER_ADDR", "MASTER_PORT", "NCCL_SOCKET_IFNAME", "NCCL_NET",
            "NCCL_IB_DISABLE", "FI_PROVIDER", "LD_LIBRARY_PATH",
        ):
            print(f"  {k}={os.environ.get(k, '<unset>')}", flush=True)
        print("=== interfaces on this node ===", flush=True)
        try:
            print(subprocess.check_output(["ip", "-o", "-4", "addr"]).decode(), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  could not list interfaces: {exc}", flush=True)

    torch.cuda.set_device(local)
    print(
        f"[rank {rank:3d}/{world}] host={socket.gethostname()} local={local} "
        f"gpu={torch.cuda.get_device_name(local)}",
        flush=True,
    )

    # A short timeout so a bootstrap failure reports quickly instead of hanging
    # for the default thirty minutes and then being killed by Slurm.
    kwargs = {"backend": "nccl", "timeout": datetime.timedelta(seconds=120)}
    try:
        dist.init_process_group(device_id=torch.device(f"cuda:{local}"), **kwargs)
    except TypeError:
        dist.init_process_group(**kwargs)
    if rank == 0:
        print(f"process group up: {dist.get_backend()}, world size {dist.get_world_size()}", flush=True)
        # The single most useful line in this output.  A path under
        # site-packages/nvidia/nccl means the wheel-bundled NCCL won and the
        # brics/nccl module did not take effect, so there is no libfabric
        # plugin and the Slingshot fabric is unreachable.
        lib = "<not found>"
        try:
            with open(f"/proc/{os.getpid()}/maps") as fh:
                for line in fh:
                    if "libnccl.so" in line:
                        lib = line.split()[-1]
                        break
        except OSError:
            pass
        print(f"libnccl in use: {lib}", flush=True)
        if "site-packages" in lib:
            print("  WARNING: this is the wheel-bundled NCCL. Run 'module load brics/nccl' "
                  "AFTER activating the venv.", flush=True)

    dev = torch.device(f"cuda:{local}")

    t = torch.full((1024,), float(rank), device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = float(sum(range(world)))
    ok_ar = abs(float(t[0]) - expected) < 1e-6
    if rank == 0:
        print(f"all_reduce: got {float(t[0])}, expected {expected} -> {'OK' if ok_ar else 'WRONG'}",
              flush=True)

    # Variable-split all_to_all, the pattern cellfate uses for walker migration.
    send_counts = torch.tensor([rank + 1] * world, dtype=torch.int64, device=dev)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    send = send_counts.cpu().tolist()
    recv = recv_counts.cpu().tolist()
    payload = torch.full((sum(send),), float(rank), device=dev)
    out = torch.empty(sum(recv), device=dev)
    dist.all_to_all_single(out, payload, output_split_sizes=recv, input_split_sizes=send)
    ok_a2a = out.numel() == sum(recv)
    if rank == 0:
        print(f"all_to_all_single (variable splits): {'OK' if ok_a2a else 'WRONG'}", flush=True)

    # Bandwidth, so a silent fallback to slow TCP is visible rather than merely
    # correct.  On Slingshot this should be tens of GB/s, not hundreds of MB/s.
    n = 64 * 1024 * 1024
    buf = torch.empty(n, dtype=torch.float32, device=dev)
    for _ in range(3):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    iters = 10
    for _ in range(iters):
        dist.all_reduce(buf)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    gb = buf.numel() * buf.element_size() / 2**30
    busbw = gb * 2 * (world - 1) / world / (ms / 1e3)
    if rank == 0:
        print("", flush=True)
        print("=" * 60, flush=True)
        print(f"  BUS BANDWIDTH: {busbw:.1f} GB/s   ({gb:.2f} GiB all_reduce in {ms:.2f} ms)", flush=True)
        print("=" * 60, flush=True)
        if busbw < 5.0:
            print("  WARNING: that is TCP-fallback territory, not Slingshot. "
                  "Check NCCL_SOCKET_IFNAME and the aws-ofi-nccl plugin.", flush=True)

    dist.barrier()
    if rank == 0:
        print("=== all checks completed ===", flush=True)
    dist.destroy_process_group()
    return 0 if (ok_ar and ok_a2a) else 1


if __name__ == "__main__":
    sys.exit(main())
