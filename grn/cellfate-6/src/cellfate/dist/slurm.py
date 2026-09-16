"""Launch under srun without torchrun.

On HPE Cray EX systems the nested ``srun`` then ``torchrun`` pattern adds a
layer of elastic agents between Slurm and the workers.  When something goes
wrong there every agent receives SIGTERM at once and the log becomes a wall of
``SignalException: signal 15`` with the original error in another stream.
Mapping ranks straight onto Slurm tasks removes that layer.

Rank source, in priority order:

``PMI_RANK``
    Set when srun is invoked with ``--mpi=pmi2``.  This is the value the
    known-good Isambard-AI recipe uses and it is preferred here for that reason.
``SLURM_PROCID``
    Equivalent in the common case and used when PMI is not initialised.

World size prefers ``SLURM_GPUS`` (set when the job requests ``--gpus=<total>``)
and falls back to ``SLURM_NTASKS``.

Note that this module cannot fix the NCCL library itself.  On Isambard-AI the
site build must be loaded with ``module load brics/nccl``; the NCCL bundled in
the PyTorch wheel has no libfabric plugin and therefore cannot use the Slingshot
fabric.  Single-node jobs succeed regardless because NCCL stays on NVLink, which
is why a multi-node failure is the first symptom.
"""

from __future__ import annotations

import os
import subprocess


def bootstrap_from_slurm(force: bool = False) -> bool:
    """Populate RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR and MASTER_PORT.

    Returns True if the environment now describes a distributed run.  A
    torchrun-provided environment is left untouched unless ``force`` is set, so
    both launchers keep working.
    """
    if not force and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_PROCID" not in os.environ and "PMI_RANK" not in os.environ:
        return False

    rank = os.environ.get("PMI_RANK") or os.environ["SLURM_PROCID"]
    world = int(os.environ.get("SLURM_GPUS") or os.environ.get("SLURM_NTASKS", "1"))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")

    if "MASTER_ADDR" not in os.environ:
        nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
        if nodelist:
            head = subprocess.check_output(
                ["scontrol", "show", "hostnames", nodelist]
            ).decode().split()[0]
            os.environ["MASTER_ADDR"] = head
    os.environ.setdefault("MASTER_PORT", "29500")
    return world > 1


def nccl_library_in_use() -> str:
    """Path of the libnccl actually loaded, for confirming the module took effect.

    A path under ``site-packages/nvidia/nccl`` means the wheel-bundled build won
    and the Slingshot plugin is absent; a system or module path means
    ``brics/nccl`` is in force.
    """
    try:
        with open(f"/proc/{os.getpid()}/maps") as fh:
            for line in fh:
                if "libnccl.so" in line:
                    return line.split()[-1]
    except OSError:
        pass
    return "<not loaded yet>"
