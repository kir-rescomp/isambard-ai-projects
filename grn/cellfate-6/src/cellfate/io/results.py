"""Result serialisation.

Raw trajectories are never written.  A production campaign generates on the
order of 10^12 walker-steps; storing them would be a petabyte-scale mistake and
would saturate the parallel filesystem long before it filled it.  What is
written instead is the reduced scientific output of each run: the flux history,
the bin occupancy, the transition matrix, the committor, and the run metadata
needed to reproduce it.

One JSON document per run, plus an optional Parquet table aggregating a sweep.
Rank 0 writes; other ranks write nothing.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _nccl_path() -> str:
    """Which libnccl is mapped into this process.

    Recorded with every run because it is the difference between using the
    Slingshot fabric and silently not using it.  A path under
    site-packages/nvidia/nccl means the wheel-bundled build won and the site
    module did not take effect.
    """
    try:
        from ..dist.slurm import nccl_library_in_use

        return nccl_library_in_use()
    except Exception:  # noqa: BLE001
        return "unknown"


def provenance(extra: dict | None = None) -> dict:
    """Everything needed to reproduce a run, recorded with the run itself."""
    import torch

    info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "libnccl": _nccl_path(),
    }
    if extra:
        info.update(extra)
    return info


def write_run(
    path: str | Path,
    summary: dict,
    logs: list,
    config: dict,
    model_params: dict,
    transitions: np.ndarray | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "provenance": provenance(),
        "config": config,
        "model": model_params,
        "summary": summary,
        "generations": [asdict(g) if hasattr(g, "__dataclass_fields__") else g for g in logs],
    }
    path.write_text(json.dumps(doc, indent=2, default=float))
    if transitions is not None:
        np.save(path.with_suffix(".transitions.npy"), transitions)
    return path


def aggregate_sweep(run_dir: str | Path, out: str | Path) -> Path:
    """Collapse a directory of run JSON documents into one table."""
    rows = []
    for p in sorted(Path(run_dir).glob("*.json")):
        d = json.loads(p.read_text())
        row = {"run": p.stem}
        row.update({f"model.{k}": v for k, v in d.get("model", {}).items()})
        s = d.get("summary", {})
        for k in ("flux", "flux_stderr", "mfpt", "mfpt_stderr", "generations"):
            row[k] = s.get(k)
        rows.append(row)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), out.with_suffix(".parquet"))
        return out.with_suffix(".parquet")
    except ImportError:
        import csv

        out = out.with_suffix(".csv")
        with out.open("w", newline="") as fh:
            if rows:
                wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                wtr.writeheader()
                wtr.writerows(rows)
        return out
