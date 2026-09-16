#!/usr/bin/env bash
# Environment build for Isambard-AI (aarch64 + GH200).
#
# The aarch64 wheel situation is the first thing to establish, not the last.
# If this script fails it fails here, in two minutes, rather than four hours
# into a 64-node allocation.
set -euo pipefail

SCRATCH=${PWD}
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module load cuda/12.6

# uv keeps the whole environment on scratch and resolves far faster than pip
# on a shared filesystem.  Fall back to python -m venv if uv is unavailable.
if command -v uv >/dev/null 2>&1; then
  uv venv "${SCRATCH}/venv" --python 3.11
  # shellcheck disable=SC1091
  source "${SCRATCH}/venv/bin/activate"
  uv pip install --index-url https://download.pytorch.org/whl/cu126 torch
  uv pip install numpy pyyaml pyarrow pytest
  uv pip install -e "${PROJECT}"
else
  python3 -m venv "${SCRATCH}/venv"
  # shellcheck disable=SC1091
  source "${SCRATCH}/venv/bin/activate"
  pip install --upgrade pip
  pip install --index-url https://download.pytorch.org/whl/cu126 torch
  pip install numpy pyyaml pyarrow pytest
  pip install -e "${PROJECT}"
fi

python - <<'PY'
import platform, torch
print("machine        :", platform.machine())
print("torch          :", torch.__version__)
print("torch cuda     :", torch.version.cuda)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         :", torch.cuda.get_device_name(0))
    print("capability     :", torch.cuda.get_device_capability(0))
    free, total = torch.cuda.mem_get_info()
    print(f"HBM            : {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free")
    print("nccl available :", torch.distributed.is_nccl_available())
PY

cat <<MSG

Environment at ${SCRATCH}/venv
Activate with:  module load cuda/12.6 && source ${SCRATCH}/venv/bin/activate

If the aarch64 CUDA wheel is unavailable for this driver, run inside the NVIDIA
PyTorch container instead.  cellfate itself has no compiled dependencies beyond
torch and numpy, so it installs cleanly into any working torch environment.
MSG
