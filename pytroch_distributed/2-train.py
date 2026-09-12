"""
Benchmark training BERT on synthetic data.

This script expects to be launched once per GPU, from a Slurm submission script
with the necessary environment variables set (RANK, LOCAL_RANK, WORLD_SIZE,
MASTER_ADDR, MASTER_PORT).

Two run modes are available, selected by RUN_MODE below:

  "steps" -- run exactly TRAINING_STEPS iterations. Use this when comparing
             configurations, since every run performs identical work. The step
             count must be calibrated against the hardware; see TRAINING_STEPS.

  "time"  -- run until TARGET_SECONDS of wall-clock time have elapsed. Use this
             when a fixed job duration matters more than a fixed workload.
             Throughput (steps per second) is the figure to compare between
             runs in this mode.

Progress is reported every LOG_EVERY steps with an estimate of the time
remaining, so a run that will not finish within its walltime can be identified
and cancelled early.

"""

import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import transformers
from huggingface_hub import snapshot_download
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertForSequenceClassification, BertTokenizer

transformers.utils.logging.set_verbosity_error()

MODEL_ID = "google-bert/bert-base-uncased"
CACHE_DIR = "/projects/public/brics/data/hf/hub"  # Use pre-downloaded models
BACKEND = "nccl"  # change to 'nccl' to use NCCL backend

# --- Workload configuration -------------------------------------------------
# Global batch, split evenly across ranks. Must be an exact multiple of the
# world size. 8192 across 8 GPUs gives 1024 sequences per GPU, which uses a
# reasonable fraction of a GH200 and improves the ratio of compute to
# communication relative to a smaller batch.
BATCH_SIZE = 8192

# Fixed padding length. Padding to a constant length keeps the cost of every
# step identical, which makes timings comparable across runs and GPU counts.
SEQ_LENGTH = 128

# --- Duration configuration -------------------------------------------------
RUN_MODE = "time"  # "steps" or "time"

# Used when RUN_MODE == "steps".
# This MUST be calibrated against the hardware and backend in use. Run once with
# TRAINING_STEPS = 200 and read the reported seconds-per-step, then set
#     TRAINING_STEPS = target_seconds / seconds_per_step
# An uncalibrated value will either finish in moments or exceed the walltime.
TRAINING_STEPS = 200

# Used when RUN_MODE == "time".
TARGET_SECONDS = 30 * 60
# How often to check the clock. All ranks must agree on when to stop, so the
# check involves a collective; doing it every step would distort the timing.
# The run will overshoot TARGET_SECONDS by up to this many steps, so allow
# headroom in the Slurm --time request.
CHECK_EVERY = 50

# --- Logging ----------------------------------------------------------------
# Progress is printed by rank 0 every this many steps.
LOG_EVERY = 50

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
DEVICE = f"cuda:{LOCAL_RANK}"


def init_process(backend):
    """
    Initialise distributed training with the provided backend.

    The world size and local rank are discovered from environment variables.

    The current device is bound before anything else touches CUDA. If the
    process group is created first, every rank creates a bare context on the
    default device (cuda:0) in addition to its own, wasting several hundred
    megabytes per rank and initialising devices the rank should not be using.
    """
    torch.cuda.set_device(LOCAL_RANK)

    print(
        f"Initializing distributed training rank {os.environ.get('RANK')} "
        f"with backend: {backend} on device: {DEVICE}",
        flush=True,
    )
    # Join this process to the process group, using the specified backend
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=60 * 5),
        world_size=int(os.environ["WORLD_SIZE"]),
    )

    # We only want to print this once; only do so in the main process
    # (i.e. the one with global rank 0)
    if dist.get_rank() == 0:
        world_size = dist.get_world_size()
        print(
            f"Distributed training initialized with {world_size} processes "
            f"using backend {backend}.",
            flush=True,
        )


def collective_device(backend):
    """
    Return the device on which small control-plane tensors should live.

    NCCL operates on device tensors; gloo is more reliable with host tensors.
    """
    return DEVICE if backend == "nccl" else "cpu"


def training_step(model, optimizer, inputs, labels):
    """Perform a single forward and backward pass, and an optimizer step."""
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)  # forward pass
    loss = outputs.loss
    loss.backward()  # backward pass, synchronises gradients across ranks
    optimizer.step()


def log_progress(step, total, loop_start):
    """Print the step rate and an estimate of the time remaining."""
    torch.cuda.synchronize()
    elapsed = time.time() - loop_start
    per_step = elapsed / step

    message = f"step {step}"
    if total is not None:
        remaining = (total - step) * per_step
        message += f"/{total} ({remaining / 60:.1f} min remaining)"
    else:
        message += f" ({elapsed / 60:.1f} min elapsed)"

    print(f"{message}, {per_step:.3f} s/step", flush=True)


def run_fixed_steps(model, optimizer, inputs, labels, start_time):
    """Run exactly TRAINING_STEPS iterations. Returns the number of steps run."""
    rank = dist.get_rank()

    for step in range(1, TRAINING_STEPS + 1):
        training_step(model, optimizer, inputs, labels)

        if rank == 0 and step % LOG_EVERY == 0:
            log_progress(step, TRAINING_STEPS, start_time)

    return TRAINING_STEPS


def run_fixed_time(model, optimizer, inputs, labels, start_time):
    """
    Run until TARGET_SECONDS have elapsed. Returns the number of steps run.

    Every rank must leave the loop on the same iteration, otherwise the ranks
    still in the loop will block indefinitely on the gradient all-reduce. Rank 0
    owns the clock and broadcasts its reading to everyone else.
    """
    rank = dist.get_rank()
    device = collective_device(BACKEND)
    step = 0

    while True:
        training_step(model, optimizer, inputs, labels)
        step += 1

        if step % LOG_EVERY == 0 and rank == 0:
            log_progress(step, None, start_time)

        if step % CHECK_EVERY == 0:
            elapsed = torch.tensor(
                [time.time() - start_time], dtype=torch.float64, device=device
            )
            dist.broadcast(elapsed, src=0)
            if elapsed.item() >= TARGET_SECONDS:
                break

    return step


def benchmark():
    """
    Run a simple training-loop benchmark on synthetic data.

    Each rank (process) runs on a slice of the global batch.
    Reports the total wall-clock time and the achieved throughput (from rank 0).
    """
    # The BERT model is pre-downloaded
    model_path = snapshot_download(
        repo_id=MODEL_ID, cache_dir=CACHE_DIR, local_files_only=True
    )
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Find the global rank and world size of the current process
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Split the batch size evenly across ranks
    # This assumes the batch size is exactly divisible by the world size
    per_gpu_batch_size = BATCH_SIZE // world_size
    if per_gpu_batch_size * world_size != BATCH_SIZE:
        raise ValueError(
            f"{BATCH_SIZE=} but {world_size=}; BATCH_SIZE must be an exact "
            f"multiple of world size.\n"
            f"Instead, {BATCH_SIZE / world_size=}"
        )

    if RUN_MODE not in ("steps", "time"):
        raise ValueError(f"{RUN_MODE=} must be either 'steps' or 'time'.")

    # Only print this once per training run
    # i.e. print it only with the first (global rank == 0) process
    if rank == 0:
        duration = (
            f"{TRAINING_STEPS} steps"
            if RUN_MODE == "steps"
            else f"{TARGET_SECONDS} seconds"
        )
        print(
            f"Running benchmark with world size: {world_size}, "
            f"batch size: {BATCH_SIZE}, per GPU batch size: {per_gpu_batch_size}, "
            f"sequence length: {SEQ_LENGTH}, target: {duration}",
            flush=True,
        )

    # Get the model into this process' GPU, then wrap it with DDP (distributed
    # data parallel). This tells us to synchronise gradients from all our
    # processes during the backwards pass.
    model = BertForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model = DDP(model, device_ids=[LOCAL_RANK])
    optimizer = torch.optim.Adam(model.parameters())

    # Separate data per worker. Note that this uses the *global* rank: using the
    # local rank would give every node an identical set of slices.
    start_idx = rank * per_gpu_batch_size
    end_idx = start_idx + per_gpu_batch_size

    # Create synthetic training data, padded to a fixed length so that the cost
    # of each step is constant.
    texts = [
        f"This is sample sentence {i} for benchmarking BERT."
        for i in range(start_idx, end_idx)
    ]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=SEQ_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    labels = torch.ones(per_gpu_batch_size, dtype=torch.long).to(DEVICE)

    # Warm up: the first step pays for CUDA context creation, kernel
    # autotuning and DDP bucket allocation, none of which belong in the timing.
    if rank == 0:
        print("Running warm-up step...", flush=True)
    training_step(model, optimizer, inputs, labels)
    torch.cuda.synchronize()

    # Wait for all ranks to finish before we start timing
    dist.barrier()
    start_time = time.time()

    if RUN_MODE == "steps":
        steps_run = run_fixed_steps(model, optimizer, inputs, labels, start_time)
    else:
        steps_run = run_fixed_time(model, optimizer, inputs, labels, start_time)

    # Ensure all queued kernels have actually completed before we stop the clock
    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()

    # Only report the time in the main process
    if rank == 0:
        elapsed = end_time - start_time
        print(
            f"Time taken for {steps_run} forward and backward pass(es) with "
            f"BATCH_SIZE={BATCH_SIZE} on {world_size} workers: {elapsed} seconds",
            flush=True,
        )
        print(
            f"Seconds per step: {elapsed / steps_run}",
            flush=True,
        )
        print(
            f"Throughput: {steps_run / elapsed} steps/second, "
            f"{steps_run * BATCH_SIZE / elapsed} sequences/second",
            flush=True,
        )


if __name__ == "__main__":
    init_process(BACKEND)
    benchmark()
    dist.destroy_process_group()
