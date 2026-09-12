"""
Benchmark training BERT on some synthetic data.

This script expects to be launched once per GPU, from a Slurm submission script
with the necessary environment variables set.

"""

import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import snapshot_download
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
from datetime import timedelta
import os

transformers.utils.logging.set_verbosity_error()

MODEL_ID = "google-bert/bert-base-uncased"
CACHE_DIR = "/projects/public/brics/data/hf/hub"  # Use pre-downloaded models
BACKEND = "gloo"  # change to 'nccl' to use NCCL backend
BATCH_SIZE = 204800
NUM_SAMPLES = 49152  # Total number of samples to process
TRAINING_STEPS = NUM_SAMPLES // BATCH_SIZE
DEVICE = f"cuda:{os.environ['LOCAL_RANK']}"


def init_process(backend):
    """
    Initialise distributed training with the provided backend.

    The world size and local rank are discovered from environment variables.
    """
    print(
        f"Initializing distributed training rank {os.environ.get('RANK')} with backend: {backend} on device: {DEVICE}"
    )
    # Join this process to the process group, using the specified backend
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=60 * 5),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # We only want to print this once; only do so in the main process (i.e. the one with global rank 0)
    if dist.get_rank() == 0:
        world_size = dist.get_world_size()
        print(
            f"Distributed training initialized with {world_size} processes using backend {backend}."
        )


def benchmark():
    """
    Run a simple training-loop benchmark on synthetic data.

    Each rank (process) runs on a slice of the global batch.
    Reports the total wall-clock time (from rank 0).

    The number of training steps is specified in TRAINING_STEPS (see the top of this file).
    """
    # The BERT model is pre-downloaded
    model_path = snapshot_download(
        repo_id=MODEL_ID, cache_dir=CACHE_DIR, local_files_only=True
    )
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Find the local and global ranks of the current process
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Split the batch size evenly across ranks
    # This assumes the batch size is exactly divisible by the world size (number of processes)
    per_gpu_batch_size = BATCH_SIZE // world_size
    if per_gpu_batch_size * world_size != BATCH_SIZE:
        raise ValueError(
            f"{BATCH_SIZE=} but {world_size=}; BATCH_SIZE must be an exact multiple of world size.\n"
            f"Instead, {BATCH_SIZE/world_size=}"
        )

    # Only print this once per training run
    # i.e. print it only with the first (global rank == 0) process
    if rank == 0:
        print(
            f"Running benchmark with world size: {world_size}, batch size: {BATCH_SIZE}, per GPU batch size: {per_gpu_batch_size}, training steps: {TRAINING_STEPS}"
        )

    # Get the model into this process' GPU, then wrap it with DDP (distributed data parallel)
    # This tells us to synchronise gradients from all our processes during the backwards pass
    model = BertForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters())

    # Separate data per worker
    start_idx = local_rank * per_gpu_batch_size
    end_idx = start_idx + per_gpu_batch_size
    # Create synthetic training data
    texts = [
        f"This is sample sentence {i} for benchmarking BERT."
        for i in range(start_idx, end_idx)
    ]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        use_fast_tokenizer=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    labels = torch.ones(per_gpu_batch_size, dtype=torch.long).to(DEVICE)

    # Wait for all ranks to finish before we start timing
    dist.barrier()
    start_time = time.time()

    for _ in range(TRAINING_STEPS):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)  # forward pass
        loss = outputs.loss
        loss.backward()  # backward pass
        optimizer.step()

    # Wait for all ranks to finish before we stop timing
    dist.barrier()
    end_time = time.time()

    # Only report the time in the main process
    if dist.get_rank() == 0:
        print(
            f"Time taken for {TRAINING_STEPS} forward and backward pass(es) with BATCH_SIZE={BATCH_SIZE} on {world_size} workers: {end_time - start_time} seconds"
        )


if __name__ == "__main__":
    init_process(BACKEND)
    benchmark()
    dist.destroy_process_group()
