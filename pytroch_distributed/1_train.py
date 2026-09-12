import os, time, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

model = torch.nn.Sequential(
    torch.nn.Linear(8192, 8192), torch.nn.ReLU(),
    torch.nn.Linear(8192, 8192), torch.nn.ReLU(),
    torch.nn.Linear(8192, 8192),
).to(device)
model = DDP(model, device_ids=[local_rank])
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

x = torch.randn(512, 8192, device=device)
y = torch.randn(512, 8192, device=device)
end = time.time() + 3600            # run ≥1 hour
step = 0
while time.time() < end:
    opt.zero_grad()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    opt.step()
    step += 1
    if step % 100 == 0 and dist.get_rank() == 0:
        print(f"step {step} loss {loss.item():.4f}", flush=True)

dist.destroy_process_group()
