import os

import torch
import nvtx


local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(
    backend="nccl",
    rank=rank,
    world_size=int(os.environ["WORLD_SIZE"]),
    device_id=local_rank,
)

device = torch.device("cuda", local_rank)
m = 2048
x_shape = (m, m)
w1_shape = (x_shape[1], m)
w2_shape = (w1_shape[1], 1)
lr = 1e-8

# Keep model weights identical across ranks; use rank-local data.
torch.manual_seed(0)
w1 = torch.randn(w1_shape, device=device)
w2 = torch.randn(w2_shape, device=device)

torch.manual_seed(rank)
x = torch.randn(x_shape, device=device)
target = torch.randn((w2_shape[1],), device=device)


@nvtx.annotate(color="red")
def train_step():
    # FORWARD PASS
    z = x @ w1
    h = torch.relu(z)
    y = h @ w2

    # GRAD DERIVATION IN learning-plan.md
    grad_y = y - target
    grad_w2 = h.T @ grad_y
    w2_work = torch.distributed.all_reduce(
        grad_w2, op=torch.distributed.ReduceOp.AVG, async_op=True
    )

    grad_h = grad_y @ w2.T
    grad_z = grad_h * (z > 0)
    grad_w1 = x.T @ grad_z
    w1_work = torch.distributed.all_reduce(
        grad_w1, op=torch.distributed.ReduceOp.AVG, async_op=True
    )

    # UPDATE WEIGHTS
    w2_work.wait()
    w2.add_(grad_w2, alpha=-lr)

    w1_work.wait()
    w1.add_(grad_w1, alpha=-lr)


def profiled_region():
    for _ in range(10):
        train_step()


# Warmup
profiled_region()

torch.distributed.barrier()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
try:
    profiled_region()
    torch.distributed.barrier()
    torch.cuda.synchronize()
finally:
    torch.cuda.cudart().cudaProfilerStop()
    torch.distributed.destroy_process_group()


print("finished!")
