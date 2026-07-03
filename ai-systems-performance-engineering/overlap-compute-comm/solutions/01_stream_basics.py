import torch
import nvtx

device = "cuda"
M = N = K = 1024

a0 = torch.rand(M, K, device=device)
b0 = torch.rand(K, N, device=device)
a1 = torch.rand(M, K, device=device)
b1 = torch.rand(K, N, device=device)

out0 = a0 @ b0
out1 = a1 @ b1

s0 = torch.cuda.Stream()
s1 = torch.cuda.Stream()

N_LOOPS = 10


@nvtx.annotate(color="green")
def matmul0():
    torch.matmul(a0, b0, out=out0)


@nvtx.annotate(color="orange")
def matmul1():
    torch.matmul(a1, b1, out=out1)


@nvtx.annotate(color="red")
def sequential():
    # sequentially
    for _ in range(N_LOOPS):
        matmul0()
    for _ in range(N_LOOPS):
        matmul1()


@nvtx.annotate(color="red")
def in_different_streams(wait_for_previous_stream=False):
    with torch.cuda.stream(s0):
        for _ in range(N_LOOPS):
            matmul0()

    if wait_for_previous_stream:
        s1.wait_stream(s0)

    with torch.cuda.stream(s1):
        for _ in range(N_LOOPS):
            matmul1()


# warmup
sequential()
in_different_streams()
in_different_streams(wait_for_previous_stream=True)

try:
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    sequential()
    torch.cuda.synchronize()
    in_different_streams()
    torch.cuda.synchronize()
    in_different_streams(wait_for_previous_stream=True)
    torch.cuda.synchronize()
finally:
    torch.cuda.cudart().cudaProfilerStop()
