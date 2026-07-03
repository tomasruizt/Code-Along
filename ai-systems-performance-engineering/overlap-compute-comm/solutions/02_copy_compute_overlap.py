import torch
import nvtx

device = "cuda"

M = N = K = 2 * 1024
A = torch.randn(M, K, device=device)
B = torch.randn(K, N, device=device)
C = torch.empty(M, N, device=device)

BSZ = 1
D_h = torch.randn(BSZ, M, K, device="cpu", pin_memory=True)

custom_stream = torch.cuda.Stream()
default_stream = torch.cuda.default_stream()


@nvtx.annotate(color="green")
def matmul():
    torch.matmul(A, B, out=C)
    return C


@nvtx.annotate(color="orange")
def copy_D():
    D_d = D_h.to(device, non_blocking=True)
    return D_d


@nvtx.annotate(color="red")
def sequential():
    C = matmul()
    D = copy_D()
    return C + D


@nvtx.annotate(color="red")
def overlapped():
    C = matmul()
    with torch.cuda.stream(custom_stream):
        D = copy_D()
    default_stream.wait_stream(custom_stream)
    return C + D


def assert_correctness():
    expected = sequential()
    actual = overlapped()
    torch.testing.assert_close(actual, expected)
    print("✅ overlapped output is equivalent")


def profiled_region():
    n = 10
    for _ in range(n):
        sequential()

    torch.cuda.synchronize()

    for _ in range(n):
        overlapped()


# Check correctness
assert_correctness()

# Warmup
profiled_region()

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
try:
    profiled_region()
    torch.cuda.synchronize()
finally:
    torch.cuda.cudart().cudaProfilerStop()
