"""Test harness for tiled GEMM. Set KERNEL_FILE=solution.cu to test the reference."""
import os
import pathlib
import sys

import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from reference import gemm_ref  # noqa: E402

# Disable TF32 so PyTorch's reference uses true float32 — apples-to-apples with our kernel.
torch.backends.cuda.matmul.allow_tf32 = False

KERNEL_FILE = os.environ.get("KERNEL_FILE", "kernel.cu")
print(f"Compiling {KERNEL_FILE} (first run JITs nvcc, ~30s; cached after).")
mod = load(
    name=f"gemm_{pathlib.Path(KERNEL_FILE).stem}",
    sources=[KERNEL_FILE],
    verbose=False,
)


def test_correctness(M: int, K: int, N: int) -> bool:
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    expected = gemm_ref(a, b)
    actual = mod.gemm(a, b)
    # Different reduction orders cause small fp32 drift, especially as K grows.
    rtol, atol = 1e-3, 1e-3
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        print(f"  [OK]    M={M:>5} K={K:>5} N={N:>5}")
        return True
    except AssertionError as e:
        print(f"  [WRONG] M={M:>5} K={K:>5} N={N:>5}")
        print(f"          {str(e).splitlines()[0]}")
        return False


def benchmark(M: int, K: int, N: int) -> None:
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    for _ in range(5):
        mod.gemm(a, b)
        a @ b
    torch.cuda.synchronize()

    iters = 20
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        mod.gemm(a, b)
    stop.record()
    torch.cuda.synchronize()
    mine_ms = start.elapsed_time(stop) / iters

    start.record()
    for _ in range(iters):
        a @ b
    stop.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(stop) / iters

    flops = 2.0 * M * N * K
    # flops / (ms * 1e-3 s/ms) / 1e12 = TFLOP/s
    print(f"  M={M} K={K} N={N}")
    print(f"    yours:  {mine_ms:.3f} ms  ({flops / mine_ms / 1e9:.2f} TFLOP/s)")
    print(f"    torch:  {torch_ms:.3f} ms  ({flops / torch_ms / 1e9:.2f} TFLOP/s)")
    print(f"    ratio:  {mine_ms / torch_ms:.2f}x")


print("\nCorrectness:")
shapes = [
    (32, 32, 32),
    (64, 32, 64),
    (33, 17, 31),       # all dims < BM/BN/BK
    (128, 256, 128),
    (513, 257, 129),    # ragged
    (1024, 1024, 1024),
]
ok = all(test_correctness(M, K, N) for M, K, N in shapes)

print("\nBenchmark:")
benchmark(1024, 1024, 1024)

print("\n" + ("✅ correct" if ok else "❌ wrong"))
sys.exit(0 if ok else 1)
