"""Test harness for transpose (naive + tiled). Set KERNEL_FILE=solution.cu to test the reference."""
import os
import pathlib
import sys

import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from reference import transpose_ref  # noqa: E402

KERNEL_FILE = os.environ.get("KERNEL_FILE", "kernel.cu")
print(f"Compiling {KERNEL_FILE} (first run JITs nvcc, ~30s; cached after).")
mod = load(
    name=f"transpose_{pathlib.Path(KERNEL_FILE).stem}",
    sources=[KERNEL_FILE],
    verbose=False,
)


def test_correctness(name: str, fn, M: int, N: int) -> bool:
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    expected = transpose_ref(x)
    actual = fn(x)
    try:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(f"  [OK]    {name:<6}  M={M:>5} N={N:>5}")
        return True
    except AssertionError as e:
        print(f"  [WRONG] {name:<6}  M={M:>5} N={N:>5}")
        print(f"          {str(e).splitlines()[0]}")
        return False


def benchmark(name: str, fn, M: int, N: int) -> None:
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)

    for _ in range(10):
        fn(x)
        x.t().contiguous()
    torch.cuda.synchronize()

    iters = 50
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(x)
    stop.record()
    torch.cuda.synchronize()
    mine_ms = start.elapsed_time(stop) / iters

    start.record()
    for _ in range(iters):
        x.t().contiguous()
    stop.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(stop) / iters

    print(f"  {name:<6}  yours: {mine_ms:.4f} ms  torch: {torch_ms:.4f} ms  ratio: {mine_ms / torch_ms:.2f}x")


print("\nCorrectness — naive:")
shapes = [(1, 1), (32, 32), (33, 31), (1024, 512), (513, 1023), (2048, 2048)]
ok_naive = all(test_correctness("naive", mod.transpose_naive, M, N) for M, N in shapes)

print("\nCorrectness — tiled:")
ok_tiled = all(test_correctness("tiled", mod.transpose_tiled, M, N) for M, N in shapes)

print("\nBenchmark (4096 x 4096):")
benchmark("naive", mod.transpose_naive, 4096, 4096)
benchmark("tiled", mod.transpose_tiled, 4096, 4096)

ok = ok_naive and ok_tiled
print("\n" + ("✅ correct" if ok else "❌ wrong"))
sys.exit(0 if ok else 1)
