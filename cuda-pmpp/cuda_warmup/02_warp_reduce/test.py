"""Test harness for warp_reduce_sum. Set KERNEL_FILE=solution.cu to test the reference."""
import os
import pathlib
import sys

import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from reference import warp_reduce_sum_ref  # noqa: E402

KERNEL_FILE = os.environ.get("KERNEL_FILE", "kernel.cu")
print(f"Compiling {KERNEL_FILE} (first run JITs nvcc, ~30s; cached after).")
mod = load(
    name=f"warp_reduce_{pathlib.Path(KERNEL_FILE).stem}",
    sources=[KERNEL_FILE],
    verbose=False,
)


def test_correctness(n: int) -> bool:
    assert n % 32 == 0
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    expected = warp_reduce_sum_ref(x)
    actual = mod.warp_reduce_sum(x)
    try:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        print(f"  [OK]    n={n:>10}")
        return True
    except AssertionError as e:
        print(f"  [WRONG] n={n:>10}")
        print(f"          {str(e).splitlines()[0]}")
        return False


def benchmark(n: int) -> None:
    x = torch.randn(n, device="cuda", dtype=torch.float32)

    for _ in range(10):
        mod.warp_reduce_sum(x)
        x.view(-1, 32).sum(dim=1)
    torch.cuda.synchronize()

    iters = 100
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        mod.warp_reduce_sum(x)
    stop.record()
    torch.cuda.synchronize()
    mine_ms = start.elapsed_time(stop) / iters

    start.record()
    for _ in range(iters):
        x.view(-1, 32).sum(dim=1)
    stop.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(stop) / iters

    print(f"  yours:  {mine_ms:.4f} ms")
    print(f"  torch:  {torch_ms:.4f} ms")
    print(f"  ratio:  {mine_ms / torch_ms:.2f}x")


print("\nCorrectness:")
sizes = [32, 64, 256, 1024, 32 * 1000, 1 << 20, 32 * 31_337]
ok = all(test_correctness(s) for s in sizes)

print(f"\nBenchmark (n={1 << 22}):")
benchmark(1 << 22)

print("\n" + ("✅ correct" if ok else "❌ wrong"))
sys.exit(0 if ok else 1)
