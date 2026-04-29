"""Test harness for causal QK^T. Set KERNEL_FILE=solution.cu to test the reference."""
import os
import pathlib
import sys

import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from reference import causal_qk_ref  # noqa: E402

# Disable TF32 so PyTorch's reference uses true float32 — apples-to-apples with our kernel.
torch.backends.cuda.matmul.allow_tf32 = False

KERNEL_FILE = os.environ.get("KERNEL_FILE", "kernel.cu")
print(f"Compiling {KERNEL_FILE} (first run JITs nvcc, ~30s; cached after).")
mod = load(
    name=f"causal_qk_{pathlib.Path(KERNEL_FILE).stem}",
    sources=[KERNEL_FILE],
    verbose=False,
)


def test_correctness(N: int, D: int) -> bool:
    q = torch.randn(N, D, device="cuda", dtype=torch.float32)
    k = torch.randn(N, D, device="cuda", dtype=torch.float32)
    expected = causal_qk_ref(q, k)
    actual = mod.causal_qk(q, k)
    rtol, atol = 1e-3, 1e-3
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
        print(f"  [OK]    N={N:>5} D={D:>5}")
        return True
    except AssertionError as e:
        print(f"  [WRONG] N={N:>5} D={D:>5}")
        print(f"          {str(e).splitlines()[0]}")
        return False


def benchmark(N: int, D: int) -> None:
    q = torch.randn(N, D, device="cuda", dtype=torch.float32)
    k = torch.randn(N, D, device="cuda", dtype=torch.float32)

    for _ in range(5):
        mod.causal_qk(q, k)
        causal_qk_ref(q, k)
    torch.cuda.synchronize()

    iters = 20
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        mod.causal_qk(q, k)
    stop.record()
    torch.cuda.synchronize()
    mine_ms = start.elapsed_time(stop) / iters

    start.record()
    for _ in range(iters):
        causal_qk_ref(q, k)
    stop.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(stop) / iters

    flops = 2.0 * N * N * D  # ignores the savings from masking; mirrors a dense GEMM count
    print(f"  N={N} D={D}")
    print(f"    yours:  {mine_ms:.3f} ms  ({flops / mine_ms / 1e9:.2f} TFLOP/s)")
    print(f"    torch:  {torch_ms:.3f} ms  ({flops / torch_ms / 1e9:.2f} TFLOP/s)")
    print(f"    ratio:  {mine_ms / torch_ms:.2f}x")


print("\nCorrectness:")
shapes = [
    (32, 32),
    (64, 32),
    (33, 17),       # ragged, < BN/BD
    (128, 64),
    (257, 65),      # ragged
    (1024, 128),
]
ok = all(test_correctness(N, D) for N, D in shapes)

print("\nBenchmark:")
benchmark(1024, 128)

print("\n" + ("✅ correct" if ok else "❌ wrong"))
sys.exit(0 if ok else 1)
