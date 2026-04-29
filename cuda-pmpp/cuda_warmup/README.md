# CUDA Warmup — 6 kernels

A structured warmup before a GPU engineering interview. Six kernels in increasing
order of difficulty. For each one, fill in the body of `kernel.cu` and run `test.py`
to validate against a PyTorch reference.

## Order and concepts

1. **`01_vec_add`** — kernel launch syntax, `blockIdx`/`threadIdx`/`blockDim`, bounds checks. (~5 min)
2. **`02_warp_reduce`** — warp shuffles (`__shfl_down_sync`), the `0xFFFFFFFF` mask, lane arithmetic. (~10 min)
3. **`03_block_reduce`** — `__shared__` memory, tree reduction, `__syncthreads()`, `atomicAdd` for cross-block. (~15 min)
4. **`04_transpose`** — 2D thread/block indexing, shared-memory tiling, `+1` padding to dodge bank conflicts. (~20 min)
5. **`05_tiled_gemm`** — the canonical GEMM tiling pattern: shared-memory tiles for A and B, register accumulation, K-loop. (~30–40 min)
6. **`06_causal_attn`** — causal attention scores `S = Q @ K^T` with `-inf` on the strict upper triangle. Tiled GEMM with the second operand transposed plus a causal mask; optional whole-tile skip when the entire output tile sits above the diagonal. (~30–40 min)

## How to run

From within a kernel directory:

```bash
cd 01_vec_add
python test.py                       # tests YOUR kernel.cu
KERNEL_FILE=solution.cu python test.py   # tests the reference solution
```

The first run will JIT-compile via `nvcc` (~30 seconds). Subsequent runs reuse the
cached binary unless the source file changed.

## If you're stuck

Try for at least 10 minutes first. The struggle is the point. After that, peek at
`solution.cu` — but read it once, close it, and write your own version from memory.

## Notes

- All kernels use `float32`. No fp16/bf16, no Tensor Cores.
- Architecture is auto-detected by PyTorch's JIT. Works on any modern NVIDIA GPU
  (sm_70+). Your interview target may be H100 (sm_90) but the patterns are identical.
- Each `kernel.cu` is self-contained: kernel + host launcher + PyTorch binding.
- The host-side launchers are pre-filled. Focus on the kernel body.
