# Overlapping Computation and Communication in PyTorch

## Learning Objective

Learn the CUDA and PyTorch primitives behind compute/communication overlap: CUDA streams, nonblocking copies, asynchronous NCCL collectives, and DDP-style gradient bucket reduction.

The standard for success is not that an API returns early. The standard is an Nsight Systems trace showing independent work overlapping on the GPU timeline: compute kernels on one stream while copy or NCCL work runs on another stream, with total time closer to `max(compute_time, comm_time)` than `compute_time + comm_time`.

All exercises are Modal-first, so no local GPU is required. The support files are:

- `modal/modal_overlap.py`: Modal runner for one-GPU and two-GPU profiling runs.
- `modal/nsys_wrapper.py`: per-rank Nsight Systems wrapper for distributed runs.
- `scripts/`: exercise implementations.
- `traces/`: downloaded `.nsys-rep` reports.

## Modal Setup

From this directory:

```bash
python -m pip install -r requirements.txt
modal setup
```

`modal setup` authenticates the local Modal CLI so `make exercise-*` can submit remote GPU jobs and download profiler reports.

Run exercises through Makefile targets. Each target runs the Modal profile and downloads the Nsight Systems report into `traces/`:

```bash
make exercise-01
make exercise-02
make exercise-03
make exercise-04
make profile-all
```

Open the downloaded `.nsys-rep` files from `traces/` in Nsight Systems.

## Profiling Contract

Each exercise script must explicitly mark the region to profile. The Modal runner uses Nsight Systems with `--capture-range=cudaProfilerApi`, so setup code is ignored until the script calls `cudaProfilerStart()`.

Structure each exercise like this:

```python
import torch


def profiled_region():
    # Put only the experiment variants here.
    ...


# Allocate tensors, initialize distributed state, and run warmup outside the profile.
...
torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStart()
try:
    profiled_region()
    torch.cuda.synchronize()
finally:
    torch.cuda.cudart().cudaProfilerStop()

# Print timings, checksums, and validation after profiling stops.
...
```

The learner's job is to keep imports, CUDA initialization, tensor allocation, warmup, and validation outside the profiler range. Only the code needed to prove overlap should run between `cudaProfilerStart()` and `cudaProfilerStop()`. Use NVTX ranges inside `profiled_region()` to label the phases that should appear in Nsight Systems.

For every exercise, the script should print a small summary after profiling: tensor sizes, iteration counts, baseline time, overlapped time, and any correctness checksum. These prints should happen after `cudaProfilerStop()` so they do not pollute the trace.

## Exercise 1: CUDA Stream Mechanics

File: `scripts/01_stream_basics.py`

Goal: learn how independent CUDA streams enqueue work, when they can overlap, and when explicit synchronization is required.

Implement:

- Build two independent PyTorch GEMM workloads (not a custom/tiled matmul). For example, allocate `a0, b0, a1, b1` on CUDA and repeatedly compute `out0 = a0 @ b0` and `out1 = a1 @ b1`.
- Choose matrix sizes and loop counts so each workload takes long enough to see in Nsight Systems, but not so large that a single GEMM monopolizes the whole GPU. Start with moderately sized square matrices and tune from there.
- Preallocate outputs with `torch.empty_like(...)` or explicit output tensors and use `torch.mm(a, b, out=out)` to avoid measuring allocation noise in every iteration.
- Baseline variant: run workload A, then workload B, both on the default CUDA stream. Concretely, this is two Python `for` loops or two calls to a helper like `run_gemm_loop("gemm_a", a0, b0, out0)` followed by `run_gemm_loop("gemm_b", a1, b1, out1)`.
- Overlap variant: create `stream_a = torch.cuda.Stream()` and `stream_b = torch.cuda.Stream()`. Submit workload A inside `with torch.cuda.stream(stream_a): ...` and workload B inside `with torch.cuda.stream(stream_b): ...`, then synchronize before reading results.
- Dependency variant: run the same two-stream code but force `stream_b.wait_stream(stream_a)` before launching workload B. This should make the timeline look sequential again.
- Add CUDA event timings for each variant and NVTX ranges named `sequential_a`, `sequential_b`, `stream_a`, `stream_b`, and `stream_dependency`. Wrap the A/B GEMM loops in NVTX ranges too, for example `gemm_a` and `gemm_b`, because otherwise identical cuBLAS matmul kernels are hard to distinguish in Nsight Systems.

Suggested shape:

```python
def gemm_loop(label, a, b, out, iters):
    with nvtx_range(label):
        for _ in range(iters):
            torch.mm(a, b, out=out)


def run_sequential():
    with nvtx_range("sequential_a"):
        gemm_loop("gemm_a", a0, b0, out0, iters)
    with nvtx_range("sequential_b"):
        gemm_loop("gemm_b", a1, b1, out1, iters)


def run_two_streams():
    with torch.cuda.stream(stream_a):
        with nvtx_range("stream_a"):
            gemm_loop("gemm_a", a0, b0, out0, iters)
    with torch.cuda.stream(stream_b):
        with nvtx_range("stream_b"):
            gemm_loop("gemm_b", a1, b1, out1, iters)
```

Run/profile:

```bash
make exercise-01
```

Prove:

- In the baseline trace, GEMM kernels from workload A finish before workload B starts.
- In the two-stream trace, kernels from workload A and workload B appear on different CUDA streams and overlap horizontally if the chosen matrix sizes allow concurrency.
- In the dependency trace, adding `stream_b.wait_stream(stream_a)` removes the overlap.
- CPU wall-clock timing without synchronization is misleading; CUDA event timing or explicit `torch.cuda.synchronize()` is required for meaningful timings.

## Exercise 2: Copy/Compute Overlap

File: `scripts/02_copy_compute_overlap.py`

Goal: learn when `non_blocking=True` and pinned host memory allow host-to-device copy to overlap with independent GPU compute.

Implement:

- Allocate a large CPU source tensor with `pin_memory=True`, a destination CUDA tensor or copied CUDA batch, and a separate CUDA compute workload that does not depend on the copied data.
- Baseline: copy CPU data to GPU, synchronize, then run the independent compute workload.
- Overlap version: launch the pinned-memory HtoD copy on `copy_stream` with `non_blocking=True`, then immediately run the independent compute workload on the default stream.
- Before consuming the copied tensor, make the default stream wait for `copy_stream`.
- Use `record_stream` for tensors whose lifetime crosses streams, especially copied CUDA tensors managed by PyTorch's caching allocator.
- NVTX ranges named `h2d_copy`, `independent_compute`, and `consume_copied_batch`.

Suggested synchronization shape:

```python
copy_stream = torch.cuda.Stream()

with torch.cuda.stream(copy_stream):
    with nvtx_range("h2d_copy"):
        batch_gpu = batch_cpu.to("cuda", non_blocking=True)
        batch_gpu.record_stream(copy_stream)

with nvtx_range("independent_compute"):
    run_compute_that_does_not_use_batch()

torch.cuda.current_stream().wait_stream(copy_stream)
with nvtx_range("consume_copied_batch"):
    use(batch_gpu)
```

Run/profile:

```bash
make exercise-02
```

Prove:

- Nsight Systems shows an HtoD memcpy overlapping compute kernels.
- Removing pinned memory or consuming the copied tensor too early reduces overlap or introduces synchronization.

## Exercise 3: Async NCCL All-Reduce

File: `scripts/03_manual_async_allreduce.py`

Goal: learn the difference between CPU-side async launch and real GPU-side overlap for NCCL collectives.

Implement:

- Two-rank `torch.distributed` setup with backend `nccl`.
- Use `LOCAL_RANK` to select the CUDA device, initialize the process group, and allocate one large tensor that represents a gradient bucket.
- A synchronous baseline: run independent compute, call blocking `dist.all_reduce(bucket)`, then run another compute block.
- An async version: launch `work = dist.all_reduce(bucket, async_op=True)`, run independent compute that does not read or write `bucket`, then call `work.wait()` before using `bucket`.
- Print rank-local timings from both ranks after profiling. Make sure every rank follows the same collective order.
- NVTX ranges named `all_reduce_sync`, `all_reduce_async_launch`, `independent_compute`, and `all_reduce_wait`.

Run/profile:

```bash
make exercise-03
```

Prove:

- NCCL kernels overlap independent compute kernels in each rank's trace.
- The async version is faster than the synchronous baseline when communication and compute durations are balanced.
- `async_op=True` alone is not counted as proof unless the GPU timeline shows overlap.

## Exercise 4: Tiny DDP-Style Gradient Buckets

File: `scripts/04_tiny_ddp_bucket_overlap.py`

Goal: reproduce the core DDP idea: launch communication for ready gradient buckets while backward compute continues for other layers.

Implement:

- A small multilayer MLP.
- A naive manual data-parallel baseline that runs `loss.backward()` fully, then reduces gradients.
- A manual bucketed version using autograd hooks: group parameters into buckets; when all gradients in a bucket are ready, flatten that bucket and launch `dist.all_reduce(..., async_op=True)`.
- A wait for all outstanding reductions before `optimizer.step()`.
- A comparison against real `torch.nn.parallel.DistributedDataParallel` with a small `bucket_cap_mb`.
- NVTX ranges for `forward`, `backward`, `bucket_<n>_all_reduce_launch`, `bucket_<n>_wait`, and `optimizer_step`.

Start with two buckets, for example early layers in one bucket and later layers in another. The goal is not to build a production reducer; the goal is to see communication for one bucket begin before the whole backward pass has finished.

Run/profile:

```bash
make exercise-04
```

Prove:

- In the naive version, NCCL all-reduces appear mostly after backward compute.
- In the bucketed version, earlier NCCL kernels appear during later backward compute.
- The real DDP trace resembles the manual bucketed version, with PyTorch's optimized reducer behavior.
