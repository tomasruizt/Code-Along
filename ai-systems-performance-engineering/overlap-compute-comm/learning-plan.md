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

Goal: learn how independent CUDA streams let work overlap on the GPU, and how an explicit stream dependency forces serialization.

Use `torch.matmul(a, b, out=out)` (cuBLAS), not a hand-written kernel, to build two independent GEMM workloads (`out0 = a0 @ b0`, `out1 = a1 @ b1`). Call each in a short loop (e.g. 10 iterations) so the region is long enough to read in Nsight Systems, and pass `out=` with a preallocated output tensor so you don't trigger a `cudaMalloc` on every iteration. Implement three variants:

- **Sequential baseline**: workload A then workload B, both on the default stream.
- **Two-stream overlap**: A on `stream_a` and B on `stream_b` via `with torch.cuda.stream(...)`, then synchronize before reading results.
- **Dependency**: the two-stream code, but call `stream_b.wait_stream(stream_a)` before launching B. The timeline should look sequential again.

Label each region with NVTX ranges (e.g. `@nvtx.annotate` on the variant functions and on the per-workload GEMM loops) so identical cuBLAS kernels are distinguishable in Nsight Systems.

Run/profile:

```bash
make exercise-01
```

Prove (the GPU timeline is the source of truth):

- **Sequential**: A's kernels finish before B's start: single stream, no overlap.
- **Two-stream**: A and B kernels appear on different streams and overlap horizontally.
- **Dependency**: `wait_stream` removes the overlap: back to sequential.

Note: when the streams overlap, the matmul kernels each run slower than in the sequential baseline. What might be the reason?

## Exercise 2: Copy/Compute Overlap

File: `scripts/02_copy_compute_overlap.py`

Goal: learn how a host-to-device copy can run concurrently with independent GPU compute using a separate stream, pinned memory, and `non_blocking=True`.

Use two pieces of work that don't touch each other's data: (1) a **copy** of a large CPU tensor to the GPU, and (2) **compute** a matmul. Size them so the copy and the compute take roughly comparable time. That's when overlap is interesting. Implement two variants:

- **Sequential baseline**: copy the CPU tensor to the GPU, then run the compute (the order doesn't matter), both on the default stream, so stream ordering already serializes them (no explicit `synchronize()` needed). Finally, sum the outputs of both ops.
- **Overlap**: launch the pinned HtoD copy with `non_blocking=True` on a new stream and run the compute on the default stream. The two are independent, so they overlap freely. The only dependency is the join (sum) where you first consume both tensors. Synchronize both streams before consuming with `default_stream.wait_stream(copy_stream)`, as otherwise you might run into a data race (sum consumes tensor before the copy finishes).

**Note**:

- The CPU tensor should be allocated with `pin_memory=True` (see last task of this exercise).
- reuse the `matmul(..., out=)` pattern from Exercise 1.
- Label the regions with NVTX ranges (`h2d_copy`, `compute`) so the copy and the matmul are easy to find in Nsight Systems.

Run/profile:

```bash
make exercise-02
```

Prove (the GPU timeline is the source of truth):

- **Sequential**: the HtoD memcpy finishes before the compute kernels start.
- **Overlap**: the memcpy and the compute kernels sit on different streams and overlap horizontally, and the overlapped region is faster: closer to `max(copy_time, compute_time)` than `copy_time + compute_time`.
- **Pinned memory matters**: drop `pin_memory=True` and `non_blocking=True` silently becomes a no-op. The copy goes synchronous and the overlap disappears.
- **The join wait is a contract, not an option**: without `wait_stream` the result can still *look* correct, but only as long as `copy_time < compute_time`, because the join (running on the default stream) is gated by the compute. Grow the copy past the compute and the consumed tensor comes back wrong, silently, with no error. The `wait_stream` is what makes it correct for every sizing.

Note: unlike Exercise 1, overlapping here makes the whole region faster. What's different about a host-to-device copy paired with a GEMM, versus two GEMMs?

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
- A manual bucketed version using autograd hooks: group parameters into buckets. When all gradients in a bucket are ready, flatten that bucket and launch `dist.all_reduce(..., async_op=True)`.
- A wait for all outstanding reductions before `optimizer.step()`.
- A comparison against real `torch.nn.parallel.DistributedDataParallel` with a small `bucket_cap_mb`.
- NVTX ranges for `forward`, `backward`, `bucket_<n>_all_reduce_launch`, `bucket_<n>_wait`, and `optimizer_step`.

Start with two buckets, for example early layers in one bucket and later layers in another. The goal is not to build a production reducer. The goal is to see communication for one bucket begin before the whole backward pass has finished.

Run/profile:

```bash
make exercise-04
```

Prove:

- In the naive version, NCCL all-reduces appear mostly after backward compute.
- In the bucketed version, earlier NCCL kernels appear during later backward compute.
- The real DDP trace resembles the manual bucketed version, with PyTorch's optimized reducer behavior.
