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

Goal: manually overlap an NCCL all-reduce with independent backward compute, and verify the overlap in Nsight Systems.

Build a two-rank toy data-parallel training step using plain CUDA tensors. Do not use `nn.Module`, `loss.backward()`, or DDP; the backward GEMMs and collective launches should be explicit in the script.

Use a bias-free two-layer MLP:

```python
z = x @ w1
h = torch.relu(z)
y = h @ w2
```

For squared error, compute the gradients manually:

```python
grad_y = y - target
grad_w2 = h.T @ grad_y
grad_h = grad_y @ w2.T
grad_z = grad_h * (z > 0)
grad_w1 = x.T @ grad_z
```

<details>
<summary>Where do these gradients come from?</summary>

Use the squared-error loss:

$$
L = \frac{1}{2}\lVert Y - T \rVert_F^2
$$

The output gradient is:

$$
\frac{\partial L}{\partial Y} = Y - T
$$

In code:

```python
grad_y = y - target
```

For the second layer, $Y = HW_2$. Its differential is:

$$
dY = dH\,W_2 + H\,dW_2
$$

Using the Frobenius inner product definition of a matrix gradient,

$$
dL = \left\langle dY, \frac{\partial L}{\partial Y} \right\rangle_F
$$

gives:

$$
\frac{\partial L}{\partial W_2} = H^\top \frac{\partial L}{\partial Y},
\qquad
\frac{\partial L}{\partial H} = \frac{\partial L}{\partial Y} W_2^\top
$$

In code:

```python
grad_w2 = h.T @ grad_y
grad_h = grad_y @ w2.T
```

Backpropagate through ReLU:

$$
\frac{\partial L}{\partial Z}
= \frac{\partial L}{\partial H} \odot \mathbf{1}_{Z > 0}
$$

In code:

```python
grad_z = grad_h * (z > 0)
```

For the first layer, $Z = XW_1$, so:

$$
\frac{\partial L}{\partial W_1} = X^\top \frac{\partial L}{\partial Z}
$$

In code:

```python
grad_w1 = x.T @ grad_z
```

</details>

The key dependency is that `grad_w2` is ready before `grad_w1`. Start reducing `grad_w2` as soon as it is computed, continue computing `grad_h`, `grad_z`, and `grad_w1`, then wait for the reduction before updating `w2`. This is the manual version of the overlap DDP tries to create automatically with gradient buckets.

Implementation requirements:

- Initialize `torch.distributed` with backend `nccl`.
- Read `LOCAL_RANK`, `RANK`, and `WORLD_SIZE` from the `torchrun` environment.
- Call `torch.cuda.set_device(local_rank)` before allocating CUDA tensors.
- Allocate tensors on `torch.device("cuda", local_rank)`.
- Initialize `w1` and `w2` identically on both ranks, but use rank-local input data.
- Use `dist.all_reduce(grad_w2, op=dist.ReduceOp.AVG, async_op=True)` immediately after `grad_w2` is computed.
- Compute `grad_w1` while the `grad_w2` all-reduce is outstanding.
- Reduce `grad_w1` too before the weight update. It can also use `async_op=True`, but it is not the overlap being studied.
- Wait for each gradient's reduction before updating the corresponding weight with manual gradient descent.
- Every rank must call collectives in the same order.

Use NVTX ranges around the training step and any phases you want to inspect.

Size the tensors so the `grad_w2` all-reduce and the remaining backward GEMMs take comparable time. If one side is tiny, there is little overlap to see.

Synchronization notes:

- `async_op=True` returns a `Work` handle. Do not use the reduced value from the host or update the corresponding weight until `work.wait()` has completed.
- `dist.all_reduce(..., async_op=False)` and `torch.distributed.barrier()` should not be treated as host-side GPU drains. If you need a clean profiling boundary, use `torch.cuda.synchronize()` before `cudaProfilerStart()` and before `cudaProfilerStop()`.
- Do not put `torch.cuda.synchronize()` inside the training step unless you are intentionally destroying the overlap.

Run/profile:

```bash
make exercise-03
```

Prove (the GPU timeline is the source of truth):

- NCCL kernels for the `grad_w2` all-reduce overlap horizontally with the GEMM kernels for the remaining backward compute.
- The overlap is visible on both ranks, not just in Python timings.
- Each weight update happens after the corresponding gradient's all-reduce wait.
- API-level `async_op=True` is not enough by itself; the Nsight Systems GPU timeline must show actual communication/compute overlap.

## Exercise 4: Tiny DDP-Style Gradient Buckets

File: `scripts/04_tiny_ddp_bucket_overlap.py`

Goal: generalize Exercise 3's hand-scheduled overlap into DDP-style scheduling: backward runs normally, gradients become ready at different times, and communication launches automatically when a bucket is ready.

Use a small multilayer MLP with enough layers that backward takes visible time in Nsight Systems. In Exercise 3, you manually placed one all-reduce between two backward compute blocks because you knew exactly when `grad_w2` was ready. Here, let autograd tell you when gradients are ready. Start with two gradient buckets, for example early layers in one bucket and later layers in another. The goal is not to build a production reducer. The goal is to see communication launch from gradient readiness instead of from a hand-written backward schedule.

Implement three variants:

- **Naive manual data parallel**: run `loss.backward()` fully, then flatten and all-reduce the gradients after all backward compute is done. This is the baseline: gradient communication starts late.
- **Manual bucketed overlap**: group parameters into buckets and register autograd hooks on their gradients. When all gradients in a bucket are ready, flatten that bucket and launch `dist.all_reduce(..., async_op=True)`. Keep the reduced flat buffer alive until the work completes, and copy the averaged values back into the original gradients before `optimizer.step()`.
- **Real DDP comparison**: wrap the same model with `torch.nn.parallel.DistributedDataParallel` and use a small `bucket_cap_mb` so PyTorch creates small buckets. This gives you a reference trace for the optimized reducer behavior.

Wait for all outstanding bucket reductions before `optimizer.step()`. Every rank must build the same buckets and launch collectives in the same order. Print rank-local timings and a small loss or gradient checksum after profiling so the three variants can be compared.

Label the phases with NVTX ranges: `forward`, `backward`, `bucket_<n>_all_reduce_launch`, `bucket_<n>_wait`, and `optimizer_step`.

Run/profile:

```bash
make exercise-04
```

Prove (the GPU timeline is the source of truth):

- **Naive**: NCCL all-reduces appear mostly after backward compute, because communication does not start until `loss.backward()` has returned.
- **Manual bucketed overlap**: ready bucket all-reduces begin while remaining backward kernels are still running, so NCCL work overlaps horizontally with backward compute.
- **Real DDP**: the DDP trace resembles the manual bucketed version, but with PyTorch's optimized reducer behavior and bucket scheduling.
- **The bucket boundary matters**: if one bucket contains almost all parameters, there is little or no overlap to see. Adjust the bucket split so one bucket becomes ready while useful backward work remains.
