# Course Structure: Overlapping Computation and Communication

This directory is a self-contained course module for learning compute/communication overlap in PyTorch with Nsight Systems evidence. The learner-facing plan is `learning-plan.md`. This file documents how the course code is organized and what conventions future changes should follow.

## Directory Layout

```text
overlap-compute-comm/
  AGENTS.md
  Makefile
  learning-plan.md
  requirements.txt
  modal/
    modal_overlap.py
    nsys_wrapper.py
  scripts/
    01_stream_basics.py
    02_copy_compute_overlap.py
    03_manual_async_allreduce.py
    04_tiny_ddp_bucket_overlap.py
  traces/
  notes/
```

## File Roles

- `learning-plan.md`: short learner-facing syllabus. Keep it focused on the learning objective, each exercise, and the commands to run/profile on Modal.
- `AGENTS.md`: maintainer-facing structure guide. Put course organization, coding conventions, and runner expectations here instead of expanding `learning-plan.md`.
- `Makefile`: convenience targets that run each exercise on Modal, profile it, and download the expected Nsight Systems report(s) into `traces/`.
- `requirements.txt`: Python dependencies for both local Modal submission and the remote Modal image. `modal/modal_overlap.py` installs this file into the image before copying exercise scripts. The PyTorch base image uses an externally managed Python environment, so remote image installs must pass `pip --break-system-packages`.
- `modal/modal_overlap.py`: Modal entrypoint. It builds the remote image, copies `scripts/` and `modal/nsys_wrapper.py`, runs exercises under Nsight Systems, writes reports to the Modal volume, and supports both one-GPU and two-GPU runs.
- `modal/nsys_wrapper.py`: rank-aware wrapper used by `torchrun` for distributed exercises. Each local rank runs its target script under a separate `nsys profile` invocation and writes one report per rank.
- `scripts/`: exercise implementations. These should be standalone Python scripts runnable by the Modal runner.
- `traces/`: local destination for downloaded `.nsys-rep` files. Treat profiler reports as generated artifacts.
- `notes/`: learner notes, screenshots, measurements, and summaries. Treat these as optional study artifacts, not required source code.

## Modal Runner Contract

Run all exercises from this directory with:

```bash
modal run modal/modal_overlap.py --script <script.py> --name <run-name> --n-procs <1-or-2>
```

`--n-procs 1` uses one Modal GPU and writes:

```text
nsys/<run-name>.nsys-rep
```

`--n-procs 2` uses two Modal GPUs in one container, launches `torch.distributed.run`, and writes:

```text
nsys/<run-name>-rank0.nsys-rep
nsys/<run-name>-rank1.nsys-rep
```

Reports are stored in the Modal volume named `overlap-compute-comm`. Download them with:

```bash
modal volume get --force overlap-compute-comm nsys/<report-name>.nsys-rep traces/
```

Use `--force` when downloading reports in automation so rerunning a target overwrites the previous local report instead of failing on an existing file. The Makefile already does this.

The runner defaults to `h100`. Advanced users can override it with `GPU=<modal-gpu-type>`, for example `GPU=a100-80gb modal run ...`.

Nsight Systems can exit with code `143` after `cudaProfilerStop()` when using `--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown`, even though it generated the report successfully. The Modal runner treats `0`, `143`, and `-15` as successful profiler shutdown codes.

## Codex Sandbox Warning

Modal commands require network access. In Codex, the default tool sandbox may restrict network access even when `modal` is installed and authenticated. If an agent runs `make exercise-*` or `modal run ...` inside the restricted sandbox, the command can appear to hang after printing the initial `modal run` line and produce no useful error.

When using Codex tools to run any Modal-backed target, request escalated execution for the command. Do not interpret a silent sandboxed `modal run` as an exercise-code failure, and do not edit files under `scripts/` to debug that symptom. First rerun the exact Make target with network access and inspect the real Modal/Nsight error output.

The Makefile wraps the same contract:

```bash
make exercise-01
make exercise-02
make exercise-03
make exercise-04
make profile-all
```

Each `exercise-*` target runs the corresponding Modal job and downloads the report(s). `profile-all` runs all four in order. To override the GPU for Makefile runs, pass `GPU=<modal-gpu-type>`, for example:

```bash
make exercise-01 GPU=a100-80gb
```

## Exercise Script Conventions

Each script in `scripts/` should:

- Be executable as a standalone Python file.
- Print enough configuration to interpret results: relevant tensor sizes, iteration counts, measured timings, device name, and rank/world size when distributed.
- Use deterministic seeds where practical.
- Include a sequential or non-overlapped baseline and an overlapped variant.
- Add NVTX ranges around the important regions so Nsight Systems traces are readable.
- Mark the captured profiling region with `torch.cuda.cudart().cudaProfilerStart()` and `torch.cuda.cudart().cudaProfilerStop()`. The Modal runner uses `--capture-range=cudaProfilerApi`, so no CUDA activity is collected until the script starts the profiler.
- Keep imports, CUDA initialization, tensor allocation, warmup, result checks, and printing outside the captured profiling region.
- Use `torch.cuda.Event` timings only as supporting measurements; the Nsight Systems timeline is the source of truth for overlap.
- Synchronize explicitly around timing boundaries.
- Keep tensor sizes and iteration counts configurable with command-line arguments when useful.

Distributed scripts should additionally:

- Initialize `torch.distributed` with backend `nccl`.
- Use `LOCAL_RANK`, `RANK`, and `WORLD_SIZE` from the `torchrun` environment.
- Call `torch.cuda.set_device(local_rank)` before allocating CUDA tensors.
- Destroy the process group before exit when practical.
- Avoid reading or mutating tensors involved in async collectives before the corresponding `work.wait()`.

## Exercise Scope

The course currently has four exercises:

- `01_stream_basics.py`: CUDA stream ordering, concurrency, and explicit stream dependencies.
- `02_copy_compute_overlap.py`: pinned-memory HtoD copy overlapped with independent compute.
- `03_manual_async_allreduce.py`: manual `dist.all_reduce(..., async_op=True)` overlapped with independent compute.
- `04_tiny_ddp_bucket_overlap.py`: DDP-style gradient bucket reductions launched from autograd hooks while backward continues.

Keep new exercises in the same naming style: two-digit order prefix, short descriptive snake-case name, and one script per exercise.

## Proof Standard

An exercise is complete only when the trace proves overlap. API-level async behavior is not enough.

Acceptable evidence:

- Compute kernels and copy/NCCL kernels appear on different CUDA streams.
- Their intervals overlap horizontally in Nsight Systems.
- The overlapped variant is faster than the sequential baseline when compute and communication/copy durations are balanced.

For distributed exercises, inspect both rank traces. Rank-specific traces are intentional; they make stream and NCCL behavior easier to audit than a single parent-launcher report.

## Generated Artifacts

Do not commit generated profiler outputs by default:

- `.nsys-rep`
- `.sqlite`
- `.qdrep`
- `.nsys-analysis`
- Python `__pycache__/`

If a small screenshot or table is useful for publication, place it under `notes/` with a clear name and keep the raw profiler report out of git unless there is a specific reason to include it.
