"""Modal runner for the compute/communication overlap exercises.

Examples:
    modal run modal/modal_overlap.py --script 01_stream_basics.py --name stream-basics --n-procs 1
    modal run modal/modal_overlap.py --script 03_manual_async_allreduce.py --name async-allreduce --n-procs 2
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal


DEFAULT_GPU = "h100"
GPU = os.environ.get("GPU", DEFAULT_GPU)
APP_NAME = "overlap-compute-comm"
VOLUME_NAME = "overlap-compute-comm"
VOLUME_PATH = "/vol-overlap"
REMOTE_ROOT = "/opt/overlap"
LOCAL_ROOT = Path(__file__).resolve().parents[1]
NSYS_TRACE = "cuda,nvtx,osrt,cudnn,cublas"


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel")
    .apt_install("cuda-nsight-systems-13-0")
    .add_local_file(
        str(LOCAL_ROOT / "requirements.txt"),
        remote_path=f"{REMOTE_ROOT}/requirements.txt",
        copy=True,
    )
    .run_commands(
        f"python -m pip install --break-system-packages -r {REMOTE_ROOT}/requirements.txt"
    )
    .add_local_file(
        str(LOCAL_ROOT / "modal" / "nsys_wrapper.py"),
        remote_path=f"{REMOTE_ROOT}/modal/nsys_wrapper.py",
        copy=True,
    )
    .add_local_dir(
        str(LOCAL_ROOT / "scripts"),
        remote_path=f"{REMOTE_ROOT}/scripts",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )
)


def _print_system_info() -> None:
    print("=== system info ===", flush=True)
    subprocess.run(["nvidia-smi"], check=False)
    subprocess.run(["nvidia-smi", "topo", "-m"], check=False)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "print('torch', torch.__version__); "
                "print('cuda', torch.version.cuda); "
                "print('nccl', torch.cuda.nccl.version()); "
                "print('device_count', torch.cuda.device_count()); "
                "[print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
            ),
        ],
        check=False,
    )


def _run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env)
    # With --capture-range-end=stop-shutdown, nsys can terminate itself via
    # SIGTERM after cudaProfilerStop(). subprocess reports that as 143 or -15
    # even when the .nsys-rep report was generated successfully.
    if result.returncode not in (0, 143, -15):
        raise subprocess.CalledProcessError(result.returncode, cmd)


@app.function(gpu=GPU, image=image, volumes={VOLUME_PATH: volume}, timeout=30 * 60)
def profile_single_gpu(script: str, name: str) -> None:
    _print_system_info()
    report_dir = f"{VOLUME_PATH}/nsys"
    os.makedirs(report_dir, exist_ok=True)
    script_path = f"{REMOTE_ROOT}/scripts/{script}"
    report_path = f"{report_dir}/{name}"

    _run(
        [
            "nsys",
            "profile",
            "-o",
            report_path,
            "--force-overwrite=true",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop-shutdown",
            f"--trace={NSYS_TRACE}",
            "--cuda-memory-usage=true",
            sys.executable,
            script_path,
        ]
    )
    volume.commit()
    print(f"Report saved to Modal volume: nsys/{name}.nsys-rep", flush=True)


@app.function(
    gpu=f"{GPU}:2", image=image, volumes={VOLUME_PATH: volume}, timeout=30 * 60
)
def profile_two_gpu(script: str, name: str) -> None:
    _print_system_info()
    report_dir = f"{VOLUME_PATH}/nsys"
    os.makedirs(report_dir, exist_ok=True)
    script_path = f"{REMOTE_ROOT}/scripts/{script}"

    _run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            f"{REMOTE_ROOT}/modal/nsys_wrapper.py",
            "--report-dir",
            report_dir,
            "--name",
            name,
            "--",
            script_path,
        ]
    )
    volume.commit()
    print(
        f"Reports saved to Modal volume: nsys/{name}-rank0.nsys-rep, nsys/{name}-rank1.nsys-rep",
        flush=True,
    )


@app.local_entrypoint()
def main(script: str, name: str | None = None, n_procs: int = 1) -> None:
    if n_procs not in (1, 2):
        raise ValueError("n_procs must be 1 or 2 for this course runner")
    run_name = name or Path(script).stem
    if n_procs == 1:
        profile_single_gpu.remote(script=script, name=run_name)
    else:
        profile_two_gpu.remote(script=script, name=run_name)
