"""Rank-aware Nsight Systems wrapper for torchrun workers.

torchrun launches this script once per local rank. Each rank then launches its
target exercise under its own nsys profiler and writes a separate report.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


NSYS_TRACE = "cuda,nvtx,osrt,cudnn,cublas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("child", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.child or args.child[0] != "--" or len(args.child) == 1:
        parser.error("expected child command after '--'")
    args.child = args.child[1:]
    return args


def main() -> int:
    args = parse_args()
    rank = os.environ["LOCAL_RANK"]
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.name}-rank{rank}"

    cmd = [
        "nsys",
        "profile",
        "-o",
        str(report_path),
        "--force-overwrite=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        f"--trace={NSYS_TRACE}",
        "--cuda-memory-usage=true",
        sys.executable,
        *args.child,
    ]
    print(f"[rank {rank}] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)

    # With --capture-range-end=stop-shutdown, nsys can terminate itself via
    # SIGTERM after cudaProfilerStop(). subprocess reports that as 143 or -15
    # even when the .nsys-rep report was generated successfully.
    if result.returncode in (0, 143, -15):
        return 0
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
