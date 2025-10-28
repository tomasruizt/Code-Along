from dataclasses import dataclass
from time import time
import timeit
import pandas as pd
from typing import Any, Callable
import torch
from fused_mm_sampling import fused_sample_triton, incremental_sample_pt, sample

torch.set_default_device("cuda")

vocab_size = 256000
hidden_size = 5120
seq_len = 256
num_samples = 1
speedtest_kwargs = dict(
    hidden_states=torch.randn((hidden_size, seq_len)).bfloat16(),
    weights=torch.randn((vocab_size, hidden_size)).bfloat16(),
    num_samples=num_samples,
    temperature=1.0,
)

sample_compiled = torch.compile(sample)


@dataclass
class Case:
    name: str
    fn: Callable[[], Any]
    n_runs: int


cases = [
    Case(
        name="fused-triton",
        fn=lambda: fused_sample_triton(
            **speedtest_kwargs, seed=0, block_size_v=8, block_size_d=16
        ),
        n_runs=10,
    ),
    Case(
        name="naive-pt",
        fn=lambda: sample(**speedtest_kwargs),
        n_runs=10,
    ),
    Case(
        name="naive-compiled",
        fn=lambda: sample_compiled(**speedtest_kwargs),
        n_runs=10,
    ),
]


def benchmark(case: Case) -> pd.DataFrame:
    print(f"Benchmarking fn='{case.name}'")

    print("Warming up...")
    for _ in range(10):
        case.fn().cpu()

    print("Timing...")
    start = time()
    times = timeit.repeat(case.fn, number=case.n_runs)
    end = time()
    total_time = end - start
    # According to time.repeat() the min() is the most informative statistic
    results = {
        "name": case.name,
        "total[s]": total_time,
        "time[s]": times,
        "time[ms]": [t * 1_000 for t in times],
        "time[µs]": [t * 1_000_000 for t in times],
    }
    df = pd.DataFrame(results)
    return df


def benchmark_all() -> pd.DataFrame:
    import gc

    gc.disable()
    try:
        dfs = [benchmark(case) for case in cases]
        return pd.concat(dfs, ignore_index=True)
    finally:
        gc.enable()


if __name__ == "__main__":
    df = benchmark_all()
    print(f"{vocab_size=}")
    print(f"{hidden_size=}")
    print(f"{seq_len=}")
    print(f"{num_samples=}")

    total_runtimes = df.groupby(["name", "total[s]"], as_index=False).size()
    print(total_runtimes.round(2))

    time_distribution = df.groupby("name")[["time[ms]"]].describe()
    print(time_distribution.round(2))
