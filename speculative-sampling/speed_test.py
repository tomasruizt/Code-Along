import timeit
import pandas as pd
from typing import Any
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

mapping = {
    "naive-pt": lambda: sample(**speedtest_kwargs),
    "naive-compiled": lambda: sample_compiled(**speedtest_kwargs),
    # "fused-pt": lambda: incremental_sample_pt(**speedtest_kwargs),
    "fused-triton": lambda: fused_sample_triton(
        **speedtest_kwargs, seed=0, block_size_v=2
    ),
}


def benchmark(fn_name: str) -> dict[str, Any]:
    print(f"Benchmarking fn='{fn_name}'")

    fn = mapping[fn_name]
    print("Warming up...")
    for _ in range(10):
        fn()

    print("Timing...")
    n_runs = 100
    total_time = timeit.timeit(fn, number=n_runs)
    # per run stats
    secs = total_time / n_runs
    msecs = secs * 1000
    nsecs = msecs * 1000
    results = {
        "name": fn_name,
        "n_runs": n_runs,
        "total_secs": total_time,
        "secs": secs,
        "msecs": msecs,
        "nsecs": nsecs,
    }
    return results


def benchmark_all() -> pd.DataFrame:
    rows = [benchmark(fn_name) for fn_name in mapping]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = benchmark_all()
    print(df.round(3))
