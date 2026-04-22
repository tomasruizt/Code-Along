from functools import partial
import math
import torch
import triton.language as tl
import triton

torch.set_default_device("cuda")


BLOCKSIZE = 64


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n: int):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCKSIZE
    nums = block_start + tl.arange(0, BLOCKSIZE)
    mask = nums < n
    x = tl.load(x_ptr + nums, mask=mask)
    y = tl.load(y_ptr + nums, mask=mask)
    z = x + y
    tl.store(out_ptr + nums, z, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    n_blocks = math.ceil(x.numel() / BLOCKSIZE)
    grid = (n_blocks,)
    add_kernel[grid](x, y, z, n=x.numel())
    return z


benchmark = partial(triton.testing.do_bench, quantiles=[0.5, 0.2, 0.8])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def full_benchmark(size: int, provider: str) -> tuple[float, float, float]:
    x = torch.rand(size)
    y = torch.rand(size)
    if provider == "torch":
        ms, min_ms, max_ms = benchmark(lambda: x + y)
    if provider == "triton":
        ms, min_ms, max_ms = benchmark(lambda: add(x, y))

    def gbps(ms):
        return 3 * x.numel() * x.element_size() * 1e-09 / (ms * 0.001)

    return gbps(ms), gbps(min_ms), gbps(max_ms)


# x = torch.arange(100)
# z = add(x, x)

full_benchmark.run(save_path="benchmarks/01/")
