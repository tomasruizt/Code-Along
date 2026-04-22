"""
Triton puzzles can be found here: https://github.com/gpu-mode/Triton-Puzzles
Open the Google Collab notebook to see the exercises.
"""

import os

os.environ["TRITON_INTERPRET"] = "1"

import inspect
import triton_viz
import triton
import triton.language as tl
import torch
from jaxtyping import Float32
from torch import Tensor


def test(puzzle, puzzle_spec, nelem={}, B={"B0": 32}, viz=True):
    B = dict(B)
    if "N1" in nelem and "B1" not in B:
        B["B1"] = 32
    if "N2" in nelem and "B2" not in B:
        B["B2"] = 32

    triton_viz.clear()
    torch.manual_seed(0)
    signature = inspect.signature(puzzle_spec)
    args = {}
    for n, p in signature.parameters.items():
        print(p)
        args[n + "_ptr"] = ([d.size for d in p.annotation.dims], p)
    args["z_ptr"] = ([d.size for d in signature.return_annotation.dims], None)

    tt_args = []
    for k, (v, t) in args.items():
        tt_args.append(torch.rand(*v) - 0.5)
        if t is not None and t.annotation.dtypes[0] == "int32":
            tt_args[-1] = torch.randint(-100000, 100000, v)
    grid = lambda meta: (
        triton.cdiv(nelem["N0"], meta["B0"]),
        triton.cdiv(nelem.get("N1", 1), meta.get("B1", 1)),
        triton.cdiv(nelem.get("N2", 1), meta.get("B2", 1)),
    )

    # for k, v in args.items():
    #    print(k, v)
    triton_viz.trace("tracer")(puzzle)[grid](*tt_args, **B, **nelem)
    z = tt_args[-1]
    tt_args = tt_args[:-1]
    z_ = puzzle_spec(*tt_args)
    match = torch.allclose(z, z_, rtol=1e-3, atol=1e-3)
    print("Results match:", match)
    failures = None
    if not match or failures:
        if failures:
            print("Invalid Access:", failures)
        print("Yours:", z)
        print("Spec:", z_)
        print(torch.isclose(z, z_))
        return
    print("Correct!")
    print()


def add_vec_block_spec(
    x: Float32[Tensor, "100"], y: Float32[Tensor, "90"]
) -> Float32[Tensor, "90 100"]:
    return x[None, :] + y[:, None]


@triton.jit
def add_vec_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    x_offs = pid_0 * B0 + tl.arange(0, B0)
    x = tl.load(x_ptr + x_offs, mask=x_offs < N0)

    y_offs = pid_1 * B1 + tl.arange(0, B1)
    y = tl.load(y_ptr + y_offs, mask=y_offs < N1)

    z_offs = x_offs[None, :] + N0 * y_offs[:, None]
    z_mask = (x_offs < N0)[None, :] & (y_offs < N1)[:, None]
    tl.store(z_ptr + z_offs, x[None, :] + y[:, None], mask=z_mask)


test(add_vec_block_kernel, add_vec_block_spec, nelem={"N0": 100, "N1": 90}, viz=False)


def mul_relu_block_spec(
    x: Float32[Tensor, "100"], y: Float32[Tensor, "90"]
) -> Float32[Tensor, "90 100"]:
    return torch.relu(x[None, :] * y[:, None])


@triton.jit
def mul_relu_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    x_offs = pid_0 * B0 + tl.arange(0, B0)
    x = tl.load(x_ptr + x_offs, mask=x_offs < N0)

    y_offs = pid_1 * B1 + tl.arange(0, B1)
    y = tl.load(y_ptr + y_offs, mask=y_offs < N1)

    z = x[None, :] * y[:, None]  # outer product
    z = tl.where(z < 0, 0, z)  # relu
    z_offs = x_offs[None, :] + N0 * y_offs[:, None]
    z_mask = (x_offs[None, :] < N0) & (y_offs[:, None] < N1)
    tl.store(z_ptr + z_offs, z, mask=z_mask)


test(mul_relu_block_kernel, mul_relu_block_spec, nelem={"N0": 100, "N1": 90})


"""Description
Puzzle 6: Fused Outer Multiplication - Backwards
Backwards of a function that multiplies a matrix with a row vector and take a relu.

Uses two program blocks. Block size B0 is always less than the vector x length N0. Block size B1 is always less than vector y length N1. Chain rule backward dz is of shape N1 by N0

𝑓(𝑥,𝑦)=relu(𝑥𝑖×𝑦𝑗) for 𝑖=1…𝑁0, 𝑗=1…𝑁1 

𝑑𝑥𝑖,𝑗=𝑓′𝑥(𝑥,𝑦)𝑖,𝑗×𝑑𝑧𝑖,𝑗
"""


def mul_relu_block_back_spec(
    x: Float32[Tensor, "90 100"],
    y: Float32[Tensor, "90"],
    dz: Float32[Tensor, "90 100"],
) -> Float32[Tensor, "90 100"]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    z.backward(dz)
    dx = x.grad
    return dx


@triton.jit
def mul_relu_block_back_kernel(
    x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    x_horizontal = pid_0 * B0 + tl.arange(0, B0)
    x_vertical = pid_1 * B1 + tl.arange(0, B1)
    x_offs = x_horizontal[None, :] + N0 * x_vertical[:, None]
    x_mask = (x_horizontal[None, :] < N0) & (x_vertical[:, None] < N1)
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    y_offs = pid_1 * B1 + tl.arange(0, B1)
    y = tl.load(y_ptr + y_offs, mask=y_offs < N1)

    z = x * y[:, None]  # outer product

    # x, dx and dz have the same shape. Therefore, we reuse the
    # same offsets and mask.
    dz = tl.load(dz_ptr + x_offs, mask=x_mask)
    fprime = tl.where(z <= 0, 0, y[:, None])  # derivative of relu(x y)
    # fprime = tl.where(z <= 0, 0, y)  # derivative of relu(x y)
    dx = fprime * dz

    tl.store(dx_ptr + x_offs, dx, mask=x_mask)


test(mul_relu_block_back_kernel, mul_relu_block_back_spec, nelem={"N0": 100, "N1": 90})


def sum_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4"]:
    return x.sum(1)


@triton.jit
def sum_kernel(
    x_ptr, z_ptr, N0, N1, T: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * B0 + tl.arange(0, B0)
    cols = tl.arange(0, B1)

    zs = tl.zeros((B0,), dtype=tl.float32)
    for _ in range(0, T, B1):
        x_offs = cols[None, :] + T * rows[:, None]
        x_mask = (cols[None, :] < T) & (rows[:, None] < N0)
        x_blk = tl.load(x_ptr + x_offs, mask=x_mask)  # [B0,B1]
        zs += x_blk.sum(1)
        cols += B1

    # zs offset = rows
    tl.store(z_ptr + rows, zs, mask=rows < N0)


test(sum_kernel, sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})


def softmax_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4 200"]:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)


@triton.jit
def softmax_kernel(
    x_ptr, z_ptr, N0, N1, T: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr
):
    pid_0 = tl.program_id(0)
    log2_e = 1.44269504
    rows = pid_0 * B0 + tl.arange(0, B0)

    # STEP 1: Compute row-wise max
    # STEP 2: Compute row-wise sum exp (numerically stable)
    m = tl.full((B0,), value=float("-inf"), dtype=tl.float32)
    sumexp = tl.zeros((B0,), dtype=tl.float32)

    cols = tl.arange(0, B1)
    for _ in range(0, T, B1):
        x_offs = cols[None, :] + T * rows[:, None]
        x_mask = (rows[:, None] < N0) & (cols[None, :] < T)
        x = tl.load(x_ptr + x_offs, mask=x_mask, other=float("-inf"))  # [B0,B1]
        new_m = tl.maximum(x.max(1), m)
        rescale = tl.exp2(log2_e * (m - new_m))
        sumexp = sumexp * rescale + tl.exp2(log2_e * (x - new_m[:, None])).sum(1)
        m = new_m
        cols += B1

    # STEP 3: Compute softmax
    cols = tl.arange(0, B1)
    for _ in range(0, T, B1):
        x_offs = cols[None, :] + T * rows[:, None]
        x_mask = (rows[:, None] < N0) & (cols[None, :] < T)
        x = tl.load(x_ptr + x_offs, mask=x_mask, other=float("-inf"))  # [B0,B1]
        zs = tl.exp2(log2_e * (x - m[:, None])) / sumexp[:, None]
        # z and x have same shape and input/output mapping
        # therefore, we reuse offsets and masks
        tl.store(z_ptr + x_offs, zs, mask=x_mask)
        cols += B1


test(
    softmax_kernel,
    softmax_spec,
    B={"B0": 1, "B1": 32},
    nelem={"N0": 4, "N1": 32, "T": 200},
)
