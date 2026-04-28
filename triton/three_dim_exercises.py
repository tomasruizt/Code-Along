import torch
import triton
import triton.language as tl


def ref():
    """
    This example creates a 3D data shaped
    x = [DOGS, CATS, DATA]
    The first kernel should copy the data for dog_i, cat_j
    """
    dvc = "cuda"
    DOGS = 123
    CATS = 321
    DATA = 16
    torch_kwargs = dict(dtype=torch.float32, device=dvc)
    x = torch.randn((DOGS, CATS, DATA), **torch_kwargs)
    dog_i = 45
    cat_j = 98
    out = torch.empty((DATA,), **torch_kwargs)
    grid = (1,)
    kernel_1[grid](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out,
        dog_i,
        cat_j,
        DATA,
    )
    assert x.stride(2) == 1
    assert out.stride(0) == 1
    torch.testing.assert_close(x[dog_i, cat_j, :], out)
    print("✅ kernel 1 pass")

    """
    The second exercise is to read data from N locations.
    """
    N = 5
    i_locs = torch.tensor([45, 23, 87, 102, 3], device=dvc, dtype=torch.int32)
    j_locs = torch.tensor([78, 200, 54, 8, 123], device=dvc, dtype=torch.int32)
    out = torch.empty((N, DATA), **torch_kwargs)
    kernel_2[(N,)](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out,
        out.stride(0),
        out.stride(1),
        i_locs,
        j_locs,
        DATA,
    )
    assert x.stride(2) == 1
    assert out.stride(1) == 1
    torch.testing.assert_close(x[i_locs, j_locs, :], out)
    print("✅ kernel 2 pass")

    """
    In the 3. exercise we want to get all full slide across the [DOG, CAT] plan
    given by an data_offset. The block size will be B_DOGS, B_CATS
    """
    out = torch.empty((DOGS, CATS), **torch_kwargs)
    B_DOGS = 16
    B_CATS = 32
    grid = (cdiv(DOGS, B_DOGS), cdiv(CATS, B_CATS))
    data_offset = 5
    kernel_3[grid](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out,
        out.stride(0),
        out.stride(1),
        data_offset,
        DOGS,
        CATS,
        B_DOGS,
        B_CATS,
    )
    torch.testing.assert_close(x[:, :, data_offset], out)
    print("✅ kernel 3 pass")

    """
    Exercise 4: For a specific cat j, pull all the dogs and data slice.
    The grid is split in B_CAT blocks.
    """
    out = torch.empty((DOGS, DATA), **torch_kwargs)
    grid = (cdiv(CATS, B_CATS),)
    kernel_4[grid](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out,
        out.stride(0),
        out.stride(1),
        cat_j,
        DATA,
        DOGS,
        B_DOGS,
    )
    torch.testing.assert_close(x[:, cat_j, :], out)
    print("✅ kernel 4 pass")

    """
    Exercise 5: For a specific dog i, pull all cats and data slice.
    """
    out = torch.empty((CATS, DATA), **torch_kwargs)
    grid = (cdiv(CATS, B_CATS),)
    kernel_5[grid](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out,
        out.stride(0),
        out.stride(1),
        dog_i,
        DATA,
        CATS,
        B_CATS,
    )
    torch.testing.assert_close(x[dog_i, :, :], out)
    print("✅ kernel 5 pass")


@triton.jit
def kernel_5(
    x_ptr,
    stride_x0,
    stride_x1,
    stride_x2,
    out_ptr,
    stride_out0,
    stride_out1,
    dog_i,
    DATA: tl.constexpr,
    CATS,
    B_CATS: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * B_CATS + tl.arange(0, B_CATS)
    rows = tl.arange(0, DATA)
    x_offs = dog_i * stride_x0 + cols[None, :] * stride_x1 + rows[:, None] * stride_x2
    x_mask = cols[None, :] < CATS
    data = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

    out_offs = cols[None, :] * stride_out0 + rows[:, None] * stride_out1
    tl.store(out_ptr + out_offs, data, mask=x_mask)


@triton.jit
def kernel_4(
    x_ptr,
    stride_x0,
    stride_x1,
    stride_x2,
    out_ptr,
    stride_out0,
    stride_out1,
    cat_j,
    DATA: tl.constexpr,
    DOGS,
    B_DOGS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * B_DOGS + tl.arange(0, B_DOGS)
    # Interestinly, even though in my mind the columns so far where CATS,
    # here I can use cols in the DATA dimension. As you can see in the striding
    # below, it doesn't even matter that the x tensor has DATA as the 3rd dimension
    # and for us load DATA as the 2nd of the tile (pattern cols[None, :]).
    cols = tl.arange(0, DATA)
    x_offs = rows[:, None] * stride_x0 + cat_j * stride_x1 + cols[None, :] * stride_x2
    x_mask = rows[:, None] < DOGS
    data = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

    out_offs = rows[:, None] * stride_out0 + cols[None, :] * stride_out1
    tl.store(out_ptr + out_offs, data, mask=x_mask)


@triton.jit
def kernel_3(
    x_ptr,
    stride_x0,
    stride_x1,
    stride_x2,
    out_ptr,
    stride_out0,
    stride_out1,
    data_offset,
    DOGS,
    CATS,
    B_DOGS: tl.constexpr,
    B_CATS: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    rows = pid_i * B_DOGS + tl.arange(0, B_DOGS)
    cols = pid_j * B_CATS + tl.arange(0, B_CATS)

    x_offs = (
        rows[:, None] * stride_x0 + cols[None, :] * stride_x1 + data_offset * stride_x2
    )
    x_mask = (rows[:, None] < DOGS) & (cols[None, :] < CATS)
    tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

    out_offs = rows[:, None] * stride_out0 + cols[None, :] * stride_out1
    out_mask = (rows[:, None] < DOGS) & (cols[None, :] < CATS)
    tl.store(out_ptr + out_offs, tile, mask=out_mask)


def cdiv(x, d):
    return (x + d - 1) // d


@triton.jit
def kernel_2(
    x_ptr,
    stride_x0,
    stride_x1,
    stride_x2,
    out_ptr,
    stride_out0,
    stride_out1,
    i_locs_ptr,
    j_locs_ptr,
    DATA: tl.constexpr,
):
    pid = tl.program_id(0)
    i = tl.load(i_locs_ptr + pid)
    j = tl.load(j_locs_ptr + pid)
    data = tl.load(
        x_ptr + stride_x0 * i + stride_x1 * j + stride_x2 * tl.arange(0, DATA)
    )
    tl.store(out_ptr + stride_out0 * pid + stride_out1 * tl.arange(0, DATA), data)


@triton.jit
def kernel_1(
    x_ptr, stride_x0, stride_x1, stride_x2, out_ptr, dog_i, cat_j, DATA: tl.constexpr
):
    data = tl.load(
        x_ptr + stride_x0 * dog_i + stride_x1 * cat_j + stride_x2 * tl.arange(0, DATA)
    )
    tl.store(out_ptr + tl.arange(0, DATA), data)


if __name__ == "__main__":
    ref()
