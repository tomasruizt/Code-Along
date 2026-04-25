import torch
import triton
import triton.language as tl

DEVICE = "cuda"


def rms_norm(x):
    """
    Root mean square normalization
    x: [B, D]
    per entry in batch
    out: x / RMS(x)
    RMS = sqrt{ (x ** 2).mean() + e }
    e = small float
    """
    e = 1e-6
    x = x.float()
    xmean = (x**2).mean(-1, keepdims=True)  # [B, 1]
    out = x * torch.rsqrt(xmean + e)
    return out.to(torch.bfloat16)


def rl_rms_norm(x):
    B, D = x.shape
    BLOCK_B = 16
    BLOCK_D = 32
    out = torch.empty_like(x, device=DEVICE)

    grid = ((B + BLOCK_B - 1) // BLOCK_B,)
    rms_norm_kernel[grid](x, out, B, D, BLOCK_B, BLOCK_D)
    return out


@triton.jit
def rms_norm_kernel(x_ptr, out_ptr, B, D, BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    # rows stay fixed per program
    rows = pid * BLOCK_B + tl.arange(0, BLOCK_B)

    # STAGE I: Compute RMS
    squared_sums = tl.zeros((BLOCK_B,), dtype=tl.float32)
    # cols will increase in the loop
    cols = tl.arange(0, BLOCK_D)
    for _ in range(0, D, BLOCK_D):
        # compute RMS
        # load x, square, sum
        mask = (cols < D)[None, :] & (rows < B)[:, None]
        x_offs = cols[None, :] + D * rows[:, None]
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)  # [BLOCK_B, BLOCK_D]
        x = x.to(tl.float32)
        squared_sums += (x * x).sum(1)  # BLOCK_B
        cols += BLOCK_D

    # STAGE II: Normalize
    eps = 1e-6
    rrms = tl.rsqrt(squared_sums / D + eps)  # [BLOCK_B]
    cols = tl.arange(0, BLOCK_D)
    for _ in range(0, D, BLOCK_D):
        # load x, normalize, write x
        mask = (cols < D)[None, :] & (rows < B)[:, None]
        x_offs = cols[None, :] + D * rows[:, None]
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)  # [BLOCK_B, BLOCK_D]
        out = x * rrms[:, None]
        # out tiles are same as x tiles, therefore reuse offests and masks
        tl.store(out_ptr + x_offs, out, mask=mask)
        cols += BLOCK_D


if __name__ == "__main__":
    B = 33
    D = 257
    torch.manual_seed(0)
    x = torch.randn((B, D), dtype=torch.bfloat16, device=DEVICE)
    expected = rms_norm(x)
    actual = rl_rms_norm(x)
    try:
        torch.testing.assert_close(actual, expected)
        print("✅ Correct!")
    except AssertionError:
        print(actual)
        print(expected)
        print(actual == expected)
        raise
