import torch
import triton
import triton.language as tl

DEVICE = "cuda"


def rms_norm(x, g):
    """
    Root mean square normalization
    x: [S, D]
    g: D
    per entry in seqlen
    out: xi gi / RMS
    RMS = sqrt{ (x ** 2).mean() + e }
    e = small float
    """
    e = 1e-6
    x = x.float()
    xmean = (x**2).mean(-1, keepdims=True)  # [B, 1]
    out = g[None, :] * x * torch.rsqrt(xmean + e)
    return out.to(torch.bfloat16)


def rl_rms_norm(x, g):
    S, D = x.shape
    BLOCK_S = 16
    BLOCK_D = 32
    out = torch.empty_like(x, device=DEVICE)

    grid = ((S + BLOCK_S - 1) // BLOCK_S,)
    rms_norm_kernel[grid](x, g, out, S, D, BLOCK_S, BLOCK_D)
    return out


@triton.jit
def rms_norm_kernel(
    x_ptr, g_ptr, out_ptr, S, D, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    # rows stay fixed per program
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)

    # STAGE I: Compute RMS
    squared_sums = tl.zeros((BLOCK_S,), dtype=tl.float32)
    # cols will increase in the loop
    cols = tl.arange(0, BLOCK_D)
    for _ in range(0, D, BLOCK_D):
        # compute RMS
        # load x, square, sum
        mask = (cols < D)[None, :] & (rows < S)[:, None]
        x_offs = cols[None, :] + D * rows[:, None]
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)  # [BLOCK_S, BLOCK_D]
        x = x.to(tl.float32)
        squared_sums += (x * x).sum(1)  # BLOCK_S
        cols += BLOCK_D

    # STAGE II: Normalize
    eps = 1e-6
    rrms = tl.rsqrt(squared_sums / D + eps)  # [BLOCK_S]
    cols = tl.arange(0, BLOCK_D)
    for _ in range(0, D, BLOCK_D):
        # load x, normalize, write x
        mask = (cols < D)[None, :] & (rows < S)[:, None]
        x_offs = cols[None, :] + D * rows[:, None]
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)  # [BLOCK_S, BLOCK_D]
        # load g
        g = tl.load(g_ptr + cols, mask=cols < D, other=0.0)  # [BLOCK_D]
        out = x * rrms[:, None] * g[None, :]
        # out tiles are same as x tiles, therefore reuse offests and masks
        tl.store(out_ptr + x_offs, out, mask=mask)
        cols += BLOCK_D


if __name__ == "__main__":
    S = 33
    D = 257
    torch.manual_seed(0)
    x = torch.randn((S, D), dtype=torch.bfloat16, device=DEVICE)
    g = torch.randn(D, dtype=torch.bfloat16, device=DEVICE)
    expected = rms_norm(x, g)
    actual = rl_rms_norm(x, g)
    try:
        torch.testing.assert_close(actual, expected)
        print("✅ Correct!")
    except AssertionError:
        print(actual)
        print(expected)
        print(actual == expected)
        raise
