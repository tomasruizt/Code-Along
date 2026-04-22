import os

os.environ["TRITON_INTERPRET"] = "1"

import triton.language as tl
import triton
import torch


def main():
    kwargs = dict(device="cuda", dtype=torch.float32)
    N = 129
    D = 64
    BLOCK_SIZE = 32
    Q = torch.randn((N, D), **kwargs)
    K = torch.randn((N, D), **kwargs)
    S = torch.full((N, N), -1.0, **kwargs)
    grid = (cdiv(N, BLOCK_SIZE), cdiv(N, BLOCK_SIZE))
    casual_self_attn[grid](
        Q,
        K,
        S,
        N,
        D,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        S.stride(0),
        S.stride(1),
        BLOCK_SIZE,
    )

    mask = torch.arange(N)[None, :] > torch.arange(N)[:, None]
    expected = Q @ K.T
    expected[mask] = float("-inf")
    torch.testing.assert_close(expected, S)
    print("✅ success! output is as expected")


@triton.jit
def casual_self_attn(
    Q_ptr,
    K_ptr,
    S_ptr,
    N,
    D: tl.constexpr,
    qn_stride,
    qd_stride,
    kn_stride,
    kd_stride,
    sn_stride,
    sm_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    # if pid_col != 0 or pid_row != 0:
    #     return

    s_xs = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    s_ys = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    s_offs = s_xs[None, :] + s_ys[:, None] * sn_stride
    s_mask = (s_xs[None, :] < N) * (s_ys[:, None] < N)

    if pid_col > pid_row:  # store all inf
        tl.store(S_ptr + s_offs, float("-inf"), mask=s_mask)
        return

    # load q, k
    q_ys = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    q_xs = tl.arange(0, D)
    q_offs = q_xs[None, :] + q_ys[:, None] * qn_stride
    q = tl.load(Q_ptr + q_offs, mask=q_ys[:, None] < N, other=0.0)

    k_ys = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    k_xs = tl.arange(0, D)
    k_offs = k_xs[None, :] + k_ys[:, None] * kn_stride
    k = tl.load(K_ptr + k_offs, mask=k_ys[:, None] < N, other=0.0)

    # dot product
    s = tl.dot(q, k.T)

    # mask (only when needed)
    if pid_row == pid_col:  # masking needed
        rows = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cols = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols[None, :] > rows[:, None]  # cols > rows
        s = tl.where(mask, float("-inf"), s)

    # save s
    tl.store(S_ptr + s_offs, s, mask=s_mask)


def cdiv(x: int, n: int) -> int:
    return (x + n - 1) // n


if __name__ == "__main__":
    main()
