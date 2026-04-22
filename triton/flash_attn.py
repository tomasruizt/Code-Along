# import os

# os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton.language as tl
import triton


def flash_attn(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    seq_len, embed_dim = query.shape

    output = torch.empty((seq_len, embed_dim), device="cuda")

    bm = embed_dim // 2
    grid = ((seq_len + bm - 1) // bm,)
    flash_attn_kernel[grid](
        query,
        key,
        value,
        output,
        bm=bm,
        bk=embed_dim,
        bn=embed_dim,
        m=seq_len,
        k=embed_dim,
        n=seq_len,
        scale=1.0 / embed_dim**0.5,
    )
    return output


@triton.jit
def flash_attn_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    out_ptr,
    m: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    bn: tl.constexpr,
    bk: tl.constexpr,
    bm: tl.constexpr,
    scale: tl.constexpr,
):
    q_descr = tl.make_tensor_descriptor(
        query_ptr, shape=[m, k], block_shape=(bm, bk), strides=(k, 1)
    )
    k_descr = tl.make_tensor_descriptor(
        key_ptr, shape=[n, k], block_shape=(bn, bk), strides=(k, 1)
    )
    v_descr = tl.make_tensor_descriptor(
        value_ptr, shape=[n, k], block_shape=(bn, bk), strides=(k, 1)
    )
    out_descr = tl.make_tensor_descriptor(
        out_ptr, shape=[m, k], block_shape=(bm, bk), strides=(k, 1)
    )

    pid = tl.program_id(axis=0)

    q = q_descr.load(offsets=[pid * bm, 0])  # 0 because we load the entire d-dim
    maxs = tl.full((bm,), float("-inf"), dtype=tl.float32)  # running max
    l = tl.zeros((bm,), dtype=tl.float32)  # running sum exp

    acc = tl.zeros((bm, bk), dtype=tl.float32)
    for i in range(0, n, bn):
        key = k_descr.load(offsets=[i, 0])
        v = v_descr.load(offsets=[i, 0])
        qk = tl.dot(q, key.T, input_precision="ieee") * scale  # [bm, bn]
        new_maxs = tl.maximum(maxs, qk.max(axis=1))
        rescale = (maxs - new_maxs).exp()
        maxs = new_maxs
        e = (qk - maxs[:, None]).exp()
        l *= rescale
        l += e.sum(axis=1)
        acc *= rescale[:, None]
        acc += tl.dot(e, v, input_precision="ieee")
    out_descr.store(offsets=[pid * bm, 0], value=acc / l[:, None])


def main():
    seq_len = 2048  # assuming prefill
    embed_dim = 64
    torch.manual_seed(0)
    query = torch.randn(seq_len, embed_dim, device="cuda")
    key = torch.randn(seq_len, embed_dim, device="cuda")
    value = torch.randn(seq_len, embed_dim, device="cuda")
    expected_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=False
    )
    actual_output = flash_attn(query, key, value)
    torch.testing.assert_close(expected_output, actual_output)
    print("Success")


if __name__ == "__main__":
    main()
