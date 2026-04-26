"""PyTorch reference implementation of vLLM-style paged KV gather.

Layout:
  kv_cache:    [NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
  block_table: [NUM_SEQS, MAX_BLOCKS_PER_SEQ] int32
  seq_lens:    [NUM_SEQS] int32

Output (gather): [NUM_SEQS, max_seq_len, NUM_KV_HEADS, HEAD_DIM], zero-padded.
"""

import math

import torch
from torch import Tensor
import triton
import triton.language as tl


def paged_kv_gather_reference(
    kv_cache: Tensor,
    block_table: Tensor,
    seq_lens: Tensor,
    max_seq_len: int,
) -> Tensor:
    """Naive loop reference. For each (seq, token), do the obvious 3-step lookup.

    Args:
        kv_cache:    [NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM] fp16/bf16
        block_table: [NUM_SEQS, MAX_BLOCKS_PER_SEQ] int32
        seq_lens:    [NUM_SEQS] int32
        max_seq_len: padded output length (typically multiple of BLOCK_SIZE)

    Returns:
        output: [NUM_SEQS, max_seq_len, NUM_KV_HEADS, HEAD_DIM], same dtype as
            kv_cache, zero-padded for positions >= seq_lens[s].
    """
    num_seqs = block_table.shape[0]
    block_size = kv_cache.shape[1]
    num_kv_heads = kv_cache.shape[2]
    head_dim = kv_cache.shape[3]

    output = torch.zeros(
        (num_seqs, max_seq_len, num_kv_heads, head_dim),
        dtype=kv_cache.dtype,
        device=kv_cache.device,
    )

    for s in range(num_seqs):
        L = int(seq_lens[s].item())
        for t in range(L):
            logical_block = t // block_size
            block_offset = t % block_size
            physical_block = int(block_table[s, logical_block].item())
            output[s, t, :, :] = kv_cache[physical_block, block_offset, :, :]

    return output


def make_test_inputs(
    num_seqs: int,
    max_blocks_per_seq: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Generate randomized inputs with a deliberately shuffled block table.

    Returns:
        kv_cache:    [NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM] dtype
        block_table: [NUM_SEQS, MAX_BLOCKS_PER_SEQ] int32, each row a random
            permutation of physical block IDs (non-contiguous, non-monotonic).
        seq_lens:    [NUM_SEQS] int32, each in [1, MAX_BLOCKS_PER_SEQ * BLOCK_SIZE].
        max_seq_len: int, max(seq_lens) rounded up to a multiple of BLOCK_SIZE.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    kv_cache = torch.randn(
        (num_blocks, block_size, num_kv_heads, head_dim),
        dtype=dtype,
        device=device,
        generator=g,
    )

    seq_lens = torch.randint(
        low=1,
        high=max_blocks_per_seq * block_size + 1,
        size=(num_seqs,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )

    # Build a deliberately shuffled (non-contiguous, non-monotonic) block table.
    # Each row gets a random permutation of physical block IDs (truncated to
    # max_blocks_per_seq). All entries are valid indices into kv_cache, so an
    # accidental read of a "garbage" tail entry won't OOB.
    assert num_blocks >= max_blocks_per_seq, (
        "num_blocks must be >= max_blocks_per_seq so each seq can have distinct physical blocks"
    )
    block_table = torch.empty(
        (num_seqs, max_blocks_per_seq), dtype=torch.int32, device=device
    )
    for s in range(num_seqs):
        perm = torch.randperm(num_blocks, device=device, generator=g)[
            :max_blocks_per_seq
        ]
        block_table[s] = perm.to(torch.int32)

    max_actual = int(seq_lens.max().item())
    max_seq_len = math.ceil(max_actual / block_size) * block_size

    return kv_cache, block_table, seq_lens, max_seq_len


def _spot_check(
    kv_cache: Tensor,
    block_table: Tensor,
    seq_lens: Tensor,
    output: Tensor,
    num_samples: int = 8,
    seed: int = 123,
) -> tuple[bool, str]:
    """Independently verify a handful of (seq, token) positions plus padding zeros.

    Args:
        kv_cache:    [NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
        block_table: [NUM_SEQS, MAX_BLOCKS_PER_SEQ] int32
        seq_lens:    [NUM_SEQS] int32
        output:      [NUM_SEQS, max_seq_len, NUM_KV_HEADS, HEAD_DIM] gather result
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    block_size = kv_cache.shape[1]
    num_seqs = block_table.shape[0]

    for _ in range(num_samples):
        s = int(torch.randint(0, num_seqs, (1,), generator=g).item())
        L = int(seq_lens[s].item())
        if L == 0:
            continue
        t = int(torch.randint(0, L, (1,), generator=g).item())
        logical_block = t // block_size
        block_offset = t % block_size
        physical_block = int(block_table[s, logical_block].item())
        expected = kv_cache[physical_block, block_offset]
        actual = output[s, t]
        if not torch.equal(expected, actual):
            return (
                False,
                f"mismatch at seq={s} token={t} (phys={physical_block}, off={block_offset})",
            )

    # Check padded tail is zero for each sequence.
    max_seq_len = output.shape[1]
    for s in range(num_seqs):
        L = int(seq_lens[s].item())
        if L < max_seq_len:
            tail = output[s, L:]
            if not torch.all(tail == 0):
                return False, f"non-zero padding for seq={s} beyond seq_len={L}"

    return True, "ok"


def paged_kv_gather_triton(
    kv_cache: Tensor,
    block_table: Tensor,
    seq_lens: Tensor,
    max_seq_len: int,
) -> Tensor:
    """
    Triton implementation. Same signature/output as paged_kv_gather_reference.

    grid = (num_seqs, max_seq_len)
    Each program collects the kv-cache for a single cell of the output.
    In terms of data, its (N_HEADS, HEAD_DIM) data.
    """
    N_BLOCKS, BLOCK_SIZE, N_HEADS, HEAD_DIM = kv_cache.shape
    N_SEQS, MAX_BLOCKS_PER_SEQ = block_table.shape
    grid = (N_SEQS, max_seq_len)
    print("grid =", grid)
    out = torch.zeros(
        (N_SEQS, max_seq_len, N_HEADS, HEAD_DIM),
        device=kv_cache.device,
        dtype=kv_cache.dtype,
    )
    kv_lookup_kernel[grid](
        seq_lens,
        block_table,
        kv_cache,
        out,
        BLOCK_SIZE,
        N_SEQS,
        MAX_BLOCKS_PER_SEQ,
        DATA_SIZE=N_HEADS * HEAD_DIM,
        MAX_SEQ_LEN=max_seq_len,
    )
    return out


@triton.jit
def kv_lookup_kernel(
    seqlens_ptr,
    blk_tbl_ptr,
    kv_cache_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    N_SEQS: tl.constexpr,
    MAX_BLOCKS_PER_SEQ: tl.constexpr,
    DATA_SIZE: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
):
    seq = tl.program_id(0)
    tok_idx = tl.program_id(1)

    # if this seq is shorter than this out_col return immediately
    seqlen = tl.load(seqlens_ptr + seq)
    if seqlen <= tok_idx:
        # e.g. seqlen=1 (single token), and tok_idx=0, false, do-work ✅
        # e.g. seqlen=1 (single token), and tok_idx=1, true, return ✅
        # e.g. seqlen=1 (single token), and tok_idx=2, true, return ✅
        return

    blk_tbl_col = tok_idx // BLOCK_SIZE
    blk_offset = tok_idx % BLOCK_SIZE

    blk_idx = tl.load(blk_tbl_ptr + MAX_BLOCKS_PER_SEQ * seq + blk_tbl_col)

    # read from kvcache at row=blk_idx, col=blk_offset
    # col stride = head_dim * n_heads
    # row stride = (head_dim * n_heads) * BLOCK_SIZE
    col_stride = DATA_SIZE
    row_stride = DATA_SIZE * BLOCK_SIZE
    data = tl.load(
        kv_cache_ptr
        + col_stride * blk_offset
        + row_stride * blk_idx
        + tl.arange(0, DATA_SIZE)
    )

    # save in the output row=seq, col=tok_idx
    # col stride = data
    # row stride = data * MAX_SEQ_LEN
    col_stride = DATA_SIZE
    row_stride = DATA_SIZE * MAX_SEQ_LEN
    tl.store(
        out_ptr + seq * row_stride + tok_idx * col_stride + tl.arange(0, DATA_SIZE),
        data,
    )


def _run_config(name, dtype, **kwargs):
    kv_cache, block_table, seq_lens, max_seq_len = make_test_inputs(
        dtype=dtype, **kwargs
    )

    ref = paged_kv_gather_reference(kv_cache, block_table, seq_lens, max_seq_len)
    out = paged_kv_gather_triton(kv_cache, block_table, seq_lens, max_seq_len)

    ref_spot_ok, ref_spot_msg = _spot_check(kv_cache, block_table, seq_lens, ref)
    try:
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        triton_ok, triton_msg = True, "ok"
    except AssertionError as e:
        triton_ok, triton_msg = False, str(e).splitlines()[0]

    ok = ref_spot_ok and triton_ok
    status = "✅ PASS" if ok else "❌ FAIL"
    print(
        f"[{status}] {name}  dtype={dtype}  "
        f"ref_spot: {ref_spot_msg}  triton_vs_ref: {triton_msg}  "
        f"max_seq_len={max_seq_len}  seq_lens={seq_lens.tolist()}"
    )
    return ok


if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "this script expects CUDA"

    configs = [
        (
            "small",
            dict(
                num_seqs=4,
                max_blocks_per_seq=8,
                block_size=16,
                num_kv_heads=8,
                head_dim=64,
                num_blocks=128,
            ),
        ),
        (
            "realistic",
            dict(
                num_seqs=8,
                max_blocks_per_seq=16,
                block_size=16,
                num_kv_heads=4,
                head_dim=128,
                num_blocks=512,
            ),
        ),
        (
            "longseq",
            dict(
                num_seqs=2,
                max_blocks_per_seq=32,
                block_size=16,
                num_kv_heads=2,
                head_dim=64,
                num_blocks=256,
            ),
        ),
    ]

    all_ok = True
    for name, cfg in configs:
        for dtype in (torch.float16, torch.bfloat16):
            ok = _run_config(name, dtype=dtype, **cfg)
            all_ok = all_ok and ok

    print()
    print("ALL PASS" if all_ok else "SOME FAILURES")
