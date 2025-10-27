# import os

# os.environ["TRITON_INTERPRET"] = "1"
import torch
import triton
import triton.language as tl


def sample(
    weights: torch.Tensor,
    hidden_states: torch.Tensor,
    num_samples: int,
    temperature: float,
    return_probs: bool = False,
):
    logits = weights @ hidden_states  # [seq_len, V]
    logits -= torch.max(logits, dim=0, keepdim=True).values
    probs = torch.nn.functional.softmax(logits / temperature, dim=0)  # [seq_len, V]
    samples = torch.multinomial(probs.T, num_samples=num_samples, replacement=True)
    if return_probs:
        return samples, probs
    return samples


def incremental_sample_pt(
    weights: torch.Tensor,
    hidden_states: torch.Tensor,
    num_samples: int,
    temperature: float,
):
    V, D = weights.shape
    D, seq_len = hidden_states.shape
    block_size = 8
    # compute logits blocks
    gumbel_max = float("-inf") * torch.ones(size=(num_samples, seq_len))
    gumbel_max_idx = torch.empty(size=(num_samples, seq_len), dtype=torch.long)
    n_blocks = cdiv(V, block_size)
    for blk_idx in range(n_blocks):
        idx_from = blk_idx * block_size
        idx_to = (blk_idx + 1) * block_size
        w_blk = weights[idx_from:idx_to]  # [block_size, D]
        logits_blk = w_blk @ hidden_states / temperature  # [seq_len, block_size]
        unif_noise = torch.rand((num_samples, *logits_blk.shape))
        gumbel_noise = -(-unif_noise.log()).log()
        new_max, new_max_idx_local = torch.max(logits_blk + gumbel_noise, dim=1)
        new_max_idx_global = idx_from + new_max_idx_local

        replace_mask = new_max > gumbel_max
        gumbel_max[replace_mask] = new_max[replace_mask]
        gumbel_max_idx[replace_mask] = new_max_idx_global[replace_mask]
    return gumbel_max_idx.T


def cdiv(n: int, div: int) -> int:
    return (n + div - 1) // div


def fused_sample_triton(
    weights: torch.Tensor,
    hidden_states: torch.Tensor,
    num_samples: int,
    temperature: float,
    seed: int,
    block_size_v: int = 8,
):
    V, D = weights.shape
    D, seq_len = hidden_states.shape
    block_size_d = triton.next_power_of_2(D)
    grid_size = triton.cdiv(V, block_size_v)

    maxs = torch.zeros((grid_size, seq_len, num_samples), dtype=torch.float32)
    maxs_idx = torch.zeros_like(maxs, dtype=torch.long)

    # def grid(meta):
    #     return (triton.cdiv(V, meta["BLOCK_SIZE"]),)

    seqlen_p2 = triton.next_power_of_2(seq_len)
    num_samples_p2 = triton.next_power_of_2(num_samples)
    noise_size = block_size_v * seqlen_p2 * num_samples_p2

    grid = (grid_size,)

    fused_sample_triton_kernel[grid](
        weights_ptr=weights,
        hidden_states_ptr=hidden_states,
        max_out_ptr=maxs,
        max_out_idx_ptr=maxs_idx,
        vocab_size=V,
        hidden_size=D,
        seq_len=seq_len,
        num_samples=num_samples,
        temperature=temperature,
        seed=seed,
        BLOCK_SIZE_V=block_size_v,
        BLOCK_SIZE_D=block_size_d,
        noise_size=noise_size,
        num_samples_p2=num_samples_p2,
        seqlen_p2=seqlen_p2,
    )

    # 2nd stage: reduction
    idxs = maxs.max(axis=0).indices
    samples = maxs_idx.gather(dim=0, index=idxs[None, :])
    return samples.squeeze(0)  # [seq_len, num_samples]


@triton.jit
def fused_sample_triton_kernel(
    weights_ptr,
    hidden_states_ptr,
    max_out_ptr,  # [grid_size, seq_len, num_samples]
    max_out_idx_ptr,  # [grid_size, seq_len, num_samples]
    vocab_size,  # V
    hidden_size: tl.constexpr,  # D
    seq_len: tl.constexpr,
    num_samples: tl.constexpr,
    temperature: float,  # should this be a tl.constexpr?
    seed: int,
    BLOCK_SIZE_V: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    noise_size: tl.constexpr,
    num_samples_p2: tl.constexpr,
    seqlen_p2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_V

    # We don't instantiate gumbel_max yet, because each program just writes
    # its local max into main memory for a parallel reduction in stage 2.

    offsets_v = block_start + tl.arange(0, BLOCK_SIZE_V)
    offsets_h = tl.arange(0, BLOCK_SIZE_D)
    mask_v = offsets_v < vocab_size
    mask_h = offsets_h < hidden_size

    w_offsets = offsets_v[:, None] * hidden_size + offsets_h[None, :]
    w_blk = tl.load(
        weights_ptr + w_offsets,
        mask=mask_v[:, None] & mask_h[None, :],
    )

    offset_seqlen = tl.arange(0, seq_len)
    hidden_states_blk = tl.load(
        hidden_states_ptr + offset_seqlen[None, :] + seq_len * offsets_h[:, None],
        mask=mask_h[:, None],
    )
    logits_blk = tl.dot(w_blk, hidden_states_blk) / temperature  # [Vblk, seq_len]

    # Note: Creating appropriately sized tensors is tricky because
    # tl.arange() only accepts tl.constexpr that are powers of 2.
    noise_offsets = tl.arange(0, noise_size).reshape(
        (num_samples_p2, BLOCK_SIZE_V, seqlen_p2)
    )
    # Note: Each program needs a different seed, otherwise they
    # all create the same noise, leading to sampling artifacts.
    unif_noise = tl.rand(seed + pid, noise_offsets)
    gumbel_noise = -tl.log(-tl.log(unif_noise))

    gumbel_max, gumbel_max_idx_local = tl.max(
        logits_blk + gumbel_noise, axis=1, return_indices=True
    )  # [num_samples_p2, seqlen_p2]
    gumbel_max_idx_global = gumbel_max_idx_local + block_start

    out_blk_start = pid * seq_len * num_samples

    # Note: It makes a difference if indices are row-major or column-major
    # Note: The stride needs to match the non-padded shape!
    out_offsets = (
        tl.arange(0, num_samples_p2)[:, None]
        + num_samples * tl.arange(0, seqlen_p2)[None, :]
    )
    out_mask = num_samples > tl.arange(0, num_samples_p2)[:, None]
    tl.store(
        max_out_ptr + out_blk_start + out_offsets,
        gumbel_max,
        mask=out_mask,
    )
    tl.store(
        max_out_idx_ptr + out_blk_start + out_offsets,
        gumbel_max_idx_global,
        mask=out_mask,
    )
