"""Triton skeleton for speculative-decoding rejection-sampling verification.

Read reference.py for the algorithm and the RNG conventions you need to
match exactly (uniform reuse on reject, bonus_uniform[b], inverse-CDF
via `sum(cdf <= u)`, output zero-init).

You are filling in two things:
    1. `verify_kernel` — the Triton kernel body.
    2. `spec_decode_verify_triton` — the launcher (allocate outputs, pick
       the grid, call the kernel).

The signature below is a suggestion, change it as you like.
"""

import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def verify_kernel(
    draft_tokens_ptr,  # [batch, K]            int32
    draft_probs_ptr,  # [batch, K, vocab]     float32
    target_probs_ptr,  # [batch, K+1, vocab]   float32 (last slice = bonus dist)
    uniform_samples_ptr,  # [batch, K]            float32
    bonus_uniform_ptr,  # [batch]               float32
    output_tokens_ptr,  # [batch, K+1]          int32
    num_accepted_ptr,  # [batch]               int32
    K: tl.constexpr,
    VOCAB: tl.constexpr,
):
    """
    Suggested grid: (batch,). One program per sequence.

    Per-program structure:
        b = tl.program_id(0)

        accepted_count = K  (will be overwritten if we reject)

        for i in 0..K-1 (sequential — rejection is order-dependent):
            t = load draft_tokens[b, i]                          (scalar int)
            u = load uniform_samples[b, i]                       (scalar fp32)
            p_t = load target_probs[b, i, t]                     (scalar fp32)
            q_t = load draft_probs[b, i, t]                      (scalar fp32)
            r   = p_t / q_t

            if u < min(1.0, r):
                store output_tokens[b, i] = t
                continue

            # rejected → residual sample at position i, reusing the same u
            v_off = tl.arange(0, VOCAB)
            tp = load target_probs[b, i, :]                      (VOCAB-vector)
            dp = load draft_probs[b, i, :]                       (VOCAB-vector)
            res = tl.maximum(tp - dp, 0.0)
            res = res / tl.sum(res)
            cdf = tl.cumsum(res, axis=0)
            tok = tl.sum((cdf <= u).to(tl.int32))                # token index

            store output_tokens[b, i] = tok
            accepted_count = i
            break out of the loop

        if accepted_count == K:
            # all K accepted → bonus token sampled from target_probs[b, K]
            u_bonus = load bonus_uniform[b]
            tp = load target_probs[b, K, :]
            cdf = tl.cumsum(tp, axis=0)
            bonus = tl.sum((cdf <= u_bonus).to(tl.int32))
            store output_tokens[b, K] = bonus

        store num_accepted[b] = accepted_count

    Hints:
        - VOCAB and K as tl.constexpr lets you use tl.arange(0, VOCAB)
          and a static-trip-count loop.
        - Triton has no early `break`; emulate it with a boolean flag like
          `done = False` and gate work with `if not done: ...`.
        - `tl.cumsum(x, axis=0)` works on 1-D tiles; cast the boolean
          comparison with `.to(tl.int32)` before `tl.sum`.
        - The output is zero-initialized in the launcher, so any positions
          you don't write stay at 0 (matches the reference).
    """
    row = tl.program_id(0)
    V_BLOCK: tl.constexpr = 256
    probs_base_d = row * K * VOCAB
    probs_base_t = row * (K + 1) * VOCAB

    num_accepted = 0
    finished = False

    k = 0
    while not finished:
        draft_tok_idx = tl.load(draft_tokens_ptr + row * K + k)
        p_d = tl.load(draft_probs_ptr + probs_base_d + k * VOCAB + draft_tok_idx)
        p_t = tl.load(target_probs_ptr + probs_base_t + k * VOCAB + draft_tok_idx)
        r = tl.minimum(p_t / p_d, 1.0)
        u = tl.load(uniform_samples_ptr + row * K + k)
        if u < r:  # accept
            tl.store(output_tokens_ptr + row * (K + 1) + k, draft_tok_idx)
            num_accepted += 1
        else:  # resample
            # Sampling Loop

            # Compute residual sum
            p_res_sum = 0.0
            for vidx in range(0, VOCAB, V_BLOCK):
                v_offs = vidx + tl.arange(0, V_BLOCK)
                p_mask = v_offs < VOCAB
                p_d_full = tl.load(
                    draft_probs_ptr + probs_base_d + k * VOCAB + v_offs,
                    mask=p_mask,
                    other=0.0,
                )
                p_t_full = tl.load(
                    target_probs_ptr + probs_base_t + k * VOCAB + v_offs,
                    mask=p_mask,
                    other=0.0,
                )
                p_new = tl.maximum(0.0, p_t_full - p_d_full)
                p_res_sum += p_new.sum()

            p_sum = 0.0
            sampled_tok_idx = 0
            finished2 = False
            vidx = 0
            while not finished2:
                v_offs = vidx + tl.arange(0, V_BLOCK)
                p_mask = v_offs < VOCAB
                p_d_full = tl.load(
                    draft_probs_ptr + probs_base_d + k * VOCAB + v_offs,
                    mask=p_mask,
                    other=0.0,
                )
                p_t_full = tl.load(
                    target_probs_ptr + probs_base_t + k * VOCAB + v_offs,
                    mask=p_mask,
                    other=0.0,
                )
                p_new = tl.maximum(0.0, p_t_full - p_d_full)
                # reuse u
                sampled_tok_idx += (
                    (p_sum + p_new.cumsum() / p_res_sum < u).sum().to(tl.int32)
                )
                if sampled_tok_idx % V_BLOCK != 0:
                    # sum_idx is the sampled token
                    tl.store(output_tokens_ptr + row * (K + 1) + k, sampled_tok_idx)
                    finished = True
                    finished2 = True
                if vidx + V_BLOCK > VOCAB:
                    finished2 = True
                vidx += V_BLOCK
                p_sum += p_new.sum() / p_res_sum
        if k == K - 1:
            finished = True
        k += +1

    # Store num_accepted
    tl.store(num_accepted_ptr + row, num_accepted)

    if num_accepted != K:
        return

    # One bonus token!
    u2 = tl.load(bonus_uniform_ptr + row)
    p_sum = 0.0
    sampled_tok_idx = 0
    finished3 = False
    vidx = 0
    # for vidx in range(0, VOCAB, V_BLOCK):
    while not finished3:
        v_offs = vidx + tl.arange(0, V_BLOCK)
        p_mask = v_offs < VOCAB
        p_t = tl.load(
            target_probs_ptr + probs_base_t + K * VOCAB + v_offs,
            mask=p_mask,
            other=0.0,
        )
        sampled_tok_idx += (p_sum + p_t.cumsum() < u2).sum().to(tl.int32)
        if sampled_tok_idx % V_BLOCK != 0:
            tl.store(output_tokens_ptr + row * (K + 1) + K, sampled_tok_idx)
            finished3 = True
        if vidx + V_BLOCK > VOCAB:
            finished3 = True
        vidx += V_BLOCK
        p_sum += p_t.sum()


def spec_decode_verify_triton(
    draft_tokens: Tensor,
    draft_probs: Tensor,
    target_probs: Tensor,
    uniform_samples: Tensor,
    bonus_uniform: Tensor,
) -> tuple[Tensor, Tensor]:
    """Launcher: allocate outputs, choose the grid, call verify_kernel."""
    B, K, VOCAB = draft_probs.shape
    output_tokens = torch.zeros(
        (B, K + 1), dtype=torch.int32, device=draft_tokens.device
    )
    num_accepted = torch.full((B,), -1, dtype=torch.int32, device=draft_tokens.device)
    grid = (B,)
    verify_kernel[grid](
        draft_tokens_ptr=draft_tokens,
        draft_probs_ptr=draft_probs,
        target_probs_ptr=target_probs,
        uniform_samples_ptr=uniform_samples,
        bonus_uniform_ptr=bonus_uniform,
        output_tokens_ptr=output_tokens,
        num_accepted_ptr=num_accepted,
        K=K,
        VOCAB=VOCAB,
    )
    return output_tokens, num_accepted
