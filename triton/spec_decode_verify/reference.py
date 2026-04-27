"""PyTorch reference for speculative-decoding rejection-sampling verification.

The verify step takes K draft tokens proposed by a fast draft model and either
accepts them, rejects+resamples at the first failing position, or appends a
bonus token if all K were accepted.

Inputs:
    draft_tokens:    [batch, K]            int32
    draft_probs:     [batch, K, vocab]     float32, draft model probabilities
    target_probs:    [batch, K+1, vocab]   float32, target model probabilities;
                                           the (K+1)-th slice is the bonus
                                           distribution conditioned on all K
                                           accepted draft tokens.
    uniform_samples: [batch, K]            float32 in [0,1)
    bonus_uniform:   [batch]               float32 in [0,1)

Outputs:
    output_tokens:   [batch, K+1]          int32, accepted tokens + (resample or bonus)
    num_accepted:    [batch]               int32

Algorithm (per sequence b):
    For i = 0..K-1:
        t = draft_tokens[b, i]
        r = target_probs[b, i, t] / draft_probs[b, i, t]
        if uniform_samples[b, i] < min(1, r):
            output_tokens[b, i] = t  (accept)
        else:
            residual[v] = max(0, target_probs[b, i, v] - draft_probs[b, i, v])
            residual /= residual.sum()
            output_tokens[b, i] = inverse_cdf_sample(residual, u_resid)
            num_accepted[b] = i; break
    If we never broke (all K accepted):
        output_tokens[b, K] = inverse_cdf_sample(target_probs[b, K], u_bonus)
        num_accepted[b] = K

RNG conventions (Triton kernel must match these to pass torch.equal):
    - uniform_samples[b, i] is consumed twice when position i rejects:
      once for the accept/reject test, then again as the inverse-CDF u for
      the residual sample at the same position. (Slightly biased but
      common; the point here is to be deterministic, not statistically pure.)
    - bonus_uniform[b] is the inverse-CDF u for the bonus token.
    - For the bonus we sample from target_probs[b, K] (the extra trailing
      distribution the target model produces for free).

Inverse-CDF sampling convention:
    cdf = cumsum(probs); idx = sum(cdf <= u). Equivalent to
    torch.searchsorted(cdf, u, right=True). Output zero-initialized so
    positions past num_accepted compare equal to the Triton output (which
    must also leave them zeroed).
"""

import torch
from torch import Tensor


def _inverse_cdf_sample(probs: Tensor, u: float) -> int:
    """Inverse-CDF sample from a 1D distribution `probs` using uniform `u`."""
    cdf = torch.cumsum(probs, dim=0)
    u_t = torch.tensor(u, device=probs.device, dtype=probs.dtype)
    idx = int((cdf <= u_t).sum().item())
    if idx >= probs.numel():
        idx = probs.numel() - 1
    return idx


def spec_decode_verify_reference(
    draft_tokens: Tensor,
    draft_probs: Tensor,
    target_probs: Tensor,
    uniform_samples: Tensor,
    bonus_uniform: Tensor,
) -> tuple[Tensor, Tensor]:
    batch, K = draft_tokens.shape
    device = draft_tokens.device

    output_tokens = torch.zeros((batch, K + 1), dtype=torch.int32, device=device)
    num_accepted = torch.zeros((batch,), dtype=torch.int32, device=device)

    for b in range(batch):
        accepted_count = K  # set to K unless we break out below
        for i in range(K):
            t = int(draft_tokens[b, i].item())
            p_t = float(target_probs[b, i, t].item())
            q_t = float(draft_probs[b, i, t].item())
            r = p_t / q_t if q_t > 0.0 else float("inf")
            u = float(uniform_samples[b, i].item())

            if u < min(1.0, r):
                output_tokens[b, i] = t
                continue

            residual = torch.clamp(
                target_probs[b, i] - draft_probs[b, i], min=0.0
            )
            s = residual.sum()
            if s > 0:
                residual = residual / s
            else:
                residual = target_probs[b, i].clone()  # degenerate fallback
            tok = _inverse_cdf_sample(residual, u)
            output_tokens[b, i] = tok
            accepted_count = i
            break

        if accepted_count == K:
            u_bonus = float(bonus_uniform[b].item())
            bonus = _inverse_cdf_sample(target_probs[b, K], u_bonus)
            output_tokens[b, K] = bonus

        num_accepted[b] = accepted_count

    return output_tokens, num_accepted
