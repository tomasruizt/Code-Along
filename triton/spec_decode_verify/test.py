"""Validates the Triton verify kernel against the PyTorch reference.

Usage:
    python test.py             # run against the Triton skeleton
    python test.py --self      # sanity check: reference vs reference (always passes)

The same (draft_probs, target_probs, draft_tokens, uniform_samples,
bonus_uniform) are fed to both implementations; we then compare
`output_tokens` and `num_accepted` exactly with `torch.equal`.
"""

import sys
import torch

from reference import spec_decode_verify_reference
from exercise import spec_decode_verify_triton


def make_test_inputs(batch, K, vocab, mode="random", device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)

    draft_probs = torch.softmax(
        torch.randn(batch, K, vocab, generator=g, device=device), dim=-1
    )

    if mode == "random":
        target_probs = torch.softmax(
            torch.randn(batch, K + 1, vocab, generator=g, device=device), dim=-1
        )
    elif mode == "all_accept":
        # first K positions equal draft_probs (so r=1, always accept);
        # the trailing slice is the bonus distribution.
        bonus_slice = torch.softmax(
            torch.randn(batch, 1, vocab, generator=g, device=device), dim=-1
        )
        target_probs = torch.cat([draft_probs.clone(), bonus_slice], dim=1)
    elif mode == "first_reject":
        target_probs = torch.softmax(
            torch.randn(batch, K + 1, vocab, generator=g, device=device), dim=-1
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    flat = draft_probs.reshape(-1, vocab)
    draft_tokens = (
        torch.multinomial(flat, num_samples=1, generator=g)
        .view(batch, K)
        .to(torch.int32)
    )

    if mode == "first_reject":
        # zero out target prob at the chosen draft token at position 0, renormalize.
        for b in range(batch):
            t = int(draft_tokens[b, 0].item())
            target_probs[b, 0, t] = 0.0
        s = target_probs[:, 0].sum(dim=-1, keepdim=True)
        target_probs[:, 0] = target_probs[:, 0] / s

    uniform_samples = torch.rand(batch, K, generator=g, device=device)
    bonus_uniform = torch.rand(batch, generator=g, device=device)

    return draft_tokens, draft_probs, target_probs, uniform_samples, bonus_uniform


def validate(impl_fn, label, **inputs_kwargs):
    inputs = make_test_inputs(**inputs_kwargs)
    ref_tokens, ref_acc = spec_decode_verify_reference(*inputs)

    try:
        out_tokens, out_acc = impl_fn(*inputs)
    except NotImplementedError as e:
        print(f"  ❌ not implemented {label}: {e}")
        return False

    ref_acc_i32 = ref_acc.to(torch.int32)
    ref_tok_i32 = ref_tokens.to(torch.int32)
    ok_acc = torch.equal(ref_acc_i32, out_acc.to(torch.int32))
    ok_tok = torch.equal(ref_tok_i32, out_tokens.to(torch.int32))

    print(f"  {'✅' if ok_acc else '❌'} num_accepted   {label}  ref={ref_acc.tolist()}")
    if not ok_acc:
        print(f"      got={out_acc.tolist()}")
    print(f"  {'✅' if ok_tok else '❌'} output_tokens  {label}")
    if not ok_tok:
        # show the first row that disagrees
        for b in range(ref_tokens.shape[0]):
            if not torch.equal(ref_tokens[b], out_tokens[b]):
                print(f"      first diff at b={b}")
                print(f"        ref={ref_tokens[b].tolist()}")
                print(f"        got={out_tokens[b].tolist()}")
                break

    return ok_acc and ok_tok


if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA required"

    self_check = "--self" in sys.argv
    impl = spec_decode_verify_reference if self_check else spec_decode_verify_triton
    impl_label = "reference (self-check)" if self_check else "triton"

    configs = [
        dict(batch=4,  K=5, vocab=128,   mode="random"),
        dict(batch=16, K=8, vocab=32000, mode="random"),
        dict(batch=4,  K=5, vocab=128,   mode="all_accept"),
        dict(batch=4,  K=5, vocab=128,   mode="first_reject"),
    ]

    print(f"\n========== {impl_label} ==========")
    all_ok = True
    for cfg in configs:
        label = f"[{cfg['mode']:13s} b={cfg['batch']:2d} K={cfg['K']} V={cfg['vocab']}]"
        ok = validate(impl, label, **cfg)
        all_ok = all_ok and ok

    print()
    print("✅ ALL PASS" if all_ok else "❌ SOME FAILURES")
    sys.exit(0 if all_ok else 1)
