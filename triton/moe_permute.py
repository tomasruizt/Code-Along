# import os

# os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton.language as tl
import triton


def moe_permute_reference(
    tokens: torch.Tensor, expert_ids: torch.Tensor, num_experts: int
):
    """
    MoE token permutation (top-1 routing).

    In a Mixture-of-Experts layer, a router assigns each input token to one
    expert. Before running the per-expert feed-forward networks, we want to
    group all tokens going to the same expert into a contiguous range so each
    expert's FFN can be a single batched matmul. This function performs that
    grouping: it reorders the rows of `tokens` so that all rows whose
    `expert_ids[i] == 0` come first, then all rows for expert 1, and so on
    through `num_experts - 1`. Within each expert's group, the original token
    order is preserved (stable sort).

    It also returns:
      - `expert_offsets`, the cumulative token counts that tell each expert's
        FFN which slice of `permuted_tokens` to consume.
      - `inverse_perm`, the permutation that un-groups the tokens back to
        their original positions after the per-expert computation finishes
        (so the layer's output lines up with its input).

    Inputs:
        tokens:      [num_tokens, hidden]    float16 / bfloat16 / float32
        expert_ids:  [num_tokens]            int64, values in [0, num_experts)
        num_experts: int

    Returns:
        permuted_tokens: [num_tokens, hidden]   tokens reordered so all rows for
                                                expert e are contiguous, in
                                                ascending expert order.
        inverse_perm:    [num_tokens]           int64. permuted_tokens[inverse_perm] == tokens.
        expert_offsets:  [num_experts + 1]      int64. Tokens for expert e live at
                                                permuted_tokens[expert_offsets[e]:expert_offsets[e+1]].
    """
    perm = torch.argsort(expert_ids, stable=True)
    permuted_tokens = tokens[perm]
    inverse_perm = torch.argsort(perm)

    counts = torch.bincount(expert_ids, minlength=num_experts).to(torch.int64)
    zero = torch.zeros(1, dtype=torch.int64, device=expert_ids.device)
    expert_offsets = torch.cat([zero, torch.cumsum(counts, dim=0)])

    return permuted_tokens, inverse_perm, expert_offsets


def moe_permute_triton(
    tokens: torch.Tensor, expert_ids: torch.Tensor, num_experts: int
):
    dvc = tokens.device
    N_TOKS, D = tokens.shape
    # I reuse the program over tokens to be a program
    # over experts. However, if there are less tokens than experts
    # this would fail. The solution is to launch at least max(N_TOKS, N_EXPERTS)
    # programs, and in the program body to ignore the program i if needed.
    grid = (max(N_TOKS, num_experts),)
    permuted_tokens = torch.empty_like(tokens, device=dvc)
    inverse_perm = torch.empty((N_TOKS,), dtype=torch.long, device=dvc)
    expert_offsets = torch.full(
        (num_experts + 1,), N_TOKS, dtype=torch.long, device=dvc
    )
    expert_offsets[0] = 0
    moe_permute_kernel[grid](
        tokens,
        expert_ids,
        permuted_tokens,
        inverse_perm,
        expert_offsets,
        N_TOKS,
        D,
        N_EXPERTS=num_experts,
    )
    return permuted_tokens, inverse_perm, expert_offsets


@triton.jit
def moe_permute_kernel(
    tokens_ptr,
    expert_ids_ptr,
    permuted_toks_ptr,
    inverse_perm_ptr,
    expert_offsets_ptr,
    N_TOKS: tl.constexpr,
    D: tl.constexpr,
    N_EXPERTS: tl.constexpr,
):
    i = tl.program_id(0)
    expert_ids = tl.load(expert_ids_ptr + tl.arange(0, N_TOKS))

    if i < N_TOKS:
        row = i
        expert_id = tl.load(expert_ids_ptr + row)
        # e.g. expert range is [2, 5]
        expert_start_idx = (expert_ids < expert_id).sum()

        # e.g. row has 2 of the same expert before
        expert_before = ((expert_ids == expert_id) & (tl.arange(0, N_TOKS) < row)).sum()
        out_row = expert_start_idx + expert_before

        # load token data
        data = tl.load(tokens_ptr + row * D + tl.arange(0, D))

        # store data
        tl.store(permuted_toks_ptr + out_row * D + tl.arange(0, D), data)

        # store inverse perm
        tl.store(inverse_perm_ptr + row, out_row)

    # store expert offsets
    if i < N_EXPERTS:
        # end idx is exclusive (not inclusive)
        expert_i_end_idx = (expert_ids <= i).sum()
        tl.store(expert_offsets_ptr + i + 1, expert_i_end_idx)


def make_test_inputs(
    num_tokens,
    hidden,
    num_experts,
    dtype=torch.float16,
    device="cuda",
    seed=0,
    imbalance="uniform",
):
    g = torch.Generator(device=device).manual_seed(seed)
    tokens = torch.randn(num_tokens, hidden, dtype=dtype, device=device, generator=g)

    if imbalance == "uniform":
        expert_ids = torch.randint(
            0, num_experts, (num_tokens,), dtype=torch.int64, device=device, generator=g
        )
    elif imbalance == "skewed":
        # expert 0 -> 50%, expert 1 -> 25%, rest split the remaining 25%
        n0 = num_tokens // 2
        n1 = num_tokens // 4
        rest = num_tokens - n0 - n1
        expert_ids = torch.empty(num_tokens, dtype=torch.int64, device=device)
        expert_ids[:n0] = 0
        expert_ids[n0 : n0 + n1] = 1
        if num_experts > 2:
            tail = torch.randint(
                2, num_experts, (rest,), dtype=torch.int64, device=device, generator=g
            )
        else:
            tail = torch.zeros(rest, dtype=torch.int64, device=device)
        expert_ids[n0 + n1 :] = tail
        # shuffle so positions are mixed (more realistic)
        shuffle_idx = torch.randperm(num_tokens, device=device, generator=g)
        expert_ids = expert_ids[shuffle_idx]
    elif imbalance == "one_hot":
        expert_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    elif imbalance == "sparse":
        # only even experts get tokens
        even_experts = torch.arange(0, num_experts, 2, device=device, dtype=torch.int64)
        idx = torch.randint(
            0,
            even_experts.numel(),
            (num_tokens,),
            dtype=torch.int64,
            device=device,
            generator=g,
        )
        expert_ids = even_experts[idx]
    else:
        raise ValueError(f"Unknown imbalance mode: {imbalance}")

    return tokens, expert_ids


def _check_grouping(expert_offsets, expert_ids, perm, num_experts):
    # For each expert e, the slice [offsets[e]:offsets[e+1]] of perm must point to
    # positions where expert_ids == e.
    for e in range(num_experts):
        lo = int(expert_offsets[e].item())
        hi = int(expert_offsets[e + 1].item())
        if lo == hi:
            continue
        slice_perm = perm[lo:hi]
        ids_in_slice = expert_ids[slice_perm]
        if not torch.all(ids_in_slice == e):
            return (
                False,
                f"expert {e}: slice contains expert ids {ids_in_slice.unique().tolist()}",
            )
    return True, ""


def _check_offsets(expert_offsets, num_tokens):
    if int(expert_offsets[0].item()) != 0:
        return False, f"offsets[0] = {int(expert_offsets[0].item())}, expected 0"
    if int(expert_offsets[-1].item()) != num_tokens:
        return (
            False,
            f"offsets[-1] = {int(expert_offsets[-1].item())}, expected {num_tokens}",
        )
    diffs = expert_offsets[1:] - expert_offsets[:-1]
    if not torch.all(diffs >= 0):
        return False, "offsets not monotonically non-decreasing"
    return True, ""


def _run_config(num_tokens, hidden, num_experts, imbalance, dtype, verbose=False):
    tokens, expert_ids = make_test_inputs(
        num_tokens, hidden, num_experts, dtype=dtype, imbalance=imbalance
    )
    permuted_tokens, inverse_perm, expert_offsets = moe_permute_reference(
        tokens, expert_ids, num_experts
    )
    perm = torch.argsort(expert_ids, stable=True)

    label = f"[n={num_tokens}, h={hidden}, E={num_experts}, {imbalance}, {dtype}]"

    if verbose:
        print(f"--- {label} ---")
        print("tokens:")
        print(tokens)
        print("expert_ids:", expert_ids.tolist())
        print("perm:", perm.tolist())
        print("permuted_tokens:")
        print(permuted_tokens)
        print("inverse_perm:", inverse_perm.tolist())
        print("expert_offsets:", expert_offsets.tolist())

    ok_group, msg_group = _check_grouping(expert_offsets, expert_ids, perm, num_experts)
    print(f"  {'✅' if ok_group else '❌'} grouping {label} {msg_group}")

    roundtrip = permuted_tokens[inverse_perm]
    ok_round = torch.equal(roundtrip, tokens)
    print(f"  {'✅' if ok_round else '❌'} roundtrip {label}")

    ok_off, msg_off = _check_offsets(expert_offsets, num_tokens)
    print(f"  {'✅' if ok_off else '❌'} offsets {label} {msg_off}")

    return ok_group and ok_round and ok_off


def validate_against_reference(triton_fn, **kwargs):
    num_tokens = kwargs.get("num_tokens")
    hidden = kwargs.get("hidden")
    num_experts = kwargs.get("num_experts")
    imbalance = kwargs.get("imbalance", "uniform")
    dtype = kwargs.get("dtype", torch.float16)

    tokens, expert_ids = make_test_inputs(**kwargs)
    ref_permuted, ref_inv, ref_offsets = moe_permute_reference(
        tokens, expert_ids, num_experts
    )
    tri_permuted, tri_inv, tri_offsets = triton_fn(tokens, expert_ids, num_experts)

    label = f"[n={num_tokens}, h={hidden}, E={num_experts}, {imbalance}, {dtype}]"

    ok_offsets = torch.equal(ref_offsets, tri_offsets)
    print(f"  {'✅' if ok_offsets else '❌'} expert_offsets match {label}")

    ok_permuted = torch.equal(ref_permuted, tri_permuted)
    print(
        f"  {'✅' if ok_permuted else '❌'} permuted_tokens match (stable order) {label}"
    )

    # Roundtrip is the load-bearing invariant: works for both stable and atomic-scatter kernels.
    roundtrip = tri_permuted[tri_inv]
    ok_round = torch.equal(roundtrip, tokens)
    print(f"  {'✅' if ok_round else '❌'} roundtrip via triton inverse_perm {label}")

    ok_inv_exact = torch.equal(ref_inv, tri_inv)
    note = (
        "(matches stable reference)"
        if ok_inv_exact
        else "(differs — kernel likely uses atomic scatter; roundtrip is what matters)"
    )
    print(f"  {'✅' if ok_inv_exact else 'ℹ️ '} inverse_perm exact match {label} {note}")

    return ok_offsets and ok_permuted and ok_round


if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA required"

    configs = [
        dict(num_tokens=16, hidden=8, num_experts=4, imbalance="uniform", verbose=True),
        dict(num_tokens=1024, hidden=512, num_experts=8, imbalance="uniform"),
        dict(num_tokens=2048, hidden=1024, num_experts=16, imbalance="skewed"),
        dict(num_tokens=128, hidden=64, num_experts=8, imbalance="one_hot"),
        dict(num_tokens=256, hidden=64, num_experts=8, imbalance="sparse"),
    ]

    all_ok = True
    for dtype in (torch.float16, torch.bfloat16):
        print(f"\n========== triton, dtype={dtype} ==========")
        for cfg in configs:
            kwargs = {k: v for k, v in cfg.items() if k != "verbose"}
            try:
                ok = validate_against_reference(
                    moe_permute_triton, dtype=dtype, **kwargs
                )
            except NotImplementedError:
                print(
                    f"  ❌ moe_permute_triton not implemented [n={kwargs['num_tokens']}, {kwargs['imbalance']}, {dtype}]"
                )
                ok = False
            all_ok = all_ok and ok

    print("\n" + ("✅ ALL CHECKS PASSED" if all_ok else "❌ SOME CHECKS FAILED"))
