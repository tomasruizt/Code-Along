"""PyTorch references for the 5 warmup ops. These are the ground truth."""
import torch


def vec_add_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def warp_reduce_sum_ref(x: torch.Tensor) -> torch.Tensor:
    # Sum each contiguous chunk of 32 elements. x.numel() must be a multiple of 32.
    return x.view(-1, 32).sum(dim=1)


def block_reduce_sum_ref(x: torch.Tensor) -> torch.Tensor:
    return x.sum()


def transpose_ref(x: torch.Tensor) -> torch.Tensor:
    return x.t().contiguous()


def gemm_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b
