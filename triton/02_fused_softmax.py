import torch


def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


def triton_softmax(x):
    y = torch.empty_like(x)
    return softmax_kernel(x, y, *x.shape)


def softmax_kernel(x_ptr, y_ptr, m, n):
    raise NotImplementedError


if __name__ == "__main__":
    x = torch.randn(5, 3)
    sm1 = naive_softmax(x)
    sm2 = triton_softmax(x)
    print(sm1)
    print(sm2)
    assert torch.allclose(sm1, sm2)
