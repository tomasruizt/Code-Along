import math
import torch


def py_softmax(x: torch.Tensor) -> torch.Tensor:
    s = [None] * len(x)
    py_softmax_kernel(x.tolist(), s, len(x))
    return torch.tensor(s)


def py_softmax_kernel(x: list[float], out: list[float], n: int):
    d_jm1 = 0.0  # normalization term
    m_jm1 = float("-inf")  # maximum value
    for j in range(n):
        xj = x[j]
        mj = max(m_jm1, xj)
        dj = d_jm1 * math.exp(m_jm1 - mj) + math.exp(xj - mj)
        m_jm1 = mj
        d_jm1 = dj

    mv = mj
    dv = dj

    for i in range(n):
        yi = math.exp(x[i] - mv) / dv
        out[i] = yi


x = torch.arange(5).float()
s_torch = x.softmax(dim=0)
s_py = py_softmax(x)
print("x =".rjust(20), repr(x))
print("torch::softmax(x) =".rjust(20), repr(s_torch))
print("python::softmax(x) =".rjust(20), repr(s_py))
assert torch.allclose(s_py, s_torch)
