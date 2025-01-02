import math
import torch

torch.manual_seed(0)

x = torch.randn(6)
s = x.softmax(dim=0)
print(x)
print(s)


def py_softmax(x, out, n):
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


s_py = [None] * len(x)
py_softmax(x.tolist(), s_py, len(x))
print(torch.tensor(s_py))
assert torch.allclose(torch.tensor(s_py), s)
