from scipy.optimize import root
from torch import Tensor
import torch
from torch.linalg import solve

jac = torch.autograd.functional.jacobian
t = torch.tensor

def g1(p, x): return (p[0] - p[1]) * x[0]**2 - x[1]

def g2(p, x): return torch.log(x[1]) + 1 - p[2] * x[0]

def g(p, x): return torch.stack([g1(p, x), g2(p, x)])

def _find_x(p: Tensor, x_guess: Tensor):
    def f(x):
        return g_np(p.numpy(), x)

    def _jac_x(x):
        return jacobians(p, t(x))[1].numpy()

    with torch.no_grad():
        solution = root(f, x0=x_guess.numpy(), jac=_jac_x, method="lm")
    
    return t(solution.x, dtype=torch.float32)

def g_np(p, x): return g(t(p), t(x)).numpy()

def jacobians(p, x):
    dgdp, dgdx = jac(g, (p, x))
    return dgdp, dgdx


class SolveG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p: Tensor, x_guess: Tensor) -> Tensor:
        x = _find_x(p, x_guess)
        ctx.save_for_backward(p, x)
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        p, x = ctx.saved_tensors
        dgdp, dgdx = jacobians(p, x)
        v = solve(dgdx.T, -grad_output)
        grad_p = v @ dgdp
        return grad_p, torch.zeros_like(x)

find_x = SolveG.apply

if __name__ == "__main__":
    p0 = t([1., 0., 1.], requires_grad=True)
    x0 = t([3.5, 3.5**2])
    x_solution = find_x(p0, x0)
    assert torch.allclose(g(p0, x_solution), t([0., 0.]))

    from torch import norm

    def f(x): return norm(x)
    def fprime(x): return x / f(x)
    def grads(p, x):
        dgdp, dgdx = jacobians(p, x)
        fp = fprime(x)
        v = solve(dgdx.T, - fp)
        grad_p = v @ dgdp
        return grad_p

    expected_grad = grads(p0, x_solution)
    f(x_solution).backward()
    actual_grad = p0.grad
    assert torch.allclose(expected_grad, actual_grad)
