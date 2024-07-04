from ast import Not
import torch
from torch import sin, cos, pi, allclose


class F:
    def __init__(self, f, requires_grad: bool = False):
        self.f = f
        self.requires_grad = requires_grad

    def __mul__(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        return F(lambda x: self.f(x) * other.f(x), requires_grad=requires_grad)
    
    def __rmul__(self, other):
        return self * other

    def __call__(self, x):
        return self.f(x)
    
    def __pow__(self, n):
        return F(lambda x: self.f(x) ** n, requires_grad=self.requires_grad)
    
    def integrate(self, a, b):
        x = torch.linspace(a, b, 500, requires_grad=self.requires_grad)
        return torch.trapezoid(self.f(x), x)
    
    def grad(self):
        raise NotImplementedError("How???")


t = torch.tensor

f1 = F(sin)
f2 = F(cos)
f3 = f1 * f2


def test_composition():
    num = torch.tensor([1.1, 1.2], requires_grad=True)
    expected = sin(num) * cos(num)
    assert allclose(f3(num), expected)

    expected.sum().backward()
    expected_grad = num.grad.clone()
    num.grad.zero_()
    f3(num).sum().backward()
    actual_grad = num.grad.clone()
    assert allclose(expected_grad, actual_grad)


def test_integration():
    f = F(torch.sin)
    # integral of sin(x) is -cos(x), range is 0 to pi
    expected_integral = -cos(t(pi)) - (-cos(t(0)))
    actual = f.integrate(0, pi)
    assert allclose(expected_integral, actual)

# A(f) = int f^2(x) dx
# dA(f) = A(f + df) - A(f)
#       = int (f + df)^2(x) - f^2(x) dx
#       = int f^2(x) + 2f(x)df(x) + df^2(x) - f^2(x) dx
#       = int 2f(x)df(x) dx
#  => 2f(x) is the gradient of A(f)
def test_integration_grad():
    def A(f):
        return (f ** 2).integrate(0, pi)
    
    f = F(sin, requires_grad=True)
    expected_grad = 2 * f
    A(f).backward()
    assert allclose(expected_grad, f.grad)
    

