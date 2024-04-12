import torch

class Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _norm(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * x / _norm(x)
    
def _norm(xs):
    return torch.sum(xs**2)**.5

norm = Norm.apply

if __name__ == "__main__":
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    loss = norm(x)
    expected_loss = torch.sqrt(torch.sum(x**2))
    assert torch.allclose(loss, expected_loss)

    loss.backward()
    expected_grad = x / norm(x)
    assert torch.allclose(x.grad, expected_grad)
