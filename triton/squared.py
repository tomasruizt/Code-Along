import torch


def my_op(a):
    a = torch.square(a)
    a = torch.square(a)
    return a


torch._logging.set_logs(output_code=True)
my_op_fast = torch.compile(my_op)
my_op_fast(torch.randn(1_000, 1_000).cuda())
