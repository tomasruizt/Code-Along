import torch
from solve_mnist_lib import get_train_conf, train_model


conf = get_train_conf("cifar100")
with torch.profiler.profile(
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    train_model(seed=0, conf=conf, max_n_steps=5)

name = "cifar100"
prof.export_chrome_trace(f"profiles/trace-{name}.json.gz")
prof.export_memory_timeline(f"profiles/memory-{name}.html")
