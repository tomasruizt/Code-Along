import torch
from solve_mnist_lib import train_and_save

with torch.profiler.profile(record_shapes=True, profile_memory=True) as prof:
    train_and_save(seed=0, dataset="mnist")

name = "nonblocking-send"
prof.export_chrome_trace(f"profiles/trace-{name}.json.gz")
#prof.export_memory_timeline(f"profiles/memory-{name}.html")
