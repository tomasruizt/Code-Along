# Profiling Results
Summarizing the results from `profile.txt` (created by running `make profile`):

| Metric | Naive (Sum) | Contiguous (Sum) | SharedMem (Sum) | SharedMem (Max) |
|--------|-------------|------------------|-----------------|-----------------|
| Memory Throughput | 33.95% | 24.59% | 36.61% | 30.10% |
| Compute Throughput | 35.62% | 22.93% | 36.61% | 30.08% |
| Duration (μs) | 8.00 | 4.99 | 5.54 | 6.24 |
| Achieved Occupancy | 81.01% | 85.11% | 79.58% | 70.47% |
| Warp Cycles/Instruction | 21.52 | 31.03 | 32.73 | 33.06 |
| L1/TEX Cache Throughput | 41.00% | 19.32% | 55.95% | 47.25% |
| Memory Coalescing Issues | 66% excessive | 0.57% excessive | None reported | None reported |
| L2 Cache Hit Rate | 80.39% | 53.51% | 10.93% | 16.43% |
| Active Warps/SM | 38.89 | 40.85 | 38.20 | 33.83 |

This table makes it easier to see that:

* The Contiguous kernel has the fastest execution time despite lower throughput
* SharedMem variants have better L1 cache utilization but worse L2 cache hit rates
* Memory coalescing dramatically improved after the Naive implementation
* Occupancy gradually decreases with each implementation variant
