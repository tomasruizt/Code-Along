# Paged KV Cache Block Allocator — Interview Drill

A staged Python implementation of the data structure that underlies vLLM's
PagedAttention block manager. Four levels, each adding one feature.

## Suggested order

1. **L1 — `BlockAllocator`** (target: 15 min). Stack-based free pool. Validates the basic alloc/free protocol.
2. **L2 — `RefCountedBlockAllocator`** (target: 20 min). Reference counting so multiple sequences can share a block.
3. **L3 — `PrefixCachingAllocator`** (target: 25 min). `dict[hash → block_id]` for automatic prefix sharing on top of L2.
4. **L4 — `EvictingPrefixAllocator`** (target: 30 min). LRU eviction of zero-ref blocks (kept around for potential reuse until the pool is exhausted).

If you blow past a target by ~50%, peek at `reference.py`.

## Workflow

1. Open `exercise.py` and start filling in Level 1.
2. Run the matching test file. Each test file has `IMPL_FILE = "reference"` at the top — switch to `IMPL_FILE = "exercise"` to point the tests at your implementation.

```bash
python test_basic.py
python test_refcounting.py
python test_prefix_caching.py
python test_eviction.py
```

You can also run the tests against `reference.py` first (the default) to confirm the test infrastructure works.

## What each level teaches

- **L1**: free-list discipline; LIFO for cache-friendly reuse; clean error semantics for double-free / out-of-range.
- **L2**: refcount-as-lifecycle; only the *last* free returns the block to the pool. `increment_ref` on a freed block is a bug, not a no-op.
- **L3**: content-addressed sharing as a thin layer over refcounting. The cache is just `hash → block_id`; freeing to zero also clears the hash entry.
- **L4**: deferred reclamation. Zero-ref blocks linger in an LRU set so they can still be hit, until the pool is exhausted and the LRU one gets evicted.

## Simplifications vs. real vLLM

This is a model, not the real thing. Real vLLM also handles:

- Cross-block prefix matching (we hash whole blocks only — block-aligned prefixes).
- Swapping evicted blocks to CPU memory instead of dropping them.
- Copy-on-write when forking a sequence mid-block (e.g., beam search).

Skipping these keeps the data structures small enough to whiteboard.
