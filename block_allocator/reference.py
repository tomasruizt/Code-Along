"""Reference implementation for the paged KV-cache block allocator drill.

Peek here only when stuck for more than ~15 min on a level. Each class is
self-contained and ~20-40 lines.

Implementation notes (apply to all levels):

- The free pool is a Python list used as a stack: pop() / append() are O(1)
  and give LIFO behavior, so recently-freed blocks are reused first. That's
  cache-friendly for the underlying GPU memory pages.
- Levels 2-4 keep a parallel `ref_counts` array of length num_blocks.
  allocate() sets ref_count = 1; increment_ref adds 1; free decrements and
  only returns the block to the pool (or, in L4, to the evictable set) on
  zero.
- Level 3 adds a `dict[hash -> block_id]` for content-addressed sharing,
  plus the inverse `dict[block_id -> hash]` so we can clean up on free.
  Hash collisions are NOT handled — we trust the hash.
- Level 4 replaces "free immediately on zero-ref" with "park in an
  OrderedDict keyed by block_id". Insertion order = recency of becoming
  evictable, so popitem(last=False) gives the LRU evictable block. A cache
  hit on an evictable block resurrects it (remove from the OrderedDict,
  bump ref_count to 1).
"""

from collections import OrderedDict


class NoFreeBlocksError(Exception):
    """Raised when allocation cannot be satisfied."""


class InvalidBlockError(Exception):
    """Raised on misuse: double-free, out-of-range id, sharing a freed block."""


# ---------------------------------------------------------------------------
# Level 1 — basic allocator
# ---------------------------------------------------------------------------


class BlockAllocator:
    """Stack-based free pool. No sharing, no eviction."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Initialize descending so pop() returns 0, 1, 2, ... in allocation order.
        self.free_list: list[int] = list(range(num_blocks - 1, -1, -1))
        self.allocated: list[bool] = [False] * num_blocks

    def allocate(self) -> int:
        if not self.free_list:
            raise NoFreeBlocksError("block pool exhausted")
        bid = self.free_list.pop()
        self.allocated[bid] = True
        return bid

    def free(self, block_id: int) -> None:
        if not 0 <= block_id < self.num_blocks:
            raise InvalidBlockError(f"block_id {block_id} out of range")
        if not self.allocated[block_id]:
            raise InvalidBlockError(f"block {block_id} is not allocated")
        self.allocated[block_id] = False
        self.free_list.append(block_id)

    def num_free_blocks(self) -> int:
        return len(self.free_list)


# ---------------------------------------------------------------------------
# Level 2 — reference counting
# ---------------------------------------------------------------------------


class RefCountedBlockAllocator:
    """Multiple sequences can share a block via increment_ref / free."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_list: list[int] = list(range(num_blocks - 1, -1, -1))
        self.ref_counts: list[int] = [0] * num_blocks

    def allocate(self) -> int:
        if not self.free_list:
            raise NoFreeBlocksError("block pool exhausted")
        bid = self.free_list.pop()
        self.ref_counts[bid] = 1
        return bid

    def increment_ref(self, block_id: int) -> None:
        self._check_id(block_id)
        if self.ref_counts[block_id] == 0:
            raise InvalidBlockError(f"cannot share freed block {block_id}")
        self.ref_counts[block_id] += 1

    def free(self, block_id: int) -> None:
        self._check_id(block_id)
        if self.ref_counts[block_id] == 0:
            raise InvalidBlockError(f"block {block_id} already freed")
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_list.append(block_id)

    def get_ref_count(self, block_id: int) -> int:
        self._check_id(block_id)
        return self.ref_counts[block_id]

    def num_free_blocks(self) -> int:
        return len(self.free_list)

    def _check_id(self, block_id: int) -> None:
        if not 0 <= block_id < self.num_blocks:
            raise InvalidBlockError(f"block_id {block_id} out of range")


# ---------------------------------------------------------------------------
# Level 3 — hash-based prefix caching
# ---------------------------------------------------------------------------


class PrefixCachingAllocator:
    """Refcounting + a hash -> block_id map for automatic prefix sharing."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_list: list[int] = list(range(num_blocks - 1, -1, -1))
        self.ref_counts: list[int] = [0] * num_blocks
        self.hash_to_block: dict[int, int] = {}
        self.block_to_hash: dict[int, int] = {}

    def get_or_allocate(self, prefix_hash: int) -> tuple[int, bool]:
        if prefix_hash in self.hash_to_block:
            bid = self.hash_to_block[prefix_hash]
            self.ref_counts[bid] += 1
            return bid, True
        if not self.free_list:
            raise NoFreeBlocksError("block pool exhausted")
        bid = self.free_list.pop()
        self.ref_counts[bid] = 1
        self.hash_to_block[prefix_hash] = bid
        self.block_to_hash[bid] = prefix_hash
        return bid, False

    def free(self, block_id: int) -> None:
        if not 0 <= block_id < self.num_blocks:
            raise InvalidBlockError(f"block_id {block_id} out of range")
        if self.ref_counts[block_id] == 0:
            raise InvalidBlockError(f"block {block_id} already freed")
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            h = self.block_to_hash.pop(block_id)
            del self.hash_to_block[h]
            self.free_list.append(block_id)

    def get_ref_count(self, block_id: int) -> int:
        return self.ref_counts[block_id]

    def num_free_blocks(self) -> int:
        return len(self.free_list)


# ---------------------------------------------------------------------------
# Level 4 — LRU eviction of zero-ref blocks
# ---------------------------------------------------------------------------


class EvictingPrefixAllocator:
    """Zero-ref blocks linger in an LRU set; evicted only when pool is exhausted."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_list: list[int] = list(range(num_blocks - 1, -1, -1))
        self.ref_counts: list[int] = [0] * num_blocks
        self.hash_to_block: dict[int, int] = {}
        self.block_to_hash: dict[int, int] = {}
        # Insertion order = recency of becoming evictable.
        # popitem(last=False) -> the LRU candidate.
        self.evictable: OrderedDict[int, None] = OrderedDict()

    def get_or_allocate(self, prefix_hash: int) -> tuple[int, bool]:
        if prefix_hash in self.hash_to_block:
            bid = self.hash_to_block[prefix_hash]
            if bid in self.evictable:
                # Resurrect: was sitting in the LRU set, now in use again.
                del self.evictable[bid]
            self.ref_counts[bid] += 1
            return bid, True
        bid = self._acquire_block()
        self.ref_counts[bid] = 1
        self.hash_to_block[prefix_hash] = bid
        self.block_to_hash[bid] = prefix_hash
        return bid, False

    def free(self, block_id: int) -> None:
        if not 0 <= block_id < self.num_blocks:
            raise InvalidBlockError(f"block_id {block_id} out of range")
        if self.ref_counts[block_id] == 0:
            raise InvalidBlockError(f"block {block_id} already freed")
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            # Park it; its hash mapping stays alive so future hits can resurrect it.
            self.evictable[block_id] = None

    def get_ref_count(self, block_id: int) -> int:
        return self.ref_counts[block_id]

    def num_free_blocks(self) -> int:
        return len(self.free_list) + len(self.evictable)

    def _acquire_block(self) -> int:
        if self.free_list:
            return self.free_list.pop()
        if self.evictable:
            bid, _ = self.evictable.popitem(last=False)  # LRU
            h = self.block_to_hash.pop(bid)
            del self.hash_to_block[h]
            return bid
        raise NoFreeBlocksError("block pool exhausted; no evictable blocks")
