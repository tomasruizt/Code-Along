"""Block allocator exercise — fill in the stubs below.

Suggested order: BlockAllocator -> RefCountedBlockAllocator ->
PrefixCachingAllocator -> EvictingPrefixAllocator. Each level builds on the
previous one's data structures.

To run a level's tests against your implementation, open the matching
test_*.py and switch IMPL_FILE = "exercise" at the top.
"""

from queue import Queue


class NoFreeBlocksError(Exception):
    """Raised when allocation cannot be satisfied."""


class InvalidBlockError(Exception):
    """Raised on misuse: double-free, out-of-range id, sharing a freed block."""


# ---------------------------------------------------------------------------
# Level 1 — basic allocator (validated by test_basic.py)
# ---------------------------------------------------------------------------


class BlockAllocator:
    """Level 1: basic allocator. See test_basic.py."""

    def __init__(self, num_blocks: int, block_size: int):
        """num_blocks: total physical blocks. block_size: tokens per block (informational)."""
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blks = list(range(num_blocks))

    def allocate(self) -> int:
        """Return a free block_id. Raise NoFreeBlocksError if pool is empty."""
        if len(self.free_blks) == 0:
            raise NoFreeBlocksError()
        return self.free_blks.pop()

    def free(self, block_id: int) -> None:
        """Return block to free pool. Raise InvalidBlockError on double-free
        or out-of-range id."""
        if block_id in self.free_blks:  # O(N) search
            raise InvalidBlockError()
        if block_id >= self.num_blocks or block_id < 0:
            raise InvalidBlockError()
        self.free_blks.append(block_id)

    def num_free_blocks(self) -> int:
        return len(self.free_blks)


# ---------------------------------------------------------------------------
# Level 2 — reference counting (validated by test_refcounting.py)
# ---------------------------------------------------------------------------


class RefCountedBlockAllocator:
    """Level 2: reference-counted allocator. See test_refcounting.py."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.block_allocs = {b: 0 for b in range(num_blocks)}

    def allocate(self) -> int:
        """Allocate a block with ref_count = 1."""
        free_blk = None
        for blk, n in self.block_allocs.items():
            if n == 0:
                free_blk = blk
        if free_blk is None:
            raise NoFreeBlocksError()
        self.block_allocs[free_blk] += 1
        return free_blk

    def increment_ref(self, block_id: int) -> None:
        """Increment ref_count. Raise InvalidBlockError if the block is freed
        (ref_count == 0) or out of range."""
        val = self.get_ref_count(block_id)
        if val <= 0:
            raise InvalidBlockError()
        self.block_allocs[block_id] = val + 1

    def free(self, block_id: int) -> None:
        """Decrement ref_count; return block to free pool when it hits 0.
        Raise InvalidBlockError if already freed or out of range."""
        val = self.get_ref_count(block_id)
        if val <= 0:
            raise InvalidBlockError()
        self.block_allocs[block_id] = val - 1

    def get_ref_count(self, block_id: int) -> int:
        if block_id not in self.block_allocs:  # O(1) lookup
            raise InvalidBlockError()
        return self.block_allocs[block_id]

    def num_free_blocks(self) -> int:
        num_free = 0
        for n in self.block_allocs.values():
            if n == 0:
                num_free += 1
        return num_free


# ---------------------------------------------------------------------------
# Level 3 — hash-based prefix caching (validated by test_prefix_caching.py)
# ---------------------------------------------------------------------------


class PrefixCachingAllocator:
    """Level 3: prefix-caching allocator. See test_prefix_caching.py."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blk_count = {b: 0 for b in range(num_blocks)}
        self.hash2block: dict[int, int] = {}

    def get_or_allocate(self, prefix_hash: int) -> tuple[int, bool]:
        """If a block exists for prefix_hash, increment its ref_count and
        return (block_id, True). Otherwise allocate a new block, register
        it under prefix_hash, and return (block_id, False).

        Hash collisions are NOT handled — we trust the hash.
        """
        hit = prefix_hash in self.hash2block
        if hit:
            blk = self.hash2block[prefix_hash]
        else:
            blk = self._get_free_blk_or_raise()
            self.hash2block[prefix_hash] = blk

        self.blk_count[blk] += 1
        return blk, hit

    def _get_free_blk_or_raise(self) -> int:
        free_blk = None
        for blk, n in self.blk_count.items():
            if n == 0:
                free_blk = blk
        if free_blk is None:
            raise NoFreeBlocksError()
        return free_blk

    def free(self, block_id: int) -> None:
        """Decrement ref_count. When it reaches 0, also remove the
        hash -> block_id mapping and return the block to the pool."""
        val = self.get_ref_count(block_id)
        if val <= 0:
            raise InvalidBlockError()
        if val - 1 <= 0:
            hash_ = self._get_hash(block_id)
            del self.hash2block[hash_]
        self.blk_count[block_id] = val - 1

    def _get_hash(self, block_id: int) -> int:
        for hash_, blk in self.hash2block.items():
            if blk == block_id:
                return hash_
        raise ValueError(f"Hash for {block_id} not found in {self.hash2block}")

    def get_ref_count(self, block_id: int) -> int:
        if block_id not in self.blk_count:  # O(1) lookup
            raise InvalidBlockError()
        return self.blk_count[block_id]

    def num_free_blocks(self) -> int:
        num_free = 0
        for n in self.blk_count.values():
            if n == 0:
                num_free += 1
        return num_free


# ---------------------------------------------------------------------------
# Level 4 — LRU eviction (validated by test_eviction.py)
# ---------------------------------------------------------------------------


class EvictingPrefixAllocator:
    """Level 4: prefix-caching allocator with LRU eviction.

    Key shift from L3: blocks with ref_count == 0 are NOT immediately
    returned to the free pool. They sit in an "evictable" LRU set where
    they could be resurrected via a prefix-cache hit, but get evicted if
    the pool is otherwise exhausted.

    See test_eviction.py.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blk_count = {b: 0 for b in range(num_blocks)}
        self.hash2block: dict[int, int] = {}

        # ordered list of blocks
        self.evictable: list[int] = []
        # tuples are (block_num, hash)
        self.evictable_hash: dict[int, int] = {}

    def get_or_allocate(self, prefix_hash: int) -> tuple[int, bool]:
        """As L3, but if the pool is exhausted, evict the LRU evictable
        block first. A cache hit on an evictable block resurrects it
        (remove from the evictable set, ref_count goes 0 -> 1)."""
        hit = prefix_hash in self.hash2block
        if hit:
            blk = self.hash2block[prefix_hash]
            if blk in self.evictable:
                idx = self.evictable.index(blk)
                self.evictable.pop(idx)
                del self.hash2block[self.evictable_hash[blk]]
                del self.evictable_hash[blk]
        else:
            blk: int | None = self._get_free_blk()
            if blk is None:
                if len(self.evictable) == 0:
                    raise NoFreeBlocksError()
                else:  # draw from evictable
                    blk, *self.evictable = self.evictable
                    del self.hash2block[self.evictable_hash[blk]]
                    del self.evictable_hash[blk]
            self.hash2block[prefix_hash] = blk

        self.blk_count[blk] += 1
        return blk, hit

    def _get_hash(self, block_id: int) -> int | None:
        for hash_, blk in self.hash2block.items():
            if blk == block_id:
                return hash_

    def _get_free_blk(self) -> int | None:
        for blk, n in self.blk_count.items():
            if n == 0 and blk not in self.evictable:
                return blk
        return None

    def free(self, block_id: int) -> None:
        """Decrement ref_count. If 0, move the block to the evictable LRU
        set; do NOT free it yet."""
        val = self.get_ref_count(block_id)
        if val <= 0:
            raise InvalidBlockError()
        new_val = val - 1
        if new_val <= 0:
            hash_ = self._get_hash(block_id)
            self.evictable.append(block_id)
            self.evictable_hash[block_id] = hash_
        self.blk_count[block_id] = new_val

    def get_ref_count(self, block_id: int) -> int:
        if block_id not in self.blk_count:  # O(1) lookup
            raise InvalidBlockError()
        return self.blk_count[block_id]

    def num_free_blocks(self) -> int:
        """Free pool size + evictable set size (both can satisfy alloc)."""
        num_free = 0
        for n in self.blk_count.values():
            if n == 0:
                num_free += 1
        return num_free
