"""Level 4 tests: LRU eviction of zero-ref blocks."""

import importlib

IMPL_FILE = "exercise"  # switch to "exercise" to test your implementation

impl = importlib.import_module(IMPL_FILE)
EvictingPrefixAllocator = impl.EvictingPrefixAllocator
NoFreeBlocksError = impl.NoFreeBlocksError
InvalidBlockError = impl.InvalidBlockError


def test_zero_ref_block_stays_evictable_not_freed():
    """Free-to-zero must NOT immediately return the block — it lingers in the LRU set."""
    alloc = EvictingPrefixAllocator(num_blocks=2, block_size=16)
    b, _ = alloc.get_or_allocate(prefix_hash=10)
    alloc.free(b)
    assert alloc.get_ref_count(b) == 0
    # The same hash must still hit because the mapping is alive.
    b2, hit = alloc.get_or_allocate(prefix_hash=10)
    assert hit is True
    assert b2 == b
    assert alloc.get_ref_count(b) == 1


def test_cache_hit_on_evictable_resurrects():
    """After a zero-ref block is hit again, it's no longer evictable."""
    alloc = EvictingPrefixAllocator(num_blocks=1, block_size=16)
    b, _ = alloc.get_or_allocate(prefix_hash=1)
    alloc.free(b)  # evictable, but only block in pool
    assert len(alloc.evictable) == 1, alloc.evictable
    b2, hit = alloc.get_or_allocate(prefix_hash=1)
    assert hit is True
    assert len(alloc.evictable) == 0, alloc.evictable
    # Now allocating a NEW hash should fail — the resurrected block is in use,
    # and there's nothing evictable left.
    try:
        alloc.get_or_allocate(prefix_hash=2)
    except NoFreeBlocksError:
        return
    raise AssertionError(
        "expected NoFreeBlocksError after resurrection consumed the only block"
    )


def test_eviction_when_pool_full():
    """With pool exhausted, a new hash evicts the LRU evictable block."""
    alloc = EvictingPrefixAllocator(num_blocks=2, block_size=16)
    b1, _ = alloc.get_or_allocate(prefix_hash=1)
    b2, _ = alloc.get_or_allocate(prefix_hash=2)
    alloc.free(b1)  # evictable, no other free blocks
    assert len(alloc.evictable) == 1, alloc.evictable
    assert alloc._get_free_blk() is None, alloc._get_free_blk()

    b3, hit = alloc.get_or_allocate(prefix_hash=3)
    assert hit is False, "hit is True"
    assert len(alloc.evictable) == 0, alloc.evictable

    # b1 was the only evictable, so it must have been reused.
    assert b3 == b1, f"{b3, b1}"
    # Old hash mapping for prefix=1 should be gone. Free a slot first so
    # the re-query doesn't trip pool-exhaustion before the miss check.
    alloc.free(b3)
    b4, hit4 = alloc.get_or_allocate(prefix_hash=1)
    assert hit4 is False, "hit4 is True"


def test_no_evictables_raises():
    alloc = EvictingPrefixAllocator(num_blocks=2, block_size=16)
    alloc.get_or_allocate(prefix_hash=1)
    alloc.get_or_allocate(prefix_hash=2)
    try:
        alloc.get_or_allocate(prefix_hash=3)
    except NoFreeBlocksError:
        return
    raise AssertionError("expected NoFreeBlocksError when no evictables exist")


def test_lru_order_is_fifo_of_becoming_evictable():
    """Eviction order = order of becoming-evictable, NOT order of allocation."""
    alloc = EvictingPrefixAllocator(num_blocks=3, block_size=16)
    b1, _ = alloc.get_or_allocate(prefix_hash=1)
    b2, _ = alloc.get_or_allocate(prefix_hash=2)
    b3, _ = alloc.get_or_allocate(prefix_hash=3)
    # Free in reverse allocation order: b3, b1, b2.
    alloc.free(b3)  # becomes evictable first (oldest in evictable set)
    alloc.free(b1)
    alloc.free(b2)  # becomes evictable last (most recent)
    # Now request 3 fresh hashes; should evict in order b3, b1, b2.
    e1, _ = alloc.get_or_allocate(prefix_hash=10)
    e2, _ = alloc.get_or_allocate(prefix_hash=11)
    e3, _ = alloc.get_or_allocate(prefix_hash=12)
    assert e1 == b3, f"first eviction should be b3 (oldest evictable), got {e1}"
    assert e2 == b1, f"second eviction should be b1, got {e2}"
    assert e3 == b2, f"third eviction should be b2, got {e3}"


def test_evicted_block_drops_old_hash_mapping():
    alloc = EvictingPrefixAllocator(num_blocks=1, block_size=16)
    b, _ = alloc.get_or_allocate(prefix_hash=1)
    alloc.free(b)
    # New hash forces eviction.
    b2, hit2 = alloc.get_or_allocate(prefix_hash=2)
    assert hit2 is False
    assert b2 == b
    # Old hash should miss now.
    alloc.free(b2)
    b3, hit3 = alloc.get_or_allocate(prefix_hash=1)
    assert hit3 is False


def test_resurrected_block_is_not_evicted():
    """A resurrected block must not be in the evictable set anymore."""
    alloc = EvictingPrefixAllocator(num_blocks=2, block_size=16)
    b1, _ = alloc.get_or_allocate(prefix_hash=1)
    b2, _ = alloc.get_or_allocate(prefix_hash=2)
    alloc.free(b1)  # b1 evictable
    # Resurrect b1 via a hit — it should leave the evictable set.
    b1_again, hit = alloc.get_or_allocate(prefix_hash=1)
    assert hit is True and b1_again == b1
    # Now ask for a new hash. Pool is full (b1 and b2 both in use, no
    # evictables) -> must raise.
    try:
        alloc.get_or_allocate(prefix_hash=3)
    except NoFreeBlocksError:
        return
    raise AssertionError("resurrected block must not be evicted")


def test_num_free_blocks_counts_evictables():
    alloc = EvictingPrefixAllocator(num_blocks=3, block_size=16)
    b1, _ = alloc.get_or_allocate(prefix_hash=1)
    b2, _ = alloc.get_or_allocate(prefix_hash=2)
    assert alloc.num_free_blocks() == 1, (
        alloc.num_free_blocks()
    )  # one block left in raw pool
    alloc.free(b1)
    # Now: 1 raw + 1 evictable = 2 satisfiable allocations.
    assert alloc.num_free_blocks() == 2, alloc.num_free_blocks()


TESTS = [
    test_zero_ref_block_stays_evictable_not_freed,
    test_cache_hit_on_evictable_resurrects,
    test_eviction_when_pool_full,
    test_no_evictables_raises,
    test_lru_order_is_fifo_of_becoming_evictable,
    test_evicted_block_drops_old_hash_mapping,
    test_resurrected_block_is_not_evicted,
    test_num_free_blocks_counts_evictables,
]


if __name__ == "__main__":
    print(f"running test_eviction.py against IMPL_FILE = {IMPL_FILE!r}")
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"✅ {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"❌ {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
