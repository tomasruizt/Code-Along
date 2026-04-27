"""Level 3 tests: hash-based prefix sharing via get_or_allocate."""

import importlib

IMPL_FILE = "exercise"  # switch to "exercise" to test your implementation

impl = importlib.import_module(IMPL_FILE)
PrefixCachingAllocator = impl.PrefixCachingAllocator
NoFreeBlocksError = impl.NoFreeBlocksError
InvalidBlockError = impl.InvalidBlockError


def test_first_call_is_a_miss():
    alloc = PrefixCachingAllocator(num_blocks=4, block_size=16)
    bid, hit = alloc.get_or_allocate(prefix_hash=0xDEADBEEF)
    assert hit is False
    assert alloc.get_ref_count(bid) == 1


def test_second_call_same_hash_is_a_hit():
    alloc = PrefixCachingAllocator(num_blocks=4, block_size=16)
    b1, hit1 = alloc.get_or_allocate(prefix_hash=42)
    b2, hit2 = alloc.get_or_allocate(prefix_hash=42)
    assert hit1 is False
    assert hit2 is True
    assert b1 == b2
    assert alloc.get_ref_count(b1) == 2


def test_different_hashes_get_different_blocks():
    alloc = PrefixCachingAllocator(num_blocks=4, block_size=16)
    b1, _ = alloc.get_or_allocate(prefix_hash=1)
    b2, _ = alloc.get_or_allocate(prefix_hash=2)
    assert b1 != b2


def test_free_to_zero_removes_hash_mapping():
    alloc = PrefixCachingAllocator(num_blocks=2, block_size=16)
    b1, _ = alloc.get_or_allocate(prefix_hash=7)
    alloc.free(b1)
    # Same hash again must be a miss because the mapping was cleared.
    b2, hit = alloc.get_or_allocate(prefix_hash=7)
    assert hit is False
    # Reused via the LIFO free pool:
    assert b2 == b1


def test_partial_free_keeps_hash_mapping():
    alloc = PrefixCachingAllocator(num_blocks=2, block_size=16)
    b1, _ = alloc.get_or_allocate(prefix_hash=7)
    b2, hit = alloc.get_or_allocate(prefix_hash=7)
    assert hit is True
    alloc.free(b1)  # ref_count 2 -> 1, mapping should remain
    b3, hit3 = alloc.get_or_allocate(prefix_hash=7)
    assert hit3 is True
    assert b3 == b1


def test_pool_exhaustion_raises():
    alloc = PrefixCachingAllocator(num_blocks=2, block_size=16)
    alloc.get_or_allocate(prefix_hash=1)
    alloc.get_or_allocate(prefix_hash=2)
    try:
        alloc.get_or_allocate(prefix_hash=3)
    except NoFreeBlocksError:
        return
    raise AssertionError("expected NoFreeBlocksError when pool is exhausted")


def test_double_free_raises():
    alloc = PrefixCachingAllocator(num_blocks=2, block_size=16)
    b, _ = alloc.get_or_allocate(prefix_hash=1)
    alloc.free(b)
    try:
        alloc.free(b)
    except InvalidBlockError:
        return
    raise AssertionError("expected InvalidBlockError on double free")


def test_three_way_share():
    alloc = PrefixCachingAllocator(num_blocks=4, block_size=16)
    b1, _ = alloc.get_or_allocate(prefix_hash=99)
    b2, _ = alloc.get_or_allocate(prefix_hash=99)
    b3, _ = alloc.get_or_allocate(prefix_hash=99)
    assert b1 == b2 == b3
    assert alloc.get_ref_count(b1) == 3
    assert alloc.num_free_blocks() == 3  # only 1 block consumed


TESTS = [
    test_first_call_is_a_miss,
    test_second_call_same_hash_is_a_hit,
    test_different_hashes_get_different_blocks,
    test_free_to_zero_removes_hash_mapping,
    test_partial_free_keeps_hash_mapping,
    test_pool_exhaustion_raises,
    test_double_free_raises,
    test_three_way_share,
]


if __name__ == "__main__":
    print(f"running test_prefix_caching.py against IMPL_FILE = {IMPL_FILE!r}")
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"✅ {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"❌ {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
