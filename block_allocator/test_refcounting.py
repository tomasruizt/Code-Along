"""Level 2 tests: shared blocks via reference counting."""

import importlib

IMPL_FILE = "exercise"  # switch to "exercise" to test your implementation

impl = importlib.import_module(IMPL_FILE)
RefCountedBlockAllocator = impl.RefCountedBlockAllocator
NoFreeBlocksError = impl.NoFreeBlocksError
InvalidBlockError = impl.InvalidBlockError


def test_initial_ref_count_is_one():
    alloc = RefCountedBlockAllocator(num_blocks=4, block_size=16)
    b = alloc.allocate()
    assert alloc.get_ref_count(b) == 1


def test_increment_ref_increases_count():
    alloc = RefCountedBlockAllocator(num_blocks=4, block_size=16)
    b = alloc.allocate()
    alloc.increment_ref(b)
    alloc.increment_ref(b)
    assert alloc.get_ref_count(b) == 3


def test_shared_block_one_frees():
    """Two sequences share a block; one frees, the other still has it."""
    alloc = RefCountedBlockAllocator(num_blocks=2, block_size=16)
    b = alloc.allocate()        # ref=1 (seq A)
    alloc.increment_ref(b)      # ref=2 (seq B joins)
    alloc.free(b)               # ref=1 (seq A leaves)
    assert alloc.get_ref_count(b) == 1
    assert alloc.num_free_blocks() == 1, "block must NOT be in pool yet"
    alloc.free(b)               # ref=0 (seq B leaves) -> released
    assert alloc.num_free_blocks() == 2


def test_free_too_many_times_raises():
    alloc = RefCountedBlockAllocator(num_blocks=2, block_size=16)
    b = alloc.allocate()
    alloc.free(b)
    try:
        alloc.free(b)
    except InvalidBlockError:
        return
    raise AssertionError("expected InvalidBlockError on extra free")


def test_increment_ref_on_freed_block_raises():
    alloc = RefCountedBlockAllocator(num_blocks=2, block_size=16)
    b = alloc.allocate()
    alloc.free(b)  # ref_count now 0
    try:
        alloc.increment_ref(b)
    except InvalidBlockError:
        return
    raise AssertionError("expected InvalidBlockError when sharing a freed block")


def test_block_returns_to_pool_after_full_free():
    alloc = RefCountedBlockAllocator(num_blocks=1, block_size=16)
    b = alloc.allocate()
    alloc.increment_ref(b)
    alloc.free(b)
    # Pool still has 0 free blocks (ref_count == 1).
    try:
        alloc.allocate()
    except NoFreeBlocksError:
        pass
    else:
        raise AssertionError("pool should still be exhausted")
    alloc.free(b)
    assert alloc.num_free_blocks() == 1
    b2 = alloc.allocate()
    assert b2 == b  # only block in the system


def test_exhaustion_raises():
    alloc = RefCountedBlockAllocator(num_blocks=2, block_size=16)
    alloc.allocate()
    alloc.allocate()
    try:
        alloc.allocate()
    except NoFreeBlocksError:
        return
    raise AssertionError("expected NoFreeBlocksError")


def test_out_of_range_raises():
    alloc = RefCountedBlockAllocator(num_blocks=2, block_size=16)
    for bad in (-1, 2):
        try:
            alloc.free(bad)
        except InvalidBlockError:
            continue
        raise AssertionError(f"expected InvalidBlockError for {bad}")


TESTS = [
    test_initial_ref_count_is_one,
    test_increment_ref_increases_count,
    test_shared_block_one_frees,
    test_free_too_many_times_raises,
    test_increment_ref_on_freed_block_raises,
    test_block_returns_to_pool_after_full_free,
    test_exhaustion_raises,
    test_out_of_range_raises,
]


if __name__ == "__main__":
    print(f"running test_refcounting.py against IMPL_FILE = {IMPL_FILE!r}")
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"✅ {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"❌ {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
