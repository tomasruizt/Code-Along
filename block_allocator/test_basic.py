"""Level 1 tests: basic allocate/free with a free-pool stack."""

import importlib

IMPL_FILE = "exercise"  # switch to "exercise" to test your implementation

impl = importlib.import_module(IMPL_FILE)
BlockAllocator = impl.BlockAllocator
NoFreeBlocksError = impl.NoFreeBlocksError
InvalidBlockError = impl.InvalidBlockError


def test_basic_allocate_free():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    assert alloc.num_free_blocks() == 4

    b1 = alloc.allocate()
    b2 = alloc.allocate()
    assert alloc.num_free_blocks() == 2
    assert b1 != b2

    alloc.free(b1)
    assert alloc.num_free_blocks() == 3

    # LIFO: the recently freed block is reused first.
    b3 = alloc.allocate()
    assert b3 == b1, f"expected reuse of {b1}, got {b3}"


def test_allocate_until_exhausted():
    alloc = BlockAllocator(num_blocks=3, block_size=16)
    ids = [alloc.allocate() for _ in range(3)]
    assert sorted(ids) == [0, 1, 2]
    assert alloc.num_free_blocks() == 0
    try:
        alloc.allocate()
    except NoFreeBlocksError:
        return
    raise AssertionError("expected NoFreeBlocksError on 4th allocation")


def test_double_free_raises():
    alloc = BlockAllocator(num_blocks=2, block_size=16)
    b = alloc.allocate()
    alloc.free(b)
    try:
        alloc.free(b)
    except InvalidBlockError:
        return
    raise AssertionError("expected InvalidBlockError on double free")


def test_free_out_of_range_raises():
    alloc = BlockAllocator(num_blocks=2, block_size=16)
    for bad_id in (-1, 2, 100):
        try:
            alloc.free(bad_id)
        except InvalidBlockError:
            continue
        raise AssertionError(f"expected InvalidBlockError for id={bad_id}")


def test_free_never_allocated_raises():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    # block_id 0 is in range but was never allocated.
    try:
        alloc.free(0)
    except InvalidBlockError:
        return
    raise AssertionError("expected InvalidBlockError when freeing un-allocated block")


def test_free_in_different_order():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    ids = [alloc.allocate() for _ in range(4)]
    # Free in scrambled order.
    for bid in (ids[2], ids[0], ids[3], ids[1]):
        alloc.free(bid)
    assert alloc.num_free_blocks() == 4
    # All blocks should be reallocatable.
    new_ids = sorted(alloc.allocate() for _ in range(4))
    assert new_ids == sorted(ids)


def test_allocate_free_allocate_reuses():
    alloc = BlockAllocator(num_blocks=2, block_size=16)
    a = alloc.allocate()
    alloc.free(a)
    b = alloc.allocate()
    assert a == b


TESTS = [
    test_basic_allocate_free,
    test_allocate_until_exhausted,
    test_double_free_raises,
    test_free_out_of_range_raises,
    test_free_never_allocated_raises,
    test_free_in_different_order,
    test_allocate_free_allocate_reuses,
]


if __name__ == "__main__":
    print(f"running test_basic.py against IMPL_FILE = {IMPL_FILE!r}")
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"✅ {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"❌ {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
