import pytest


class CircularBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._memory = [None] * (capacity + 1)
        self.start = 0
        self.end = 0

    def __len__(self):
        len_ = self.end - self.start
        if len_ < 0:
            len_ += self.capacity + 1
        return len_

    def append(self, item: float):
        if len(self) == self.capacity:
            raise OverflowError("Buffer is full")

        self._memory[self.end] = item
        self.end = (self.end + 1) % (self.capacity + 1)

    def tolist(self):
        if self.start <= self.end:
            return self._memory[self.start : self.end]
        else:
            return self._memory[self.start :] + self._memory[: self.end]

    def rmhead(self, n: int):
        if n > len(self):
            raise IndexError("Not enough elements to remove")
        self.start = (self.start + n) % (self.capacity + 1)


def test_capacity1():
    buffer = CircularBuffer(capacity=1)
    assert len(buffer) == 0
    buffer.append(1)
    assert len(buffer) == 1
    assert buffer.tolist() == [1]


def test_append():
    buffer = CircularBuffer(capacity=3)
    assert len(buffer) == 0

    buffer.append(1)
    buffer.append(10)
    assert len(buffer) == 2
    assert buffer.tolist() == [1, 10]


def test_overflow():
    buffer = CircularBuffer(capacity=1)
    buffer.append(1)
    with pytest.raises(OverflowError):
        buffer.append(1)


def test_rmhead():
    buffer = CircularBuffer(capacity=3)
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)
    buffer.rmhead(n=1)
    assert buffer.tolist() == [2, 3]

    with pytest.raises(IndexError):
        buffer.rmhead(n=3)

    assert len(buffer) == 2
    buffer.rmhead(n=2)
    assert buffer.tolist() == []

    with pytest.raises(IndexError):
        buffer.rmhead(n=1)


def test_wrap_around():
    buffer = CircularBuffer(capacity=1)
    buffer.append(1)
    buffer.rmhead(n=1)
    assert len(buffer) == 0

    buffer.append(2)
    assert len(buffer) == 1
    assert buffer.tolist() == [2]


def test_wrap_around2():
    buffer = CircularBuffer(capacity=2)
    buffer.append(1)
    buffer.append(2)
    assert buffer.tolist() == [1, 2]
    buffer.rmhead(n=2)
    assert buffer.tolist() == []
    buffer.append(3)
    buffer.append(4)
    assert buffer.tolist() == [3, 4]
    buffer.rmhead(n=2)
    assert buffer.tolist() == []
