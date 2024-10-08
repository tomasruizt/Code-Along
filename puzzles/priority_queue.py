from dataclasses import dataclass
from heapq import heappop, heappush


@dataclass
class Entry:
    key: int
    value: str

    def __lt__(self, other):
        return self.key < other.key


els: list[Entry] = []
for entry in [Entry(3, "c"), Entry(1, "a"), Entry(2, "b")]:
    heappush(els, entry)

print(els)

for _ in range(len(els)):
    print(heappop(els))
