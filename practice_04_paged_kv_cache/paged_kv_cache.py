"""Map logical token positions to physical KV slots using only Python lists.

A label such as "A0" stands for that token's K/V record, not a real KV tensor.
"""

from __future__ import annotations

from collections.abc import Iterable


class BlockPool:
    """Own fixed-size physical blocks; allocate the lowest available block ID."""

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        # None means FREE; an allocated block contains block_size token slots.
        self._blocks: list[list[str | None] | None] = [None] * num_blocks

    @property
    def free_count(self) -> int:
        return sum(block is None for block in self._blocks)

    def allocate(self) -> int:
        for block_id, block in enumerate(self._blocks):
            if block is None:
                self._blocks[block_id] = [None] * self.block_size
                return block_id
        raise MemoryError("no free physical KV blocks")

    def _get_block(self, block_id: int) -> list[str | None]:
        if not 0 <= block_id < self.num_blocks:
            raise IndexError("physical block ID out of range")
        block = self._blocks[block_id]
        if block is None:
            raise ValueError(f"B{block_id} is FREE")
        return block

    def release(self, block_id: int) -> None:
        self._get_block(block_id)  # Reject releasing a FREE or invalid block.
        self._blocks[block_id] = None

    def write(self, block_id: int, offset: int, value: str) -> None:
        block = self._get_block(block_id)
        if not 0 <= offset < self.block_size:
            raise IndexError("block offset out of range")
        if not isinstance(value, str):
            raise ValueError("a KV placeholder must be a string")
        block[offset] = value

    def read(self, block_id: int, offset: int) -> str | None:
        block = self._get_block(block_id)
        if not 0 <= offset < self.block_size:
            raise IndexError("block offset out of range")
        return block[offset]

    def snapshot(self) -> tuple[tuple[str | None, ...] | None, ...]:
        """Return immutable block contents; None outside a block means FREE."""
        return tuple(None if block is None else tuple(block) for block in self._blocks)


class BlockTable:
    """One request's logical-block index -> physical-block ID mapping."""

    def __init__(self, pool: BlockPool):
        self.pool = pool
        self._physical_blocks: list[int] = []

    @property
    def physical_blocks(self) -> tuple[int, ...]:
        return tuple(self._physical_blocks)

    def ensure_capacity(self, num_tokens: int) -> None:
        if num_tokens < 0:
            raise ValueError("num_tokens must be nonnegative")
        required = (num_tokens + self.pool.block_size - 1) // self.pool.block_size
        additional = max(0, required - len(self._physical_blocks))
        # Check first so an out-of-capacity append cannot partially allocate.
        if additional > self.pool.free_count:
            raise MemoryError(f"need {additional} free blocks; have {self.pool.free_count}")
        for _ in range(additional):
            self._physical_blocks.append(self.pool.allocate())

    def locate(self, token_index: int) -> tuple[int, int]:
        """Resolve an allocated slot; Request checks whether its token exists."""
        if token_index < 0:
            raise IndexError("token index must be nonnegative")
        logical_block = token_index // self.pool.block_size
        offset = token_index % self.pool.block_size
        if logical_block >= len(self._physical_blocks):
            raise IndexError("token index exceeds allocated blocks")
        physical_block = self._physical_blocks[logical_block]
        return physical_block, offset

    def release(self) -> None:
        for block_id in self._physical_blocks:
            self.pool.release(block_id)
        self._physical_blocks.clear()


class Request:
    """Own a logical sequence length and a block table, not a contiguous cache."""

    def __init__(self, request_id: str, pool: BlockPool):
        self.request_id = request_id
        self.block_table = BlockTable(pool)
        self._num_tokens = 0
        self._released = False

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    def append(self, kv_labels: Iterable[str]) -> None:
        if self._released:
            raise ValueError("request has been released")
        labels = list(kv_labels)
        if any(not isinstance(label, str) for label in labels):
            raise ValueError("KV placeholders must be strings")
        self.block_table.ensure_capacity(self.num_tokens + len(labels))
        for label in labels:
            block_id, offset = self.block_table.locate(self._num_tokens)
            self.block_table.pool.write(block_id, offset, label)
            self._num_tokens += 1

    def locate(self, token_index: int) -> tuple[int, int]:
        if not 0 <= token_index < self.num_tokens:
            raise IndexError("token index outside the request's logical sequence")
        return self.block_table.locate(token_index)

    def read(self, token_index: int) -> str:
        block_id, offset = self.locate(token_index)
        value = self.block_table.pool.read(block_id, offset)
        assert value is not None, "a logical token must have a stored KV placeholder"
        return value

    def read_all(self) -> list[str]:
        return [self.read(index) for index in range(self.num_tokens)]

    def release(self) -> None:
        self.block_table.release()
        self._num_tokens = 0
        self._released = True


def print_state(title: str, pool: BlockPool, *requests: Request) -> None:
    print(f"\n{title}")
    for request in requests:
        print(f"{request.request_id}: {request.num_tokens} logical tokens")
        for logical_block, physical_block in enumerate(request.block_table.physical_blocks):
            print(f"  logical block {logical_block} -> B{physical_block}")
    for block_id, block in enumerate(pool.snapshot()):
        contents = "FREE" if block is None else "[" + " ".join(
            "--" if slot is None else slot for slot in block
        ) + "]"
        print(f"B{block_id}: {contents}")


def main() -> None:
    pool = BlockPool(num_blocks=4, block_size=4)
    a = Request("A", pool)
    b = Request("B", pool)
    print_state("1. Initial physical block pool", pool)

    a.append(f"A{i}" for i in range(6))
    print_state("2. A appends 6 tokens", pool, a)

    b.append(f"B{i}" for i in range(4))
    print_state("3. B appends 4 tokens (sequential allocation)", pool, a, b)

    a.append(["A6", "A7", "A8"])
    print_state("4. A appends 3 tokens; its blocks are now B0, B1, B3", pool, a, b)
    print("\nA's token mapping:")
    print("token_index  logical_block  physical_block  offset  KV_label")
    for index in range(a.num_tokens):
        physical_block, offset = a.locate(index)
        print(f"{index:11}  {index // pool.block_size:13}  B{physical_block:<13} {offset:6}  {a.read(index)}")
    print("A reads in logical order:", a.read_all())

    a.release()
    print_state("5. A releases its blocks; B is unchanged", pool, b)
    c = Request("C", pool)
    c.append(f"C{i}" for i in range(5))
    print_state("6. C reuses B0 and B1; unwritten slots contain no old A data", pool, b, c)


if __name__ == "__main__":
    main()
