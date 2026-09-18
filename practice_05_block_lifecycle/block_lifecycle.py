"""Observe physical block reuse and allocation identity, in a single thread."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockHandle:
    """An allocation reference, valid only within the pool that issued it."""

    block_id: int
    generation: int

    def __str__(self) -> str:
        return f"B{self.block_id}:g{self.generation}"


class BlockPool:
    """Reuse fixed physical slots; increment a block's generation on allocation."""

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.block_size = block_size
        self._owners: list[str | None] = [None] * num_blocks
        self._generations = [0] * num_blocks
        self._blocks: list[list[str | None]] = [
            [None] * block_size for _ in range(num_blocks)
        ]

    def allocate(self, request_id: str) -> BlockHandle:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        for block_id, owner in enumerate(self._owners):
            if owner is None:
                self._generations[block_id] += 1
                self._owners[block_id] = request_id
                handle = BlockHandle(block_id, self._generations[block_id])
                self._log("ALLOC", request_id, handle)
                return handle
        raise MemoryError("no free physical blocks")

    def _validate(self, request_id: str, handle: BlockHandle) -> list[str | None]:
        block_id = handle.block_id
        if not 0 <= block_id < len(self._blocks):
            raise IndexError("physical block ID out of range")
        current_generation = self._generations[block_id]
        if handle.generation != current_generation:
            raise ValueError(f"stale handle {handle}; current generation is g{current_generation}")
        owner = self._owners[block_id]
        if owner is None:
            raise ValueError(f"{handle} is FREE")
        if request_id != owner:
            raise ValueError(f"{handle} belongs to request={owner}, not request={request_id}")
        return self._blocks[block_id]

    def _validate_offset(self, offset: int) -> None:
        if not 0 <= offset < self.block_size:
            raise IndexError("block offset out of range")

    def write(self, request_id: str, handle: BlockHandle, offset: int, value: str) -> None:
        block = self._validate(request_id, handle)
        self._validate_offset(offset)
        if not isinstance(value, str):
            raise ValueError("a KV placeholder must be a string")
        block[offset] = value
        self._log("WRITE", request_id, handle, f" offset={offset} value={value}")

    def read(self, request_id: str, handle: BlockHandle, offset: int) -> str | None:
        block = self._validate(request_id, handle)
        self._validate_offset(offset)
        return block[offset]

    def free(self, request_id: str, handle: BlockHandle) -> None:
        block = self._validate(request_id, handle)
        # Clear in place for this teaching example; physical slots remain.
        block[:] = [None] * self.block_size
        self._owners[handle.block_id] = None
        # Do not reset generation: the next allocation must get a new identity.
        self._log("FREE", request_id, handle)

    @staticmethod
    def _log(event: str, request_id: str, handle: BlockHandle, detail: str = "") -> None:
        print(f"{event:<5} request={request_id} block={handle.block_id} "
              f"generation={handle.generation}{detail}")


def main() -> None:
    pool = BlockPool(num_blocks=1, block_size=4)
    a = pool.allocate("A")
    pool.write("A", a, 0, "A0")
    pool.free("A", a)
    b = pool.allocate("B")
    pool.write("B", b, 0, "B0")

    print(f"\nPhysical identity: B{a.block_id} == B{b.block_id}")
    print(f"Allocation identity: {a} != {b}")
    print("\nTry writing through A's old handle:")
    try:
        pool.write("A", a, 0, "late A write")
    except ValueError as error:
        print(f"REJECT {error}")
    print(f"B still reads: {pool.read('B', b, 0)}")
    pool.free("B", b)


if __name__ == "__main__":
    main()
