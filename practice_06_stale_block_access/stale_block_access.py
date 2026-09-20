"""Deterministically detect a KV lifetime violation without any threads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockReference:
    """Keep the allocation's generation, even after its physical block is reused."""

    request_id: str
    block_id: int
    generation: int

    def __str__(self) -> str:
        return f"B{self.block_id}:g{self.generation}"


class StaleBlockAccess(RuntimeError):
    """A structured report of an allocation-generation mismatch."""

    def __init__(self, reference: BlockReference, current_generation: int):
        self.request_id = reference.request_id
        self.block_id = reference.block_id
        self.expected_generation = reference.generation
        self.current_generation = current_generation
        super().__init__(
            "STALE BLOCK ACCESS\n"
            f"request={self.request_id}\n"
            f"block={self.block_id}\n"
            f"expected_generation={self.expected_generation}\n"
            f"current_generation={self.current_generation}"
        )


def check_generation(reference: BlockReference, current_generation: int) -> None:
    """The minimal detector: compare the saved version with the current version."""
    if reference.generation != current_generation:
        raise StaleBlockAccess(reference, current_generation)


class Allocator:
    """A single-threaded pool; each block holds one symbolic KV payload."""

    def __init__(self, num_blocks: int = 1):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self._owners: list[str | None] = [None] * num_blocks
        self._generations = [0] * num_blocks
        self._payloads: list[str | None] = [None] * num_blocks

    def allocate(self, request_id: str) -> BlockReference:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        for block_id, owner in enumerate(self._owners):
            if owner is None:
                self._generations[block_id] += 1
                self._owners[block_id] = request_id
                reference = BlockReference(request_id, block_id, self._generations[block_id])
                print(f"ALLOC {request_id} {reference}")
                return reference
        raise MemoryError("no free physical blocks")

    def _validate(self, reference: BlockReference) -> int:
        block_id = reference.block_id
        if not 0 <= block_id < len(self._owners):
            raise IndexError("physical block ID out of range")
        # Compare before inspecting the owner or touching the payload.
        check_generation(reference, self._generations[block_id])
        owner = self._owners[block_id]
        if owner is None:
            raise ValueError(f"{reference} is FREE")
        if owner != reference.request_id:
            raise ValueError(f"{reference} belongs to request={owner}")
        return block_id

    def free(self, reference: BlockReference) -> None:
        block_id = self._validate(reference)
        self._payloads[block_id] = None
        self._owners[block_id] = None
        print(f"FREE  {reference.request_id} {reference}")

    def write(self, reference: BlockReference, payload: str) -> None:
        block_id = self._validate(reference)
        if not isinstance(payload, str):
            raise ValueError("KV payload must be a string placeholder")
        self._payloads[block_id] = payload
        print(f"WRITE {reference.request_id} {reference}")

    def read(self, reference: BlockReference) -> str | None:
        # This is an attempted read, not evidence of a successful payload access.
        print(f"{reference.request_id} READ {reference}")
        block_id = self._validate(reference)
        return self._payloads[block_id]


def main() -> None:
    allocator = Allocator(num_blocks=1)
    a_block = allocator.allocate("A")
    print(f"\nA remembers {a_block}\n")
    allocator.free(a_block)
    print()
    b_block = allocator.allocate("B")
    allocator.write(b_block, "B's KV data")
    print()

    try:
        allocator.read(a_block)  # Deliberately retain and use an old reference.
    except StaleBlockAccess as error:
        print(error)
    else:
        raise AssertionError("the detector failed to reject A's stale reference")


if __name__ == "__main__":
    main()
