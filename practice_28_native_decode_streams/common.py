"""Standard-library helpers; no device initialization in the controller."""
import hashlib
import json
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Bindings:
    """Restore exact attributes even after partial entry or a Python exception.

    Values remain owned by the caller. Exiting restores Python bindings, NOT
    device completion; the submission owner must drain before releasing storage.
    """
    def __init__(self, entries):
        self.entries = list(entries)
        self.previous = []

    def __enter__(self):
        try:
            for obj, name, value in self.entries:
                owned = name in vars(obj)
                previous = getattr(obj, name, None)
                self.previous.append((obj, name, owned, previous))
                setattr(obj, name, value)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for obj, name, owned, value in reversed(self.previous):
            if owned:
                setattr(obj, name, value)
            elif name in vars(obj):
                delattr(obj, name)
        self.previous.clear()


def disjoint(named_ranges):
    """Cross-owner live storage intersection; addresses are process-local."""
    conflicts = []
    for i, (owner, name, begin, end) in enumerate(named_ranges):
        for other, other_name, lo, hi in named_ranges[i + 1:]:
            if owner != other and max(begin, lo) < min(end, hi):
                conflicts.append([owner, name, other, other_name])
    return conflicts


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative interval")
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def overlap(a, b):
    a, b = union(a), union(b)
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total
