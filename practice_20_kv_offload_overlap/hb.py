"""Sparse causal reachability; required memory edges never enter the HB graph."""
from collections import defaultdict, deque


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.successors = defaultdict(set)

    def node(self, identifier, **fields):
        if identifier in self.nodes:
            raise ValueError("duplicate node " + identifier)
        self.nodes[identifier] = dict(id=identifier, **fields)
        return identifier

    def edge(self, source, target, kind, **evidence):
        if source == target:
            raise ValueError("self edge")
        if target not in self.successors[source]:
            self.successors[source].add(target)
            self.edges.append(dict(source=source, target=target, kind=kind, **evidence))

    def topological(self):
        incoming = {n: 0 for n in self.nodes}
        for a, targets in self.successors.items():
            if a not in incoming:
                raise ValueError("unknown source")
            for b in targets:
                if b not in incoming:
                    raise ValueError("unknown target")
                incoming[b] += 1
        ready = deque(n for n, count in incoming.items() if not count)
        order = []
        while ready:
            n = ready.popleft()
            order.append(n)
            for b in self.successors[n]:
                incoming[b] -= 1
                if not incoming[b]:
                    ready.append(b)
        if len(order) != len(incoming):
            raise ValueError("cycle in happens-before graph")
        return order

    def verify(self, required):
        """Propagate only queried ancestor bits, not an O(all_nodes²) matrix."""
        order = self.topological()
        sources = {e['source'] for e in required}
        indices = {n: i for i, n in enumerate(sorted(sources))}
        ancestors = {n: 0 for n in self.nodes}
        for n in order:
            bits = ancestors[n] | ((1 << indices[n]) if n in indices else 0)
            for target in self.successors[n]:
                ancestors[target] |= bits
        return [{**e, 'satisfied': bool(ancestors[e['target']] & (1 << indices[e['source']]))}
                for e in required]
