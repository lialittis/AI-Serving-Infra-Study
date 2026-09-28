"""Dependency construction and offline scheduling, with no accelerator imports.

Intervals are half-open byte ranges. Bounding ranges deliberately overestimate
strided/indexed accesses. Edges retain RAW/WAR/WAW and allocation reuse separately.
"""
from collections import defaultdict
from decimal import Decimal
import heapq
import math


def ns(value):
    return int(Decimal(str(value)) * 1000)


def topological(nodes, edges):
    ids = [n['id'] for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate node')
    successors = {n: set() for n in ids}
    predecessors = {n: set() for n in ids}
    for edge in edges:
        a, b = edge['source'], edge['target']
        if a not in successors or b not in successors or a == b:
            raise ValueError('dangling or self edge')
        successors[a].add(b)
        predecessors[b].add(a)
    incoming = {n: len(predecessors[n]) for n in ids}
    ready = [n for n in ids if not incoming[n]]
    heapq.heapify(ready)
    order = []
    while ready:
        n = heapq.heappop(ready)
        order.append(n)
        for child in sorted(successors[n]):
            incoming[child] -= 1
            if not incoming[child]:
                heapq.heappush(ready, child)
    if len(order) != len(ids):
        raise ValueError('cycle')
    return order, predecessors, successors


def reachability(nodes, edges):
    order, _, successors = topological(nodes, edges)
    indices = {n: i for i, n in enumerate(order)}
    reachable = {}
    reduced = set()
    for n in reversed(order):
        covered = 0
        for child in sorted(successors[n], key=indices.get):
            bit = 1 << indices[child]
            if not covered & bit:
                reduced.add((n, child))
            covered |= bit | reachable[child]
        reachable[n] = covered
    return indices, reachable, reduced


class MemoryFrontier:
    """Interval splitting preserves old writers outside a partial overwrite.

    Storage generations do not erase physical-memory hazards: address reuse
    adds a lifetime edge even between two reads of different allocations.
    """
    def __init__(self):
        self.segments = defaultdict(list)
        self.edges = []
        self.external_reads = []

    def access(self, node, access):
        device, lo, hi = access['device'], access['lo'], access['hi']
        generation, mode = access['generation'], access['mode']
        if mode not in ('R', 'W') or hi < lo:
            raise ValueError('invalid memory access')
        if lo == hi:
            return
        previous = self.segments[device]
        overlap = [s for s in previous if s[0] < hi and lo < s[1]]
        cuts = sorted({lo, hi} | {max(lo, s[0]) for s in overlap}
                      | {min(hi, s[1]) for s in overlap})
        updated = [s for s in previous if s[1] <= lo or s[0] >= hi]
        for a, b, gen, writer, readers in overlap:
            if a < lo:
                updated.append((a, lo, gen, writer, set(readers)))
            if b > hi:
                updated.append((hi, b, gen, writer, set(readers)))
        for a, b in zip(cuts, cuts[1:]):
            old = next((s for s in overlap if s[0] <= a and b <= s[1]), None)
            gen, writer, readers = (old[2], old[3], set(old[4])) if old else (generation, None, set())

            def edge(source, kind):
                if source is not None and source != node:
                    self.edges.append(dict(source=source, target=node, kind=kind,
                                           device=device, lo=a, hi=b,
                                           generation=generation))

            if gen != generation:
                for prior in readers | ({writer} if writer else set()):
                    edge(prior, 'allocation_reuse')
                writer, readers = None, set()
            if mode == 'R':
                edge(writer, 'RAW')
                if writer is None:
                    self.external_reads.append(dict(node=node, device=device, lo=a, hi=b,
                                                    generation=generation))
                readers.add(node)
            else:
                edge(writer, 'WAW')
                for prior in readers:
                    edge(prior, 'WAR')
                writer, readers = node, set()
            updated.append((a, b, generation, writer, readers))
        merged = []
        for segment in sorted(updated, key=lambda s: s[0]):
            if merged and merged[-1][1] == segment[0] and merged[-1][2:] == segment[2:]:
                merged[-1] = (merged[-1][0], segment[1], *segment[2:])
            else:
                merged.append(segment)
        self.segments[device] = merged


def metrics(nodes, edges):
    order, pred, succ = topological(nodes, edges)
    duration = {n['id']: n['duration_ns'] for n in nodes}
    if any(not isinstance(d, int) or d < 0 for d in duration.values()):
        raise ValueError('duration must be nonnegative integer nanoseconds')
    early, finish, parent = {}, {}, {}
    for n in order:
        parent[n] = max(sorted(pred[n]), key=lambda p: finish[p], default=None)
        early[n] = finish[parent[n]] if parent[n] is not None else 0
        finish[n] = early[n] + duration[n]
    last = max(order, key=lambda n: finish[n], default=None)
    critical = finish[last] if last is not None else 0
    tail, late = {}, {}
    for n in reversed(order):
        tail[n] = duration[n] + max((tail[c] for c in succ[n]), default=0)
        late[n] = critical - tail[n]
    path = []
    while last is not None:
        path.append(last)
        last = parent[last]
    path.reverse()
    activity = defaultdict(int)
    for n in order:
        if duration[n]:
            activity[early[n]] += 1
            activity[finish[n]] -= 1
    active = peak = 0
    for t in sorted(activity):
        active += activity[t]
        peak = max(peak, active)
    work = sum(duration.values())
    return dict(work_ns=work, critical_path_ns=critical, critical_path=path,
                work_over_span=work / critical if critical else 0,
                asap_peak_active=peak, earliest_start_ns=early,
                slack_ns={n: late[n] - early[n] for n in order}, upward_rank_ns=tail)


def schedule(nodes, edges, streams, event_cost_ns=0):
    """Deterministic upward-rank list scheduling; costs are assumptions, not timings.

    No insertion into previously scheduled gaps. Every cross-stream edge has an
    explicit event generation; the emitted plan is never applied to a live model.
    """
    if streams < 1 or event_cost_ns < 0:
        raise ValueError('invalid scheduling parameters')
    order, pred, succ = topological(nodes, edges)
    m = metrics(nodes, edges)
    duration = {n['id']: n['duration_ns'] for n in nodes}
    incoming = {n: len(pred[n]) for n in order}
    ready = [(-m['upward_rank_ns'][n], n) for n in order if not incoming[n]]
    heapq.heapify(ready)
    available = [0] * streams
    placed, rows = {}, []
    while ready:
        _, n = heapq.heappop(ready)
        options = []
        for stream in range(streams):
            release = max((placed[p]['end_ns'] +
                           (event_cost_ns if placed[p]['stream'] != stream else 0)
                           for p in pred[n]), default=0)
            start = max(available[stream], release)
            options.append((start + duration[n], start, stream))
        end, start, stream = min(options)
        row = dict(id=n, stream=stream, start_ns=start, end_ns=end)
        rows.append(row)
        placed[n] = row
        available[stream] = end
        for child in sorted(succ[n]):
            incoming[child] -= 1
            if incoming[child] == 0:
                heapq.heappush(ready, (-m['upward_rank_ns'][child], child))
    # Transitive reduction avoids one event wait for every redundant RAW/barrier
    # edge. FIFO + emitted waits must still imply *every* original dependency.
    _, _, reduced = reachability(nodes, edges)
    events = []
    for n in order:
        targets = sorted(p for p in pred[n] if (p, n) in reduced and placed[p]['stream'] != placed[n]['stream'])
        for p in targets:
            events.append(dict(event='event:' + p, generation=1, record_after=p,
                               wait_before=n, source_stream=placed[p]['stream'],
                               target_stream=placed[n]['stream']))
    result = dict(stream_count=streams, event_cost_ns=event_cost_ns,
                  makespan_ns=max(available),
                  lower_bound_ns=max(m['critical_path_ns'], math.ceil(m['work_ns'] / streams)),
                  rows=rows, events=events)
    validate_schedule(nodes, edges, result)
    return result


def validate_schedule(nodes, edges, plan):
    topological(nodes, edges)
    rows = {r['id']: r for r in plan['rows']}
    if len(rows) != len(plan['rows']) or set(rows) != {n['id'] for n in nodes}:
        raise ValueError('schedule coverage mismatch')
    lanes = defaultdict(list)
    for n in nodes:
        r = rows[n['id']]
        if not 0 <= r['stream'] < plan['stream_count'] or r['start_ns'] < 0:
            raise ValueError('invalid stream/start')
        if r['end_ns'] - r['start_ns'] != n['duration_ns']:
            raise ValueError('duration changed')
        lanes[r['stream']].append(r)
    happens_before = []
    for lane in lanes.values():
        lane.sort(key=lambda r: (r['start_ns'], r['end_ns']))
        if any(a['end_ns'] > b['start_ns'] for a, b in zip(lane, lane[1:])):
            raise ValueError('stream overlap')
        happens_before += [dict(source=a['id'], target=b['id']) for a, b in zip(lane, lane[1:])]
    waits, generations = set(), {}
    for event in plan['events']:
        producer, consumer = event['record_after'], event['wait_before']
        if producer not in rows or consumer not in rows:
            raise ValueError('dangling event')
        if (event['source_stream'] != rows[producer]['stream']
                or event['target_stream'] != rows[consumer]['stream']
                or event['source_stream'] == event['target_stream']):
            raise ValueError('event stream mismatch')
        if not isinstance(event['generation'], int) or event['generation'] < 1:
            raise ValueError('invalid event generation')
        key = (event['event'], event['generation'])
        if key in generations and generations[key] != producer:
            raise ValueError('event generation has multiple producers')
        generations[key] = producer
        waits.add((producer, consumer))
        happens_before.append(dict(source=producer, target=consumer))
        if rows[producer]['end_ns'] + plan['event_cost_ns'] > rows[consumer]['start_ns']:
            raise ValueError('event timing violation')
    indices, reachable, _ = reachability(nodes, happens_before)
    for e in edges:
        a, b = rows[e['source']], rows[e['target']]
        cross = a['stream'] != b['stream']
        cost = plan['event_cost_ns'] if cross else 0
        if a['end_ns'] + cost > b['start_ns']:
            raise ValueError('dependency violation')
        if cross and not (reachable[a['id']] & (1 << indices[b['id']])):
            raise ValueError('missing cross-stream event')
    if plan['makespan_ns'] != max((r['end_ns'] for r in rows.values()), default=0):
        raise ValueError('incorrect makespan')
    if plan.get('atomic_native_scopes'):
        groups = defaultdict(list)
        for n in nodes:
            groups[n.get('owner') or n['id']].append(rows[n['id']])
        for members in groups.values():
            if len({r['stream'] for r in members}) != 1:
                raise ValueError('native scope split across streams')
            if any(a['end_ns'] != b['start_ns'] for a, b in zip(members, members[1:])):
                raise ValueError('native scope not contiguous')


def schedule_scopes(nodes, edges, streams, event_cost_ns=0):
    """Keep all owned kernels of a native/public call contiguous on one stream.

    The native API cannot independently submit each of its hidden kernels onto a
    different stream. Group contraction can expose an unsupported interleaving;
    reject that cycle instead of silently dropping constraints.
    """
    groups, group_for = {}, {}
    for n in nodes:
        owner = n.get('owner') or n['id']
        groups.setdefault(owner, []).append(n)
        group_for[n['id']] = owner
    ids = {owner: f'g{i:04d}' for i, owner in enumerate(groups)}
    coarse_nodes = [dict(id=ids[owner], duration_ns=sum(n['duration_ns'] for n in members))
                    for owner, members in groups.items()]
    pairs = {(ids[group_for[e['source']]], ids[group_for[e['target']]]) for e in edges
             if group_for[e['source']] != group_for[e['target']]}
    coarse_edges = [dict(source=a, target=b) for a, b in sorted(pairs)]
    coarse = schedule(coarse_nodes, coarse_edges, streams, event_cost_ns)
    by_coarse = {ids[owner]: members for owner, members in groups.items()}
    rows = []
    for row in coarse['rows']:
        time = row['start_ns']
        for n in by_coarse[row['id']]:
            rows.append(dict(id=n['id'], stream=row['stream'], start_ns=time, end_ns=time + n['duration_ns']))
            time += n['duration_ns']
    events = []
    for e in coarse['events']:
        producer = by_coarse[e['record_after']][-1]['id']
        consumer = by_coarse[e['wait_before']][0]['id']
        events.append(dict(e, event='event:' + producer, record_after=producer, wait_before=consumer))
    result = dict(coarse, rows=rows, events=events, atomic_native_scopes=True, atomic_scope_count=len(groups))
    validate_schedule(nodes, edges, result)
    return result
