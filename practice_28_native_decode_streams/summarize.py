"""Summarize only completed, numerically qualified measurements."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from common import save


def read(path):
    return json.loads(path.read_text())


def stats(values):
    return dict(n=len(values), median=statistics.median(values), min=min(values), max=max(values))


def summarize(run):
    status, recovery = read(run / 'status.json'), read(run / 'recovery.json')
    qualification = {}
    for path in sorted(run.glob('*-qualification/status.json')):
        checks = path.with_name('qualification.json')
        qualification[path.parent.name] = dict(status=read(path),
            checks=len(read(checks)) if checks.exists() else 0)
    result = dict(run=run.name, status=status, recovery=recovery, qualification=qualification,
                  revisions=read(run / 'revisions.json'), performance=[], diagnostics=[])
    runs = sorted(run.glob('measure-*/measurements.json'))
    if runs:
        assert status['status'] == 'passed' and recovery['passed'], 'incomplete suite: exclude performance'
        assert len(runs) == 6, 'three independent processes per backend required'
        grouped = defaultdict(list)
        for path in runs:
            child = read(path.with_name('status.json'))
            assert child['status'] == 'passed'
            checks = read(path.with_name('correctness.json'))
            assert len(checks) == 96 and all(c['passed'] for c in checks)
            records = read(path)
            assert len(records) == 48
            for case in ('same32', 'mixed16_47'):
                cases = [r for r in records if r['case'] == case]
                by_strategy = {}
                for strategy in ('serial', 'parallel'):
                    rows = [r for r in cases if r['strategy'] == strategy]
                    assert sorted(r['repeat'] for r in rows) == list(range(12))
                    by_strategy[strategy] = dict(
                        wall_us=stats([r['wall_us'] for r in rows]),
                        device_ready_us=stats([max(r['ready_us'].values()) for r in rows]),
                        host_submission_us=stats([max(r['host_submission_us'].values()) for r in rows]),
                        peak_allocated_bytes=stats([r['allocated_peak'] for r in rows]))
                serial, parallel = by_strategy['serial'], by_strategy['parallel']
                grouped[child['mode'], case].append(dict(process=path.parent.name,
                    wall_speedup=serial['wall_us']['median']/parallel['wall_us']['median'],
                    device_speedup=serial['device_ready_us']['median']/parallel['device_ready_us']['median'],
                    strategies=by_strategy))
        for (mode, case), processes in sorted(grouped.items()):
            assert len(processes) == 3
            result['performance'].append(dict(mode=mode, case=case, processes=processes,
                wall_speedup=stats([p['wall_speedup'] for p in processes]),
                device_speedup=stats([p['device_speedup'] for p in processes])))
    for path in sorted(run.glob('*-*/analysis_summary.json')):
        result['diagnostics'].append(dict(process=path.parent.name, trials=read(path)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    save(a.output, summarize(a.run))


if __name__ == '__main__':
    main()
