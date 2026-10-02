"""Extract the selected model launch from the original P30 profiler trace."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
D = lambda x: Decimal(str(x))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    a = p.parse_args()
    report = ROOT.parent / 'report/attention/attention.json'
    attention = json.loads(report.read_text())
    link, = [x for x in attention['links'] if x['task']['name'] == 'FusedInferAttentionScore']
    trace = json.loads(a.trace.read_text(), parse_float=Decimal)
    if isinstance(trace, dict):
        trace = trace['traceEvents']
    node = link['cann']
    matches = [(i, e) for i, e in enumerate(trace)
               if e.get('name') == 'AscendCL@aclrtLaunchKernelWithHostArgs'
               and str(e.get('tid')) == str(node['tid']) and e.get('ph') == 'X'
               and D(node['ts']) <= D(e['ts'])
               and D(e['ts']) + D(e['dur']) <= D(node['ts']) + D(node['dur'])]
    assert len(matches) == 1, matches
    index, launch = matches[0]
    result = dict(run='attention-01', trace_path=str(a.trace), trace_index=index,
                  trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),
                  attention_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                  launch=launch, link=link)
    (ROOT / 'evidence/model_launch.json').write_text(json.dumps(result, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
