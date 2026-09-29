"""Archive original replay inputs after acquisition, retaining large CANN data remotely."""
import argparse
import json
import tarfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    assert json.loads((a.results/'formal-r01/completed.json').read_text())['status']=='passed'
    selected=[]
    for run in ('qualification-r01','qualification-r02','formal-r01'):
        for file in sorted((a.results/run).rglob('*')):
            if not file.is_file():continue
            rel=file.relative_to(a.results)
            if '__pycache__' in rel.parts or 'analysis' in rel.parts:continue
            if 'profiler' in rel.parts:
                if file.name not in ('trace_view.json','kernel_details.csv','profiler_info.json','profiler_metadata.json'):continue
            selected.append((file,rel))
    # Original stdout/stderr logs retained alongside each acquisition directory.
    for run in ('qualification-r01','qualification-r02','formal-r01'):
        file=a.results.parents[1]/f'p27-{run}.log'
        if file.exists():selected.append((file,Path(run)/'stdout.log'))
    with tarfile.open(a.output,'w:gz') as tar:
        for file,rel in selected:tar.add(file,arcname=str(rel),recursive=False)
    print(json.dumps(dict(files=len(selected),bytes=a.output.stat().st_size,archive=str(a.output))),flush=True)


if __name__=='__main__':main()
