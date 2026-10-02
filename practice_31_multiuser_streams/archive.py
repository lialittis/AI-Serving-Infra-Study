"""Export compact evidence; keep bulky raw profiler files at recorded remote paths."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('destination',type=Path)
    a=p.parse_args();run=a.run.resolve();out=a.destination.resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):
        p.error('Source and destination must be separate trees')
    cases=sorted(run.glob('c*-*'))
    for case in cases:
        status=json.loads((case/'status.json').read_text())
        if status['status']!='passed':raise ValueError('Incomplete run: '+str(case))
        summary=json.loads((case/'analysis/summary.json').read_text())
        if summary.get('analysis_status','passed')!='passed':raise ValueError('Incomplete analysis: '+str(case))
    out.mkdir(parents=True,exist_ok=False)
    retained=[];raw=[]
    for path in sorted(run.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(run)
        if 'profiler' in relative.parts:
            raw.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=digest(path)))
            continue
        destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        retained.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=digest(path)))
    # Analysis can improve after collection. Preserve the code that generated
    # this archive separately from the immutable collection-time snapshot.
    for name in ('analyze.py','render.py','viewer.html','archive.py','audit_raw.py','check_report.cjs'):
        path=Path(__file__).with_name(name)
        if not path.exists():continue
        relative=Path('analysis_sources')/name
        destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        retained.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=digest(path)))
    manifest=dict(schema=1,remote_host='ascend910',remote_root=str(run),
                  raw_profiler_retained_remote=True,raw_profiler_files=raw,retained_files=retained,
                  reproduction='Restore raw files at the relative paths above, then run analyze.py; metadata and compact reports alone cannot rerun the raw flow join.')
    (out/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(files=len(retained),retained_bytes=sum(x['bytes'] for x in retained),raw_bytes=sum(x['bytes'] for x in raw))))


if __name__=='__main__':main()
