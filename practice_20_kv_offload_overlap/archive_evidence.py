"""Archive completed runs without redundant raw profiler databases."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for directory in sorted(a.run.glob('block-*')):
        assert json.loads((directory/'shutdown.json').read_text())['exit_code']==0
    manifest={}
    with tarfile.open(a.output,'w:gz') as archive:
        for path in sorted(a.run.rglob('*')):
            if not path.is_file():
                continue
            relative=path.relative_to(a.run)
            if 'profiler' in relative.parts and path.name not in ('trace_view.json','kernel_details.csv'):
                continue
            if 'analysis' in relative.parts or '__pycache__' in relative.parts:
                continue
            archive.add(path,arcname=str(relative),recursive=False)
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):
                    h.update(chunk)
            manifest[str(relative)]=dict(bytes=path.stat().st_size,sha256=h.hexdigest())
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(archive=str(a.output),files=len(manifest),bytes=a.output.stat().st_size)))


if __name__=='__main__':
    main()
