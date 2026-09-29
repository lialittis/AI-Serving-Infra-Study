"""Archive replayable evidence; keep large CANN buffers and compiler caches remote."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    assert len(list(a.root.glob('*/complete.json')))==8,'finish all eight runs before archiving'
    manifest={}
    for run in sorted(a.root.iterdir()):
        if not run.is_dir():continue
        cache=[dict(path=str(f.relative_to(run)),bytes=f.stat().st_size,sha256=digest(f)) for f in sorted((run/'compiler_cache').rglob('*')) if f.is_file()]
        (run/'compiler_cache_manifest.json').write_text(json.dumps(cache,indent=2)+'\n')
    with tarfile.open(a.output/'evidence.tgz','w:gz') as archive:
        for f in sorted(a.root.rglob('*')):
            if not f.is_file() or any(x in f.parts for x in ('compiler_cache','__pycache__','FRAMEWORK','logs','analysis')):continue
            if 'profiler' in f.parts and f.name not in ('trace_view.json','kernel_details.csv','profiler_info_0.json','profiler_metadata.json'):continue
            name=str(Path(a.root.name)/f.relative_to(a.root));manifest[name]=dict(bytes=f.stat().st_size,sha256=digest(f));archive.add(f,arcname=name)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(files=len(manifest),uncompressed_bytes=sum(x['bytes'] for x in manifest.values()),archive_bytes=(a.output/'evidence.tgz').stat().st_size)))
if __name__=='__main__':main()
