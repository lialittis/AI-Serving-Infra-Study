"""Freeze portable evidence and standalone offline tools, with per-file hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert json.loads((a.run/'status.json').read_text())['status']=='passed'
    a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent;repo=root.parent
    tools=[root/name for name in ('analyze.py','exact_join.py','render.py','viewer.html','compare.py',
                                 'forward_analysis.py','test_analysis.py','test_forward.py')]
    tools.append(repo/'practice_17_vllm_multistream/analyze_run.py')
    for source in tools:
        target=a.run/'analysis_tools'/source.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    manifest={}
    with tarfile.open(a.output/'evidence.tgz','w:gz') as tar:
        for file in sorted(a.run.rglob('*')):
            rel=file.relative_to(a.run)
            if not file.is_file() or {'FRAMEWORK','__pycache__','triton_cache','vllm_cache','logs'}.intersection(rel.parts):continue
            if 'profiler' in rel.parts and file.name not in ('trace_view.json','kernel_details.csv','profiler_info_0.json','profiler_metadata.json'):continue
            key=str(Path(a.run.name)/rel);manifest[key]=dict(bytes=file.stat().st_size,sha256=digest(file));tar.add(file,arcname=key)
    for source,name in [(a.run/'status.json','status.json'),(a.run/'analysis/summary.json','summary.json'),
                        (a.run/'analysis/validation.json','validation.json')]:shutil.copy2(source,a.output/name)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.output/'archive.json').write_text(json.dumps(dict(files=len(manifest),archive_sha256=digest(a.output/'evidence.tgz'),
        bytes=(a.output/'evidence.tgz').stat().st_size,uncompressed_bytes=sum(x['bytes'] for x in manifest.values())),indent=2)+'\n')


if __name__=='__main__':main()
