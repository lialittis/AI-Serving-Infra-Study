"""Archive reproducible portable evidence, leaving compiler/device binaries remote."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert json.loads((a.run/'status.json').read_text())['status'] in ('passed','failed')
    a.output.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parent.parent
    offline=[repo/'practice_29_prequeued_streams'/f for f in ('analyze.py','exact_join.py','render.py')]
    offline += [repo/'practice_28_native_decode_streams/common.py',repo/'practice_26_decode_utilization/analyze.py',
                repo/'practice_17_vllm_multistream/analyze_run.py']
    for source in offline:
        target=a.run/'analysis_tools'/source.relative_to(repo)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    manifest={}
    with tarfile.open(a.output/'evidence.tgz','w:gz') as tar:
        for file in sorted(a.run.rglob('*')):
            rel=file.relative_to(a.run)
            if not file.is_file() or {'FRAMEWORK','__pycache__','triton_cache','vllm_cache','logs'}.intersection(rel.parts):continue
            if 'profiler' in rel.parts and file.name not in ('trace_view.json','kernel_details.csv','profiler_info_0.json','profiler_metadata.json'):continue
            name=str(Path(a.run.name)/rel)
            manifest[name]=dict(bytes=file.stat().st_size,sha256=digest(file))
            tar.add(file,arcname=name)
    for name in ('status.json','analysis_summary.json','analysis_validation.json','gate_analysis.json'):
        if (a.run/name).exists():shutil.copy2(a.run/name,a.output/name)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.output/'archive.json').write_text(json.dumps(dict(archive_sha256=digest(a.output/'evidence.tgz'),
        files=len(manifest),archive_bytes=(a.output/'evidence.tgz').stat().st_size),indent=2)+'\n')


if __name__=='__main__':main()
