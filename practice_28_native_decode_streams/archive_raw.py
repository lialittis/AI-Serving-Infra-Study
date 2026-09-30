"""Archive portable evidence, excluding device binaries and compiler caches."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from common import save, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not (a.run / 'status.json').exists() or not (a.run / 'recovery.json').exists():
        raise RuntimeError('controller must finish recovery before archiving')
    if json.loads((a.run / 'status.json').read_text())['status'] not in ('passed', 'failed'):
        raise RuntimeError('cannot archive a running suite')
    a.output.mkdir(parents=True, exist_ok=False)
    # Collection code is frozen by the controller. Freeze the final offline
    # analysis separately, including its two repository-local helper modules.
    repo = Path(__file__).resolve().parent.parent
    sources = list(Path(__file__).resolve().parent.glob('*.py')) + [
        repo / 'practice_26_decode_utilization/analyze.py',
        repo / 'practice_17_vllm_multistream/analyze_run.py']
    analysis_hashes = {}
    for source in sources:
        relative = source.relative_to(repo)
        target = a.run / 'analysis_tools' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        analysis_hashes[str(relative)] = sha256(target)
    save(a.run / 'analysis_hashes.json', analysis_hashes)
    excluded = {'__pycache__', 'triton_cache', 'vllm_cache', 'FRAMEWORK', 'logs'}
    allowed_profiler = {'trace_view.json', 'kernel_details.csv', 'profiler_metadata.json', 'profiler_info_0.json'}
    manifest = {}
    with tarfile.open(a.output / 'evidence.tgz', 'w:gz') as archive:
        for path in sorted(a.run.rglob('*')):
            rel = path.relative_to(a.run)
            if not path.is_file() or excluded.intersection(rel.parts):
                continue
            if 'profiler' in rel.parts and path.name not in allowed_profiler:
                continue
            key = str(Path(a.run.name) / rel)
            manifest[key] = dict(bytes=path.stat().st_size, sha256=sha256(path))
            archive.add(path, arcname=key)
    save(a.output / 'manifest.json', manifest)
    save(a.output / 'archive.json', dict(remote_root=str(a.run.resolve()),
        archive_sha256=sha256(a.output / 'evidence.tgz'), files=len(manifest),
        uncompressed_bytes=sum(x['bytes'] for x in manifest.values()),
        archive_bytes=(a.output / 'evidence.tgz').stat().st_size))


if __name__ == '__main__':
    main()
