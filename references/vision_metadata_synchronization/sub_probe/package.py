"""Package completed runs, retaining trace and CSV but not CANN binaries."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root=Path(sys.argv[1]);suffix=sys.argv[2] if len(sys.argv)>2 else 'r02'
target=root/f'delivery-{suffix}';target.mkdir(exist_ok=False)
runs=([f'profile-{mode}-{suffix}' for mode in ('sub','tolist','sync')] if suffix!='r02' else
      ['api-r02']+[f'{kind}-{mode}-r02' for kind in ('plain','profile') for mode in ('sub','tolist','sync')])
for name in runs:
    source=root/name;dest=target/name;dest.mkdir()
    assert json.loads((source/'completed.json').read_text())['status']=='passed'
    for p in source.iterdir():
        if p.is_file():shutil.copy2(p,dest/p.name)
    if name.startswith('profile'):
        for filename in ('trace_view.json','kernel_details.csv'):
            files=list(source.rglob(filename));assert len(files)==1
            data=files[0].read_bytes()
            if filename.endswith('.json'):(dest/(filename+'.gz')).write_bytes(gzip.compress(data,mtime=0))
            else:(dest/filename).write_bytes(data)
        for filename in ('profiler_info.json','profiler_metadata.json'):
            files=list(source.rglob(filename))
            if len(files)==1:shutil.copy2(files[0],dest/filename)
(target/'device_after.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))
manifest={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(target.rglob('*')) if p.is_file()}
(target/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Packaged',len(manifest),'files')
