"""Balanced unprofiled E/G/G/E controls, then independent plain/pipe captures."""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=a.resume)
cases=[('control-%d-%s'%(i,m),m,'none',5) for i,m in enumerate(('eager','graph','graph','eager'))]
cases += [(m+'-'+profile,m,profile,1) for profile in ('plain','pipe') for m in ('eager','graph')]
for name,mode,profile,count in cases:
    out=a.output/name
    if out.exists():
        if a.resume and (out/'complete.json').exists():continue
        raise RuntimeError('partial output preserved: '+str(out))
    cmd=[sys.executable,str(Path(__file__).with_name('run.py')),'--mode',mode,'--profile',profile,'--requests',str(count),'--output',str(out),'--port','8026']
    print('START',name,flush=True);subprocess.run(cmd,check=True)
print('SUITE COMPLETE',flush=True)
