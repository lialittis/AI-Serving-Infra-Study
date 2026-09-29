"""Generate ABI-checked CANN audit wrappers; compile against installed headers."""
import argparse
import json
from pathlib import Path
import subprocess

# Fields are stream, object (event/model/output handle), extra, stream ID.
APIS = [
('aclrtCreateStream','aclrtStream *s','s','rc==0?*s:0','0','0'),
('aclrtCreateStreamWithConfig','aclrtStream *s, uint32_t priority, uint32_t flag','s,priority,flag','rc==0?*s:0','priority','flag'),
('aclrtCreateStreamV2','aclrtStream *s, const aclrtStreamConfigHandle *cfg','s,cfg','rc==0?*s:0','cfg','0'),
('aclrtDestroyStream','aclrtStream s','s','s','0','0'),
('aclrtDestroyStreamForce','aclrtStream s','s','s','0','0'),
('aclrtStreamGetId','aclrtStream s, int32_t *id','s,id','s','rc==0?*id:-1','0'),
('aclrtCtxGetCurrentDefaultStream','aclrtStream *s','s','rc==0?*s:0','0','0'),
('aclrtCreateEvent','aclrtEvent *e','e','0','rc==0?*e:0','0'),
('aclrtCreateEventWithFlag','aclrtEvent *e, uint32_t flag','e,flag','0','rc==0?*e:0','flag'),
('aclrtCreateEventExWithFlag','aclrtEvent *e, uint32_t flag','e,flag','0','rc==0?*e:0','flag'),
('aclrtDestroyEvent','aclrtEvent e','e','0','e','0'),
('aclrtRecordEvent','aclrtEvent e, aclrtStream s','e,s','s','e','0'),
('aclrtResetEvent','aclrtEvent e, aclrtStream s','e,s','s','e','0'),
('aclrtStreamWaitEvent','aclrtStream s, aclrtEvent e','s,e','s','e','0'),
('aclrtStreamWaitEventWithTimeout','aclrtStream s, aclrtEvent e, int32_t timeout','s,e,timeout','s','e','timeout'),
('aclrtSynchronizeStream','aclrtStream s','s','s','0','0'),
('aclrtSynchronizeStreamWithTimeout','aclrtStream s, int32_t timeout','s,timeout','s','0','timeout'),
('aclrtSynchronizeEvent','aclrtEvent e','e','0','e','0'),
('aclrtQueryEventStatus','aclrtEvent e, aclrtEventRecordedStatus *status','e,status','0','e','rc==0?*status:-1'),
('aclrtSetStreamResLimit','aclrtStream s, aclrtDevResLimitType type, uint32_t value','s,type,value','s','type','value'),
('aclmdlRICaptureBegin','aclrtStream s, aclmdlRICaptureMode mode','s,mode','s','0','mode'),
('aclmdlRICaptureEnd','aclrtStream s, aclmdlRI *m','s,m','s','rc==0?*m:0','0'),
('aclmdlRIExecuteAsync','aclmdlRI m, aclrtStream s','m,s','s','m','0'),
('aclmdlRIDestroy','aclmdlRI m','m','0','m','0'),
('aclmdlRIBindStream','aclmdlRI m, aclrtStream s, uint32_t flag','m,s,flag','s','m','flag'),
('aclmdlRIUnbindStream','aclmdlRI m, aclrtStream s','m,s','s','m','0'),
('aclmdlRICaptureTaskGrpBegin','aclrtStream s','s','s','0','0'),
('aclmdlRICaptureTaskGrpEnd','aclrtStream s, aclrtTaskGrp *h','s,h','s','rc==0?*h:0','0'),
('aclmdlRICaptureTaskUpdateBegin','aclrtStream s, aclrtTaskGrp h','s,h','s','h','0'),
('aclmdlRICaptureTaskUpdateEnd','aclrtStream s','s','s','0','0'),
]


def build(output, cann):
    root=Path(__file__).parent
    code=(root/'native_template.c').read_text()
    wrappers=[]
    for i,(name,signature,call,s,obj,extra) in enumerate(APIS):
      for variant in range(4):
        wrappers.append('''static aclError p22_%s_%d(%s) {
 __typeof__(&%s) fn=(__typeof__(&%s))originals[%d]; uint64_t begin=now_ns();
 aclError rc=fn(%s); uint64_t finish=now_ns();
 record("%s",begin,finish,(uintptr_t)(%s),(uintptr_t)(%s),(uint64_t)(%s),rc,__builtin_return_address(0)); return rc;
}''' % (name,variant,signature,name,name,i*4+variant,call,name,s,obj,extra))
    code=code.replace('/*WRAPPERS*/','\n'.join(wrappers))
    code=code.replace('/*NAMES*/',','.join('"'+a[0]+'"' for a in APIS))
    code=code.replace('/*FUNCTIONS*/',','.join('{'+','.join('(void*)p22_'+a[0]+'_'+str(v) for v in range(4))+'}' for a in APIS))
    output.mkdir(parents=True,exist_ok=True)
    source=output/'native_stream_trace.c';source.write_text(code)
    command=['gcc','-shared','-fPIC','-O2','-g','-Wall','-Werror','-Wl,-Bsymbolic','-I'+str(cann/'include'),str(source),'-o',str(output/'native_stream_trace.so'),'-ldl']
    result=subprocess.run(command,text=True,capture_output=True)
    (output/'build.json').write_text(json.dumps(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
    result.check_returncode()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cann',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'))
    a=p.parse_args();build(a.output,a.cann)
