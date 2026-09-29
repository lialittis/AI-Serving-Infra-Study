"""Numerical investigation helpers; diagnostics never contribute timing samples."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import setup,make_case,prepare,MODEL
from run_experiment import snapshot,compare


def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(torch,a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    a=a.detach().cpu().double();b=b.detach().cpu().double();d=(a-b).abs();flat=d.flatten();i=int(flat.argmax())
    return dict(shape=list(a.shape),exact=bool(torch.equal(a,b)),different=int((a!=b).sum()),elements=a.numel(),
        max_abs=float(flat[i]),max_index=i,reference_at_max=float(a.flatten()[i]),actual_at_max=float(b.flatten()[i]),
        relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-30)),rmse=float((a-b).square().mean().sqrt()),
        original_tolerance_pass=bool(torch.allclose(a,b,atol=.0625,rtol=.02)),finite=bool(torch.isfinite(b).all()))


def provenance(torch,torch_npu,transformers,model,out):
    import datetime,inspect,platform,subprocess
    from transformers.models.qwen2 import modeling_qwen2
    from transformers import cache_utils,modeling_rope_utils
    (out/'device_before.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))
    manifest={}
    paths=[Path(m.__file__) for m in (modeling_qwen2,cache_utils,modeling_rope_utils)]+[Path(inspect.getfile(torch.npu.Stream)),ROOT/'model.py',ROOT/'run_experiment.py']+list(Path(__file__).parent.glob('*.py'))
    for path in paths:
        dest=out/'sources'/path.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(path.read_bytes());manifest[path.name]=dict(original=str(path),sha256=sha(path))
    save(out/'sources.json',manifest)
    save(out/'environment.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,torch_npu=torch_npu.__version__,
        transformers=transformers.__version__,python=platform.python_version(),model=MODEL,device=str(torch.npu.get_device_properties(0)),
        config=model.config.to_dict(),checkpoint={p.name:sha(p) for p in Path(MODEL).glob('*.safetensors')}))
