"""Export exact BF16 operands for standard-library-only rational replay."""
import argparse,json
from pathlib import Path
from common import save,sha


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();out=a.run/'operands';out.mkdir(exist_ok=False)
    data=torch.load(a.run/'operands.pt',map_location='cpu',weights_only=True);manifest={}
    for name,t in {'inputs':data['inputs'],**data['weights']}.items():
        target=out/(name+'.bf16');target.write_bytes(t.contiguous().view(torch.uint16).numpy().astype('<u2').tobytes())
        manifest[name]=dict(file=target.name,shape=list(t.shape),dtype='bfloat16',byteorder='little',bytes=target.stat().st_size,sha256=sha(target))
    save(out/'manifest.json',dict(source_pt_sha256=sha(a.run/'operands.pt'),tensors=manifest))
    print('EXPORTED',out,flush=True)


if __name__=='__main__':main()
