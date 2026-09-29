"""Qualify native complete multimodal forward against separated real stages."""
import argparse,traceback
from pathlib import Path
from model import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    try:
        torch,tn,tr,model,processor=setup();checks=[];streams={n:torch.npu.Stream() for n in ['L','V']}
        with torch.inference_mode():
            for name in ['beach.jpeg','beijing.jpeg']:
                req=prepare_request(torch,model,processor,name,64)
                full=model(input_ids=req['ids'],pixel_values=req['pixel_values'],image_grid_thw=req['grid'].to('npu'),mm_token_type_ids=req['mm_types'],attention_mask=req['attention'],use_cache=True,logits_to_keep=1)
                torch.npu.synchronize();ref=snapshot(full);native_delta=model.model.rope_deltas.detach().cpu();assert torch.equal(native_delta,req['delta'])
                model.model.rope_deltas=None
                f=vision(model,req);e=merge(model,req,f);split=language(model,e,req['position'],req['mask'],new_cache(model));torch.npu.synchronize()
                c=compare(torch,ref,snapshot(split));assert c['exact'] and c['finite'],c
                checks.append(dict(image=name,full_vs_split=c,grid=req['grid'].tolist(),visual_tokens=f.shape[0],next_token=processor.tokenizer.decode(split.logits.argmax(-1)[0])))
                save(out/'full_vs_split.json',checks)
            for phase in ['prefill','decode']:
                case=make_case(torch,model,processor,'beach.jpeg',64,phase)
                result,_=execute(torch,model,case,'serial',streams);ref=result_snapshot(result)
                for order in ['LV','VL']:
                    result,metric=execute(torch,model,case,'parallel',streams,order)
                    c=check_result(torch,ref,result);save(out/f'{phase}-{order}.json',dict(checks=c,metric=metric))
                from run import native
                c=compare(torch,native(torch,model,case['A'],decode=(phase=='decode')),ref['outputs']['A'])
                assert c['exact'] and c['finite'] and c['greedy_equal'],c
                save(out/f'{phase}-native-A.json',c)
                print('QUALIFIED',phase,flush=True)
            save(out/'completed.json',dict(status='passed',torch=torch.__version__,torch_npu=tn.__version__,transformers=tr.__version__))
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
