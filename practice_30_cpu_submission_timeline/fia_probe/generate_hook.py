"""Generate forwarding wrappers from the frozen, installed CANN declarations."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
headers='\n'.join((ROOT/'sources'/n).read_text() for n in ('acl_rt.h','aclnn_fused_infer_attention_score_v3.h'))
specs={
 'aclnnFusedInferAttentionScoreV3GetWorkspaceSize':('aclnnStatus', '''
 r.u[0]=workspaceSize?*workspaceSize:0;r.u[1]=executor?(uintptr_t)*executor:0;
 r.u[2]=numHeads;r.u[3]=numKeyValueHeads;r.u[4]=preTokens;r.u[5]=nextTokens;
 r.u[6]=sparseMode;r.u[7]=innerPrecise;r.u[8]=blockSize;r.u[9]=antiquantMode;
 r.u[10]=softmaxLseFlag;r.u[11]=keyAntiquantMode;r.u[12]=valueAntiquantMode;
 r.scalar=scaleValue;snprintf(r.text,sizeof(r.text),"%s",inputLayout);
 r.u[13]=(uintptr_t)query;r.u[14]=(uintptr_t)key;r.u[15]=(uintptr_t)value;
 r.u[16]=(uintptr_t)attenMaskOptional;r.u[17]=(uintptr_t)blockTableOptional;
 r.u[18]=(uintptr_t)attentionOut;r.u[19]=(uintptr_t)softmaxLse;
 ''', ''),
 'aclnnFusedInferAttentionScoreV3':('aclnnStatus','r.u[0]=(uintptr_t)workspace;r.u[1]=workspaceSize;r.u[2]=(uintptr_t)executor;r.u[3]=(uintptr_t)stream;',''),
 'aclrtBinaryLoadFromFile':('aclError','snprintf(r.text,sizeof(r.text),"%s",binPath);r.u[0]=binHandle?(uintptr_t)*binHandle:0;',''),
 'aclrtBinaryLoadFromData':('aclError','r.u[0]=binHandle?(uintptr_t)*binHandle:0;r.u[1]=length;',
     'if(r.scope && data && length){unsigned char h[32];SHA256((const unsigned char*)data,length,h);for(int i=0;i<32;i++)snprintf(r.text+2*i,3,"%02x",h[i]);}'),
 'aclrtBinaryGetFunction':('aclError','r.u[0]=(uintptr_t)binHandle;r.u[1]=funcHandle?(uintptr_t)*funcHandle:0;snprintf(r.text,sizeof(r.text),"%s",kernelName);',''),
 'aclrtBinaryGetFunctionByEntry':('aclError','r.u[0]=(uintptr_t)binHandle;r.u[1]=funcHandle?(uintptr_t)*funcHandle:0;r.u[2]=funcEntry;',''),
 'aclrtLaunchKernelWithHostArgs':('aclError','r.u[0]=(uintptr_t)funcHandle;r.u[1]=numBlocks;r.u[2]=(uintptr_t)stream;r.u[3]=argsSize;r.u[4]=placeHolderNum;',''),
}
definitions=[];resolutions=[]
for name,(ret,after,before) in specs.items():
 match=re.search(r'\b'+name+r'\s*\((.*?)\)\s*;',headers,re.S)
 assert match,name
 signature=match.group(1)
 names=[re.search(r'(\w+)\s*$',p.strip()).group(1) for p in signature.split(',')]
 definitions.append(f'''static decltype(&{name}) real_{name};
extern "C" {ret} {name}({signature}) {{
 if(!real_{name})real_{name}=(decltype(real_{name}))resolve(RTLD_NEXT,"{name}");
 if(!real_{name})abort();Row r=begin("{name}");{before}
 auto result=real_{name}({','.join(names)});r.end=now();r.result=result;
 {after}record(r);return result;
}}
''')
 resolutions.append(f'if(!strcmp(name,"{name}") && p){{real_{name}=(decltype(real_{name}))p;return (void*)&{name};}}')
(ROOT/'generated.inc').write_text('\n'.join(definitions)+'''
extern "C" void *dlsym(void *handle,const char *name) noexcept {
 void *p=resolve(handle,name);
'''+ '\n'.join(resolutions)+'''
 return p;
}
''')
