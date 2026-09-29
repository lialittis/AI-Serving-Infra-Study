"""Initialize the installed Qwen2 MoE block and Ascend backend, without checkpoints."""
import json
from pathlib import Path


def setup(output: Path):
    import torch
    import torch_npu
    from transformers import Qwen2MoeConfig
    from vllm_ascend.utils import adapt_patch
    adapt_patch(is_global_patch=True)
    adapt_patch()
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.utils import register_ascend_customop, enable_custom_op, set_weight_prefetch_method
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.bfloat16)
    hf = Qwen2MoeConfig(hidden_size=1024, intermediate_size=2048,
        moe_intermediate_size=512, shared_expert_intermediate_size=1024,
        num_experts=8, num_experts_per_tok=2, norm_topk_prob=True,
        num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=4,
        max_position_embeddings=4096, vocab_size=1024,
        architectures=['Qwen2MoeForCausalLM'], torch_dtype='bfloat16')
    model_dir=output/'config';hf.save_pretrained(model_dir)
    config=VllmConfig(model_config=ModelConfig(model=str(model_dir),
        tokenizer='/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct',
        skip_tokenizer_init=True, enforce_eager=True, dtype='bfloat16', max_model_len=4096),
        additional_config={'multistream_overlap_shared_expert':True,
                           'multistream_overlap_gate':False})
    ctx=set_current_vllm_config(config);ctx.__enter__()
    ascend_config=init_ascend_config(config)
    set_weight_prefetch_method(ascend_config.weight_prefetch_config)
    init_distributed_environment(world_size=1,rank=0,local_rank=0,
        distributed_init_method='file://'+str((output/'distributed-init').resolve()),backend='hccl')
    initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
    enable_custom_op();register_ascend_customop(config)
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeSparseMoeBlock
    from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
    with torch.device('npu'):
        layer=Qwen2MoeSparseMoeBlock(hf,prefix='model.layers.0.mlp')
    assert isinstance(layer.experts,AscendFusedMoE),type(layer.experts)
    generator=torch.Generator(device='cpu').manual_seed(20260928)
    weights={}
    with torch.no_grad():
        for name,p in layer.named_parameters():
            value=(torch.randn(p.shape,generator=generator,device='cpu',dtype=torch.float32)*0.02).to(p.dtype)
            p.copy_(value);weights[name]=value
        for _,module in layer.named_modules():
            method=getattr(module,'quant_method',None)
            if method is not None:
                method.process_weights_after_loading(module)
    layer.eval();torch.npu.synchronize()
    return torch,torch_npu,config,ctx,layer,weights


def cleanup(ctx):
    from vllm.distributed import destroy_model_parallel,destroy_distributed_environment
    destroy_model_parallel();destroy_distributed_environment();ctx.__exit__(None,None,None)
