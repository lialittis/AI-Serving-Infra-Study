"""Native vLLM fixed-step execution, isolated at the Python resource boundary.

Installed source files are never modified. Run only in a disposable child.
"""
import copy
from contextlib import contextmanager
import dataclasses
import hashlib
import inspect
import time
from pathlib import Path

from common import Bindings, disjoint, save

MODEL = '/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct'
STEPS = (16, 32, 47)


def settings(mode):
    config = dict(model=MODEL, dtype='bfloat16', tensor_parallel_size=1,
                  max_model_len=256, max_num_seqs=1, max_num_batched_tokens=256,
                  seed=123, gpu_memory_utilization=0.3, kv_cache_memory_bytes=16 * 1024**2,
                  block_size=128, enable_prefix_caching=False, enable_chunked_prefill=False,
                  async_scheduling=False, distributed_executor_backend='uni',
                  additional_config={'enable_async_exponential': False},
                  disable_log_stats=True)
    if mode == 'eager':
        config['enforce_eager'] = True
    else:
        config['compilation_config'] = dict(mode=3, cudagraph_mode='PIECEWISE',
                                          cudagraph_capture_sizes=[1], custom_ops=['all'])
    return config


def tensors(value, path='root'):
    import torch
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, path + '/' + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensors(item, path + '/' + str(index))
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from tensors(getattr(value, field.name), path + '/' + field.name)


def cpu(value):
    return {name: tensor.detach().cpu().clone() for name, tensor in tensors(value)}


def clone(value):
    # KV views are BF16 views over a byte pool. Tensor.__deepcopy__ tries to
    # reconstruct typed storage and fails on this native layout; clone values
    # explicitly while retaining repeated references via deepcopy's memo.
    memo = {id(t): t.detach().clone() for _, t in tensors(value)}
    return copy.deepcopy(value, memo)


def fingerprint(value):
    return {name: dict(shape=list(t.shape), dtype=str(t.dtype),
                       sha256=hashlib.sha256(t.contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest())
            for name, t in cpu(value).items()}


def compare(actual, expected):
    import torch
    actual = cpu(actual)
    if set(actual) != set(expected):
        raise AssertionError('tensor inventory changed: ' + str(set(actual) ^ set(expected)))
    result = {}
    for name, a in actual.items():
        b = expected[name]
        finite = not a.is_floating_point() or bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        equal = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        diff = float((a.float() - b.float()).abs().max()) if a.numel() and a.shape == b.shape else None
        result[name] = dict(equal=equal, finite=finite, max_abs=diff)
    return dict(passed=all(v['equal'] and v['finite'] for v in result.values()), tensors=result)


def copy_into(dest, source):
    dt, st = dict(tensors(dest)), dict(tensors(source))
    if set(dt) != set(st):
        raise AssertionError('fixed tensor structure changed')
    for name, tensor in dt.items():
        other = st[name]
        if tensor.shape != other.shape or tensor.dtype != other.dtype:
            raise AssertionError('fixed shape/dtype changed: ' + name)
        tensor.copy_(other)


class GlobalState:
    """Own mutable Python globals; never release these while NPU work is live."""
    def __init__(self, fresh=False):
        import torch
        import vllm.v1.worker.workspace as workspace
        import vllm_ascend.ops.rotary_embedding as rope
        import vllm_ascend.compilation.acl_graph as graphs
        import vllm_ascend.utils as utils
        from vllm.platforms import current_platform
        self.entries = [(workspace, '_manager', workspace._manager)]
        self.entries += [(rope, name, getattr(rope, name)) for name in
                         ('_cos_mla', '_sin_mla', '_cos_cache', '_sin_cache', '_cos_sin_cache',
                          '_cos', '_sin', '_cos_slice', '_sin_slice')]
        self.entries += [(graphs, name, getattr(graphs, name)) for name in
                         ('_graph_params', '_draft_graph_params', '_draft_graph_prefill_params')]
        self.entries += [(type(current_platform), '_global_graph_pool', current_platform.get_global_graph_pool()),
                         (utils, '_CURRENT_STREAM', utils._CURRENT_STREAM)]
        if fresh:
            self.entries = [(obj, name, None) for obj, name, value in self.entries]
            self.entries[0] = (workspace, '_manager', workspace.WorkspaceManager(torch.device('npu:0')))
            self.entries[-2] = (type(current_platform), '_global_graph_pool', torch.npu.graph_pool_handle())

    @contextmanager
    def active(self):
        import torch
        entries = [(obj, name, torch.npu.current_stream() if name == '_CURRENT_STREAM' else value)
                   for obj, name, value in self.entries]
        with Bindings(entries):
            try:
                yield
            finally:
                self.entries = [(obj, name, getattr(obj, name)) for obj, name, value in self.entries]

    def mutable_tensors(self):
        for obj, name, value in self.entries:
            if name == '_manager':
                yield from tensors(value._current_workspaces, 'workspace')
            elif name in ('_cos', '_sin', '_cos_mla', '_sin_mla', '_cos_slice', '_sin_slice'):
                yield from tensors(value, name)


def make_engine(mode):
    import torch
    import torch_npu
    from vllm import LLM
    llm = LLM(**settings(mode))
    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    return torch, llm, runner


def prompts(llm):
    tok = llm.get_tokenizer()
    a = tok.encode('请简要解释为什么天空是蓝色的。', add_special_tokens=False)
    b = tok.encode('请解释计算机如何存储和处理数据。', add_special_tokens=False)
    # Persist token IDs; fixed-length input is explicit rather than inferred from text.
    if len(a) != 10:
        raise AssertionError('P26 tokenizer/prompt drift')
    b = (b + tok.encode('谢谢。', add_special_tokens=False))[:10]
    assert len(b) == 10 and a != b
    return {'A': a, 'B': b}


def generate(llm, ids):
    from vllm import SamplingParams
    return llm.generate([{'prompt_token_ids': ids}], SamplingParams(
        temperature=0, max_tokens=64, ignore_eos=True, logprobs=1), use_tqdm=False)[0]


def collect_states(torch, llm, runner, out):
    """Observe actual native forward inputs and sample results, outside timing."""
    from vllm.forward_context import get_forward_context
    snapshots = {}
    ids = prompts(llm)
    outputs = {}
    original_forward, original_sample = runner._model_forward, runner._sample
    signature = inspect.signature(original_forward)
    for role, prompt in ids.items():
        for _, tensor in tensors(runner.kv_caches):
            tensor.zero_()
        torch.npu.synchronize()
        index = -1
        active = None

        def forward(*args, **kwargs):
            nonlocal index, active
            index += 1
            active = None
            if index in STEPS:
                torch.npu.synchronize()
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                arguments = dict(bound.arguments)
                extras = arguments.pop('model_kwargs', {})
                arguments.update(extras)
                context = copy.copy(get_forward_context())
                context.attn_metadata = clone(context.attn_metadata)
                context.slot_mapping = clone(context.slot_mapping)
                active = dict(role=role, index=index, args=clone(arguments),
                              context=context, initial_kv=clone(runner.kv_caches),
                              sampling=clone(runner.input_batch.sampling_metadata),
                              logits_indices=runner.logits_indices.clone())
                snapshots[(role, index)] = active
            return original_forward(*args, **kwargs)

        def sample(logits, spec):
            output = original_sample(logits, spec)
            if active is not None:
                torch.npu.synchronize()
                active['expected'] = cpu(dict(logits=logits, sample=output, kv=runner.kv_caches))
            return output

        with Bindings([(runner, '_model_forward', forward), (runner, '_sample', sample)]):
            response = generate(llm, prompt)
        outputs[role] = list(response.outputs[0].token_ids)
        assert len(outputs[role]) == 64 and index == 63
    save(out / 'native_reference.json', dict(prompts=ids, output_token_ids=outputs,
         states={f'{role}-{step}': dict(index=step, positions=s['args']['positions'].cpu().tolist(),
                 inputs=fingerprint(s['args']), expected=fingerprint(s['expected']))
                 for (role, step), s in snapshots.items()}))
    return snapshots


class Capsule:
    def __init__(self, name, runner, state, snapshot):
        self.name, self.runner, self.state = name, runner, state
        self.args = clone(snapshot['args'])
        # Native capture binds runner-owned persistent input addresses. Reuse
        # these buffers instead of replaying a graph against cloned pointers.
        if self.args.get('input_ids') is not None:
            self.args['input_ids'] = runner.input_ids.gpu[:self.args['input_ids'].shape[0]]
        if self.args.get('positions') is not None:
            self.args['positions'] = runner.positions[:self.args['positions'].shape[0]]
        self.context = copy.copy(snapshot['context'])
        self.context.attn_metadata = clone(snapshot['context'].attn_metadata)
        self.context.slot_mapping = clone(snapshot['context'].slot_mapping)
        self.context.no_compile_layers = runner.compilation_config.static_forward_context
        self.context.model_instance = runner.model
        self.context.input_ids = self.args.get('input_ids')
        self.context.capturing = False
        self.indices = snapshot['logits_indices'].clone()
        self.output = None

    def restore(self, snapshot):
        copy_into(self.args, snapshot['args'])
        copy_into(self.runner.kv_caches, snapshot['initial_kv'])
        # Metadata is graph-external in PIECEWISE; fixed model input addresses stay owned.
        self.context.attn_metadata = clone(snapshot['context'].attn_metadata)
        self.context.slot_mapping = clone(snapshot['context'].slot_mapping)
        self.context.batch_descriptor = snapshot['context'].batch_descriptor
        self.runner.input_batch.sampling_metadata = clone(snapshot['sampling'])
        self.indices.copy_(snapshot['logits_indices'])
        self.context.capturing = False
        with self.state.active():
            from vllm_ascend.ops.rotary_embedding import update_cos_sin
            update_cos_sin(self.args['positions'])

    def submit(self):
        import torch
        from vllm.forward_context import override_forward_context
        from vllm.config import set_current_vllm_config
        with torch.inference_mode(), self.state.active(), set_current_vllm_config(self.runner.vllm_config), override_forward_context(self.context):
            hidden = self.runner._model_forward(**self.args)
            logits = self.runner.model.compute_logits(hidden[self.indices])
            sample = self.runner._sample(logits, None)
            self.output = dict(logits=logits, sample=sample, kv=self.runner.kv_caches)
        return self.output

    def resources(self):
        yield from tensors(self.args, 'inputs')
        yield from tensors(self.runner.kv_caches, 'kv')
        yield from tensors(self.context.attn_metadata, 'metadata')
        yield from tensors(self.context.slot_mapping, 'slot_mapping')
        yield from tensors(self.runner.input_batch.sampling_metadata, 'sampling')
        yield from tensors(self.indices, 'logits_indices')
        yield from tensors({n:t for n,t in self.runner.model.named_buffers()
                            if n not in self.readonly_buffers()}, 'model_buffers')
        yield from self.state.mutable_tensors()
        if self.output:
            yield from tensors(self.output, 'output')

    def readonly_buffers(self):
        # Pinned Qwen path: AscendRotaryEmbedding.forward_oot passes this table
        # to rope_forward_triton; its kernel loads the table and stores Q/K.
        # get_rope's _ROPE_DICT shares the module across runners. Do not reset
        # that cache or alter the model merely to make a read-only alias vanish.
        result = {}
        for name, module in self.runner.model.named_modules():
            if type(module).__name__ == 'AscendRotaryEmbedding':
                result[name + '.cos_sin_cache'] = module.cos_sin_cache
        if not result:
            raise AssertionError('pinned Qwen rotary implementation changed')
        return result


def make_second(mode, first, snapshot):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import set_current_vllm_config
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    config = EngineArgs(**settings(mode)).create_engine_config()
    state = GlobalState(fresh=True)
    with state.active(), set_current_vllm_config(config):
        runner = NPUModelRunner(config, first.device)
        runner.load_model()
        first_params = dict(first.model.named_parameters())
        for name, parameter in runner.model.named_parameters():
            reference = first_params[name]
            assert parameter.shape == reference.shape and parameter.dtype == reference.dtype
            parameter.data = reference.data
        runner.initialize_kv_cache(copy.deepcopy(first.kv_cache_config))
        if mode == 'graph':
            # Use native dummy capture: attention is graph-external for
            # PIECEWISE. Capturing with live attention metadata incorrectly
            # enters the FULL-only task-group update path.
            runner.capture_model()
    return Capsule('B', runner, state, snapshot)


def ranges(capsules):
    result = []
    for capsule in capsules:
        for name, t in capsule.resources():
            if t.device.type == 'cpu' or not t.numel():
                continue
            storage = t.untyped_storage()
            result.append((capsule.name, name, storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
    conflicts = disjoint(result)
    if conflicts:
        raise AssertionError('cross-capsule mutable storage alias: ' + str(conflicts[:10]))
    return result


def pair(torch, capsules, streams, mode, order='AB', observer=None, inject=False):
    origin = torch.npu.Event(enable_timing=True)
    ends = {role: torch.npu.Event(enable_timing=True) for role in order}
    submitted, recorded, values, submitted_at = {}, set(), {}, {}
    from contextlib import nullcontext
    def mark(kind, role=None):
        return observer.marker(kind, role) if observer else nullcontext()
    begin = time.perf_counter_ns()
    with mark('origin'):
        origin.record()
    try:
        for index, role in enumerate(order):
            stream = streams['B'] if mode == 'parallel' and role == 'B' else streams['A']
            submitted[role] = stream  # Before any call that could submit then raise.
            with torch.npu.stream(stream):
                with mark('wait_origin', role):
                    stream.wait_event(origin)
                scope = observer.scope(role) if observer else nullcontext()
                with scope:
                    values[role] = capsules[role].submit()
                with mark('terminal', role):
                    ends[role].record()
                recorded.add(role)
            submitted_at[role] = (time.perf_counter_ns() - begin) / 1000
            if inject and index == 0:
                raise RuntimeError('injected_after_first_submit')
    finally:
        for role, stream in submitted.items():
            with mark('join', role):
                if role in recorded:
                    ends[role].synchronize()
                else:
                    stream.synchronize()
    elapsed = (time.perf_counter_ns() - begin) / 1000
    return values, dict(wall_us=elapsed, ready_us={r: origin.elapsed_time(e) * 1000 for r, e in ends.items()},
                        host_submission_us=submitted_at)
