"""One warmed graph-external attention; original methods and arguments preserved."""
from contextlib import contextmanager
import functools
import inspect

from common import Bindings


class AttentionObserver:
    def __init__(self, observer):
        import torch_npu
        import vllm.model_executor.layers.attention.attention as attention
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        self.observer = observer
        self.active = False
        self.selected = 0
        self.context_calls = 0
        self.entries = []
        self.metadata = dict(target_step=32, target_attention_ordinal=0,
            limits='One warmed graph-external attention. Shape/host metadata only; no tensor contents, extra waits, stream changes or changed arguments. Wrappers/profiler perturb timings.')
        self.add(attention, 'get_attention_context', 'attn_context', context=True)
        cls = AscendAttentionBackendImpl
        for name, kind in [('forward', 'attn_backend'),
                           ('reshape_and_cache', 'attn_kv_prepare'),
                           ('forward_impl', 'attn_dispatch'),
                           ('forward_fused_infer_attention', 'attn_fia'),
                           ('_get_fia_params', 'attn_fia_params')]:
            self.add(cls, name, kind, outer=name == 'forward')
        self.add(torch_npu, '_npu_reshape_and_cache', 'attn_kv_submit')
        self.add(torch_npu, 'npu_fused_infer_attention_score', 'attn_fia_submit')

    @staticmethod
    def tensor_meta(value):
        if value is None:
            return None
        return dict(shape=list(value.shape), stride=list(value.stride()),
                    dtype=str(value.dtype), device=str(value.device))

    def add(self, owner, name, kind, outer=False, context=False):
        original = getattr(owner, name)
        fn = inspect.unwrap(original)
        try:
            source = dict(path=inspect.getsourcefile(fn),
                          line=inspect.getsourcelines(fn)[1], qualname=fn.__qualname__)
        except (OSError, TypeError):
            source = dict(path=None, line=None, qualname=getattr(fn, '__qualname__', name),
                          source_role='exported torch-npu callable boundary; native body not timed separately')
        self.observer.sources[kind] = source

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if context:
                if self.observer.index != 32 or self.context_calls or self.selected:
                    return original(*args, **kwargs)
                self.context_calls += 1
                with self.observer.phase(kind):
                    result = original(*args, **kwargs)
                self.metadata['context_layer'] = str(args[0])
                return result
            chosen = outer and self.observer.index == 32 and self.selected == 0
            if outer and not chosen:
                return original(*args, **kwargs)
            if not outer and not self.active:
                return original(*args, **kwargs)
            if chosen:
                self.selected += 1
                self.active = True
            try:
                if chosen:
                    impl, layer, query, key, value, kv_cache, meta = args[:7]
                    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
                    # Metadata inspection is outside the inner timed method, but
                    # still perturbs its enclosing custom-op/profiler scope.
                    self.metadata['call'] = dict(layer_name=layer.layer_name,
                        implementation=type(impl).__qualname__, attn_state=str(meta.attn_state),
                        capturing=bool(_EXTRA_CTX.capturing), causal=bool(meta.causal),
                        sinks=impl.sinks is not None, sliding_window=impl.sliding_window,
                        hamming_sparse=bool(impl.enable_hamming_sparse),
                        is_kv_producer=bool(impl.is_kv_producer),
                        cache_already_initialized=impl.key_cache is not None,
                        num_actual_tokens=int(meta.num_actual_tokens),
                        actual_seq_lengths_q=list(meta.actual_seq_lengths_q),
                        seq_lens_list=list(meta.seq_lens_list),
                        query=self.tensor_meta(query), key=self.tensor_meta(key),
                        value=self.tensor_meta(value), output=self.tensor_meta(kwargs.get('output')),
                        block_tables=self.tensor_meta(meta.block_tables),
                        slot_mapping=self.tensor_meta(meta.slot_mapping))
                    assert not _EXTRA_CTX.capturing, 'selected attention must be outside capture'
                if kind == 'attn_fia_submit':
                    self.metadata['fia_arguments'] = {
                        k: self.tensor_meta(v) if hasattr(v, 'shape') else v
                        for k, v in kwargs.items()}
                with self.observer.phase(kind):
                    return original(*args, **kwargs)
            finally:
                if chosen:
                    self.active = False
        self.entries.append((owner, name, wrapped))

    @contextmanager
    def installed(self):
        previous = [(o, n, n in vars(o), getattr(o, n)) for o, n, _ in self.entries]
        try:
            with Bindings(self.entries):
                yield self
        finally:
            self.active = False
            self.restored = all((n in vars(o)) == owned and getattr(o, n) == value
                                for o, n, owned, value in previous)
            self.metadata.update(selected_calls=self.selected, context_calls=self.context_calls,
                                 restored=self.restored)
            assert self.restored, 'attention bindings not restored'
