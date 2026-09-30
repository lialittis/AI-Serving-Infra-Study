"""One warmed RoPE call in decode 32; instance wrappers, no new device waits."""
from contextlib import contextmanager
import functools
import inspect

from common import Bindings


class ForwardObserver:
    def __init__(self, observer):
        import vllm_ascend.ops.rotary_embedding as rotary
        import vllm_ascend.ops.triton.rope as rope
        self.observer = observer
        self.jit = rope._triton_rope
        self.active = False
        self.selected = 0
        self.entries = []
        self.cache_before = self.cache_inventory()
        assert self.cache_before, 'RoPE must be warmed before instrumentation'
        self.add(rotary, 'rope_forward_triton', 'rope_python', outer=True)
        self.add(self.jit, 'run', 'rope_jit')
        self.add(self.jit, 'binder', 'rope_bind')
        self.add(self.jit, '_do_compile', 'rope_compile')
        kernels = {id(k): k for cache in self.jit.cache.values() for k in cache.values()}
        self.launchers = []
        for kernel in kernels.values():
            # Do not initialize a new handle while observing: cached variants
            # must already have loaded handles from initialization/warmup.
            assert kernel.module is not None, 'cached kernel not loaded'
            original = kernel.run
            self.launchers.append(dict(name=kernel.name,hash=kernel.hash,
                launcher_type=f'{type(original).__module__}.{type(original).__qualname__}',
                native_body_source='not captured; only callable boundary and profiler events'))
            self.add(kernel, 'run', 'rope_native', native=True)
        self.metadata = dict(target_step=32,target_rope_ordinal=0,
            cache_before=self.cache_before,launchers=self.launchers,
            limits='Only one warmed eager RoPE invocation. Timings include observation cost; native internals are not Python stacks.')

    def cache_inventory(self):
        return sorted((str(device),str(key),kernel.hash)
                      for device,cache in self.jit.cache.items() for key,kernel in cache.items())

    def add(self, owner, name, kind, outer=False, native=False):
        original = getattr(owner,name)
        if native:
            # The Python callsite is available, but the native body is not.
            fn = inspect.unwrap(self.jit.run)
            lines,start = inspect.getsourcelines(fn)
            line = next(start+i for i,s in enumerate(lines) if 'kernel.run(' in s)
            source = dict(path=inspect.getsourcefile(fn),line=line,
                qualname='JITFunction.run → cached kernel.run (native callable)',
                source_role='Python callsite, not native implementation')
        else:
            fn = inspect.unwrap(original)
            try:source=dict(path=inspect.getsourcefile(fn),line=inspect.getsourcelines(fn)[1],qualname=fn.__qualname__)
            except (OSError,TypeError):source=dict(path=None,line=None,qualname=getattr(fn,'__qualname__',type(fn).__qualname__),source_role='generated callable; no source text')
        self.observer.sources[kind] = source

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            chosen = outer and self.observer.index==32 and self.selected==0
            if outer and not chosen:return original(*args,**kwargs)
            if not outer and not self.active:return original(*args,**kwargs)
            extra = {}
            if chosen:
                self.selected += 1
                self.active = True
                # Shape/stride metadata only; no data_ptr reads, CPU copies or
                # data-dependent tensor operations are introduced.
                q,k=args[:2]
                extra=dict(q_shape=list(q.shape),k_shape=list(k.shape),
                    q_stride=list(q.stride()),k_stride=list(k.stride()))
            if kind=='rope_native':extra['grid']=list(args[:3])
            try:
                with self.observer.phase(kind,extra):return original(*args,**kwargs)
            finally:
                if chosen:self.active=False
        self.entries.append((owner,name,wrapped))

    @contextmanager
    def installed(self):
        previous=[(o,n,n in vars(o),getattr(o,n)) for o,n,_ in self.entries]
        try:
            with Bindings(self.entries):yield self
        finally:
            self.active=False
            self.restored=all((n in vars(o))==owned and getattr(o,n)==value
                              for o,n,owned,value in previous)
            self.metadata.update(selected_calls=self.selected,restored=self.restored,
                cache_after=self.cache_inventory(),cache_unchanged=self.cache_inventory()==self.cache_before)
            assert self.restored,'RoPE bindings not restored'
            assert self.metadata['cache_unchanged'],'RoPE compiled cache changed'
