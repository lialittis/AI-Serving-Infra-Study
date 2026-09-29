"""Request-local shape metadata; no installed-package changes or feature caching."""
import contextlib
import hashlib
import inspect
import textwrap
import time
import types
from pathlib import Path

SOURCE_SHA = '9d6d15040bdb985d9518117ed029f6a318a38abc06091462d96f16cf1d3d7820'
VARIANTS = ('native', 'lengths', 'cached')


def checked_attention(module):
    path = Path(inspect.getfile(module))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA, 'Re-audit installed vision source'
    source = textwrap.dedent(inspect.getsource(module.Qwen2_5_VLVisionAttention.forward))
    old = 'lengths = cu_seqlens[1:] - cu_seqlens[:-1]'
    assert source.count(old) == 1 and source.count('lengths.tolist()') == 1
    source = source.replace(old, 'lengths = self._p27_lengths').replace('lengths.tolist()', 'lengths')
    namespace = dict(vars(module))
    exec(compile(source, '<p27-source-pinned-attention>', 'exec'), namespace)
    return namespace['forward'], source


def prepare(torch, model, request):
    """Cost is measured separately, after inputs exist and before timed execution."""
    visual = model.model.visual
    grid = request['grid']
    assert grid.device.type == 'cpu' and len(grid) == 1
    torch.npu.synchronize()
    begin = time.perf_counter_ns()
    window, cumulative = visual.get_window_index(grid)
    cumulative = torch.unique_consecutive(torch.tensor(cumulative, dtype=torch.int32))
    full = torch.nn.functional.pad(torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(0, dtype=torch.int32), (1, 0))
    lengths = {False: tuple((cumulative[1:] - cumulative[:-1]).tolist()), True: tuple((full[1:] - full[:-1]).tolist())}
    cpu_us = (time.perf_counter_ns() - begin) / 1000
    rotary = visual.rot_pos_emb(grid)
    seq = int(grid.prod(-1).sum())
    rotary = rotary.reshape(seq // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)[window].reshape(seq, -1)
    emb = torch.cat((rotary, rotary), dim=-1)
    positions = (emb.cos(), emb.sin())
    window_npu = window.to('npu')
    reverse_npu = torch.argsort(window).to('npu')
    torch.npu.synchronize()
    return dict(grid=grid.tolist(), pixels_shape=list(request['pixel_values'].shape), lengths=lengths,
                positions=positions, window=window_npu, reverse=reverse_npu, seq=seq,
                preparation=dict(cpu_lengths_us=cpu_us, total_us=(time.perf_counter_ns()-begin)/1000,
                    persistent_device_bytes=sum(t.numel()*t.element_size() for t in (*positions,window_npu,reverse_npu)),
                    lengths={str(k):list(v) for k,v in lengths.items()}))


def cached_vision(model, request, meta):
    """Keep patch projection, all 32 blocks and merger live; only metadata is reused."""
    visual = model.model.visual
    assert request['grid'].tolist() == meta['grid']
    assert list(request['pixel_values'].shape) == meta['pixels_shape']
    hidden = visual.patch_embed(request['pixel_values'].type(visual.dtype))
    hidden = hidden.reshape(meta['seq']//visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
    hidden = hidden[meta['window']].reshape(meta['seq'], -1)
    for block in visual.blocks:
        hidden = block(hidden, cu_seqlens=None, position_embeddings=meta['positions'])
    return visual.merger(hidden)[meta['reverse']]


@contextlib.contextmanager
def use_variant(base, model, request, variant, meta, attention):
    """Only this experiment's model instances are patched, always restored on exit."""
    assert variant in VARIANTS
    assert request['grid'].tolist() == meta['grid'] and list(request['pixel_values'].shape) == meta['pixels_shape']
    visual = model.model.visual
    assert visual.config._attn_implementation == 'eager' and not visual.training
    old_vision = base.vision
    originals = []
    try:
        if variant != 'native':
            for index, block in enumerate(visual.blocks):
                a = block.attn
                assert '_p27_lengths' not in a.__dict__
                originals.append((a, a.__dict__.get('forward')))
                a._p27_lengths = meta['lengths'][index in visual.fullatt_block_indexes]
                a.forward = types.MethodType(attention, a)
        if variant == 'cached':
            base.vision = lambda m,r: cached_vision(m,r,meta)
        yield
    finally:
        base.vision = old_vision
        for a, original in originals:
            del a._p27_lengths
            if original is None:
                del a.forward
            else:
                a.forward = original
