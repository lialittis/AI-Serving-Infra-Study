"""Pure Python integer oracle. No torch arithmetic, dtype narrowing or device reads."""
MIN32, MAX32 = -(1 << 31), (1 << 31) - 1
CHECKS = {
    'pre_int32': '转换前所有边界在 int32 范围',
    'pre_origin': '转换前首边界为 0',
    'pre_increasing': '转换前边界严格递增',
    'pre_terminal': '转换前末边界等于实际序列长度',
    'source_exact': '转换前边界等于独立参考',
    'device_boundary_exact': '设备边界等于转换前 Python 整数',
    'positive': '输出长度全部为正',
    'count': '输出段数等于参考边界数减 1',
    'sum': 'Python 整数长度总和等于序列长度',
    'exact': '输出逐元素等于独立参考',
    'input_unchanged': '父存储（含 guard）未改写',
    'layout': '实际 view 与预期 offset/stride/范围一致',
    'no_output_alias': '输出不与父存储重叠',
}


def prefix(lengths):
    out = [0]
    for n in lengths:
        assert type(n) is int
        out.append(out[-1] + n)
    return out


def grid_lengths(grid, *, full=False, window=8):
    """Non-padding enumeration of real patch extents in temporal/window order."""
    out=[]
    for t,h,w in grid:
        assert all(type(x) is int and x>0 for x in (t,h,w))
        assert h%2==0 and w%2==0
        if full:
            out.extend([h*w]*t)
        else:
            for _ in range(t):
                for y in range(0,h,window):
                    for x in range(0,w,window):
                        out.append(min(window,h-y)*min(window,w-x))
    return out


def checks(case, observed, device_boundaries, memory):
    raw=case['boundaries'];ref=case['reference'];seq=case['seq']
    expected=[b-a for a,b in zip(ref,ref[1:])]
    assert all(type(x) is int for x in raw+ref+observed)
    return dict(pre_int32=all(MIN32<=x<=MAX32 for x in raw),pre_origin=bool(raw) and raw[0]==0,
        pre_increasing=len(raw)>=2 and all(a<b for a,b in zip(raw,raw[1:])),
        pre_terminal=bool(raw) and raw[-1]==seq,source_exact=raw==ref,
        device_boundary_exact=device_boundaries==raw,positive=all(x>0 for x in observed),
        count=len(observed)==len(ref)-1,sum=sum(observed)==seq,exact=observed==expected,**memory)


def make_cases(preparation):
    cases=[]
    def add(name,label,raw=None,ref=None,seq=10,**kw):
        ref=[0,3,8,10] if ref is None else ref
        cases.append(dict(id=name,label=label,boundaries=list(ref if raw is None else raw),
            reference=ref,seq=seq,pad=0,stride=1,operation='sub',mutation=None,expect_valid=False,**{}))
        cases[-1].update(kw)
    add('baseline','正常差分',expect_valid=True)
    add('offset','合法非零 offset：前置 3 元素',pad=3,expect_valid=True)
    add('stride2','合法 stride=2 的 view',pad=2,stride=2,expect_valid=True)
    add('single','单个长度 1',ref=[0,1],seq=1,expect_valid=True)
    add('int32_limit','合法 int32 最大累计边界',ref=[0,MAX32-1,MAX32],seq=MAX32,expect_valid=True)
    add('swap','互换 Sub 操作数',operation='swap')
    add('reverse_lengths','颠倒输出顺序',mutation='reverse')
    add('single_error','单个长度加 1',mutation='increment')
    add('same_sum','总和不变的错误分段',mutation='redistribute')
    add('single_bit','软件注入一个数值 bit 改动',mutation='bit')
    add('wrong_count','段数减少但总和正确',mutation='merge')
    add('duplicate','未去重的重复边界',raw=[0,3,3,10])
    add('changed_boundary','合法形状但语义错误的边界',raw=[0,4,8,10])
    add('wrong_origin','边界整体平移 2，差分不变',raw=[2,5,10,12])
    add('reversed_boundary','颠倒累计边界顺序',raw=[10,8,3,0])
    add('overflow_cast','累计边界超 int32，差分可能仍正确',ref=[0,MAX32,MAX32+2],seq=MAX32+2)
    add('overflow_sub','int32 Sub 正向溢出',ref=[MIN32,MAX32],seq=MAX32-MIN32)
    add('underflow_sub','int32 Sub 负向溢出',ref=[MAX32,MIN32],seq=MIN32-MAX32)
    add('wrap_origin','边界加 2^32 后窄化，差分与基线相同',raw=[(1<<32)+x for x in [0,3,8,10]])
    seen=set()
    for p in preparation:
        grid=p['grid'];key=str(grid)
        if key in seen:continue
        seen.add(key)
        for full in (False,True):
            sizes=grid_lengths(grid,full=full)
            assert sizes==p['lengths'][str(full)],'Independent enumeration disagrees with published P27'
            seq=sum(t*h*w for t,h,w in grid)
            add(f'grid_{len(seen)}_{"full" if full else "window"}',f'P27 grid={grid} / {"full" if full else "window"}',
                ref=prefix(sizes),seq=seq,grid=grid,full=full,expect_valid=True)
    for name,grid in [('exact_window',[[1,8,8]]),('edge_window',[[1,10,10]]),('two_frames',[[2,10,10]])]:
        sizes=grid_lengths(grid)
        add(name,f'窗口边界样例 {grid}',ref=prefix(sizes),seq=sum(t*h*w for t,h,w in grid),grid=grid,full=False,expect_valid=True)
    return cases
