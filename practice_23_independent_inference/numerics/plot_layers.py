"""Standalone scientific figure: growth of native BF16 batch-shape differences."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('layers',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    records=json.loads(a.layers.read_text())['records'];plt.rcParams['svg.hashsalt']='p23-numerics'
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for task,color in [('A','#208678'),('B','#357dba')]:
        rs=[r for r in records if r['task']==task and r['module'].startswith('model.layers.')]
        xs=[int(r['module'].split('.')[-1]) for r in rs]
        for ax,key in zip(axes,['max_abs','relative_l2']):
            ax.plot(xs,[r[key] for r in rs],label='Task '+task,color=color,marker='o',markersize=3,linewidth=1.6)
            ax.set_yscale('log');ax.set_xlabel('Decoder layer (0-based)');ax.grid(alpha=.2);ax.set_xticks([0,4,8,12,16,20,23])
    axes[0].set_ylabel('Maximum absolute difference');axes[1].set_ylabel('Relative L2 difference');axes[0].legend()
    fig.suptitle('Native BF16: batch=2 vs two batch=1 forwards | 1024-token prefill')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output.with_suffix('.svg'),metadata={'Date':None});fig.savefig(a.output.with_suffix('.png'),dpi=160)
    svg=a.output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
