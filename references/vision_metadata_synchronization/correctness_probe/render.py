"""Build one offline HTML from checked raw evidence, without external dependencies."""
import json
from pathlib import Path
from analyze import ROOT,main as analyze,read


def main():
    analyze()
    data=dict(summary=read(ROOT/'results/summary.json'),cases=read(ROOT/'results/run-r01/cases.json'))
    template=(ROOT/'template.html').read_text()
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    out=ROOT/'report';out.mkdir(exist_ok=True)
    (out/'index.html').write_text(template.replace('__PAYLOAD__',payload))
    print(out/'index.html')


if __name__=='__main__':main()
