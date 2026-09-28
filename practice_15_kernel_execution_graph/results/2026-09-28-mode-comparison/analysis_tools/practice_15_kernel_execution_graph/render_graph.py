"""Render a standalone, offline execution-graph explorer."""
import json
from pathlib import Path


def render(graph, destination):
    template = Path(__file__).with_name('viewer.html').read_text()
    payload = json.dumps(graph, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    destination.write_text(template.replace('__GRAPH_DATA__', payload))
