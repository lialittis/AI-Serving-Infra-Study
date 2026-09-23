"""Generate a phase comparison and evidence-based diagrams from validated data."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def make_dot(result, phase):
    p = phase
    name, tokens = p["phase"], p["tokens"]
    chains = [c for c in result["operator_chains"] if c["phase"] == name]
    kv, fia = chains[:2]
    cache = p["context"]["kv_cache"]
    out = p["attention_inputs"]["output"]
    lines = ['digraph attention {', 'rankdir=TB; graph [bgcolor="white", nodesep=0.4,ranksep=0.5];',
             'node [shape=box,style="rounded,filled",fillcolor="#e7f1ff",fontname="sans-serif",fontsize=11];',
             'edge [fontname="sans-serif",fontsize=10];']

    def node(key, label, color="#e7f1ff"):
        lines.append('%s [label=%s,fillcolor="%s"];' % (key, json.dumps(label), color))

    def edge(a, b, label, dashed=False):
        lines.append('%s -> %s [label=%s%s];' % (a, b, json.dumps(label),
                     ',style=dashed,color="#ad5b00"' if dashed else ''))

    node("producer", "%s: submod_0\n%s\nQ [%s,14,64]; K/V [%s,2,64]" % (
        name, "compiled callable" if name == "prefill" else "ACL graph replay", tokens, tokens))
    node("write", "KV write\nReshapeAndCacheNdKernel\nstream %s / task %s" % (
        kv["kernel"]["args"]["Physic Stream Id"], kv["kernel"]["args"]["Task Id"]))
    node("cache", "KV cache %s\nK storage %s\nV storage %s" % (
        cache[0]["shape"], hex(cache[0]["storage_ptr"]), hex(cache[1]["storage_ptr"])), "#e3f5e5")
    node("metadata", "Attention metadata\nblock table %s; slots %s\nCPU table: %s" % (
        p["context"]["metadata"]["block_tables"]["shape"],
        p["context"]["metadata"]["slot_mapping"]["shape"], p["host_blocks"]["rows"][0]), "#f0eefb")
    node("fia", "FusedInferAttentionScore\nKV length %s\nstream %s / task %s" % (
        p["fia_inputs"]["actual_seq_lengths_kv"][0], fia["kernel"]["args"]["Physic Stream Id"],
        fia["kernel"]["args"]["Task Id"]))
    node("output", "Shared output buffer [%s,14,64]\nstorage %s" % (tokens, hex(out["storage_ptr"])), "#e3f5e5")
    consumer = "submod_2: view -> O projection\ninput storage verified\n"
    consumer += ("Python call + Matmul task %s verified" % chains[2]["kernel"]["args"]["Task Id"]
                 if name == "prefill" else "ACL replay verified\nindividual kernel / FX match NOT established")
    node("consumer", consumer, "#e7f1ff" if name == "prefill" else "#fff0d4")
    edge("producer", "write", "current K/V")
    edge("metadata", "write", "slot mapping tensor")
    edge("write", "cache", "writes")
    edge("producer", "fia", "current Q/K/V" if name == "prefill" else "current Q")
    if name == "decode":
        edge("cache", "fia", "cached K/V views")
        edge("metadata", "fia", "block table + lengths")
    edge("write", "fia", "device order verified on same stream", True)
    edge("fia", "output", "result copied into caller's buffer")
    edge("output", "consumer", "same pointer / offset; view shares storage")
    return "\n".join(lines + ['}']) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    output = args.run / "analysis"
    result = json.loads((output / "attention_evidence.json").read_text())
    for phase in result["phases"]:
        stem = phase["phase"]
        (output / (stem + ".dot")).write_text(make_dot(result, phase))
        if shutil.which("dot"):
            subprocess.run(["dot", "-Tsvg", str(output / (stem + ".dot")),
                            "-o", str(output / (stem + ".svg"))], check=True)
    template = Path(__file__).with_name("viewer_template.html").read_text()
    payload = json.dumps(result, ensure_ascii=False).replace("<", "\\u003c")
    (output / "attention_viewer.html").write_text(template.replace("__EVIDENCE__", payload))
    print("Generated prefill/decode diagrams and offline attention viewer")


if __name__ == "__main__":
    main()
