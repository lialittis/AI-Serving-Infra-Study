"""Verify exported artifacts and replay every kernel-graph analysis offline."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from analyze_run import analyze


def sha256_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def verify(run):
    manifest = json.loads((run / "artifact_manifest.json").read_text())
    for name, metadata in manifest["files"].items():
        path = run / name
        if path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
            raise ValueError("artifact mismatch: %s" % path)
    trace = next((run / "profiler").rglob("trace_view.json.gz"))
    with gzip.open(trace, "rb") as stream:
        if sha256_stream(stream) != manifest["raw_trace"]["sha256"]:
            raise ValueError("decompressed trace mismatch: %s" % trace)
    result = analyze(run)
    stored = json.loads((run / "analysis" / "summary.json").read_text())
    if {"summary": result["summary"], "steps": result["steps"]} != stored:
        raise ValueError("analysis replay mismatch: %s" % run)
    return result["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    runs = sorted(path for path in args.results.iterdir()
                  if path.is_dir() and (path / "artifact_manifest.json").exists())
    if not runs:
        raise ValueError("no exported runs")
    summaries = {run.name: verify(run) for run in runs}
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
