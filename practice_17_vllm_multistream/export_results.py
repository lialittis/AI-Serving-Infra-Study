"""Create a compact, replayable evidence bundle from full profiler runs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


FILES = (
    "command.json", "device_after.txt", "environment.json", "instrumentation_hashes.json",
    "launcher.json", "profile_control.json", "prompt_info.json", "ready.json", "request.json",
    "request_window.json", "response.json", "server.log", "shutdown.json", "source_manifest.json",
    "warmup_response.json",
)
DIRECTORIES = ("analysis", "events", "instrumentation", "sources")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def only(items, description):
    values = list(items)
    if len(values) != 1:
        raise ValueError("%s: got %d" % (description, len(values)))
    return values[0]


def copy_run(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"source_run": source.name, "files": {}}
    for name in FILES:
        path = source / name
        if path.exists():
            shutil.copy2(path, destination / name)
    for name in DIRECTORIES:
        shutil.copytree(source / name, destination / name)
    profiler = destination / "profiler" / "evidence"
    profiler.mkdir(parents=True)
    trace = only((source / "profiler").rglob("trace_view.json"), "trace")
    trace_gz = profiler / "trace_view.json.gz"
    with trace.open("rb") as reader, gzip.open(trace_gz, "wb", compresslevel=9) as writer:
        shutil.copyfileobj(reader, writer)
    kernel_csv = only((source / "profiler").rglob("kernel_details.csv"), "kernel CSV")
    shutil.copy2(kernel_csv, profiler / "kernel_details.csv")
    manifest["raw_trace"] = {"path": str(trace), "bytes": trace.stat().st_size,
                             "sha256": sha256(trace)}
    manifest["raw_kernel_csv"] = {"path": str(kernel_csv), "bytes": kernel_csv.stat().st_size,
                                  "sha256": sha256(kernel_csv)}
    for path in sorted(p for p in destination.rglob("*") if p.is_file()):
        relative = str(path.relative_to(destination))
        manifest["files"][relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (destination / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in args.runs:
        copy_run((args.results / name).resolve(), args.output / name)


if __name__ == "__main__":
    main()
