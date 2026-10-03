"""Preserve exact compact experiment files and hash remote raw profiler data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source, dest = args.run.resolve(), args.destination.resolve()
    if dest.is_relative_to(source) or source.is_relative_to(dest):
        parser.error("source and destination must be separate trees")
    dest.mkdir(parents=True, exist_ok=False)
    retained, raw = [], []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        entry = dict(path=str(relative), bytes=path.stat().st_size, sha256=digest(path))
        if "profiler" in relative.parts:
            raw.append(entry)
        else:
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            retained.append(entry)
    for name in ("analyze.py", "analyze_profile.py", "archive.py"):
        path = Path(__file__).with_name(name)
        target = dest / "analysis_sources" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
        retained.append(dict(path=str(target.relative_to(dest)), bytes=target.stat().st_size, sha256=digest(target)))
    manifest = dict(schema=1, remote_host="ascend910", remote_root=str(source),
                    retained_files=retained, raw_profiler_files=raw,
                    raw_profiler_retained_remote=bool(raw),
                    reproduction="Restore raw_profiler_files at their relative paths before rerunning raw flow analysis. Exact snapshots and run metadata are retained.")
    (dest / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(run=source.name, retained_files=len(retained),
                         retained_bytes=sum(x["bytes"] for x in retained), raw_bytes=sum(x["bytes"] for x in raw))))


if __name__ == "__main__":
    main()
