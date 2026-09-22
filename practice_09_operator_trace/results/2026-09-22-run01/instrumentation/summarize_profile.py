"""List exported profiler artifacts; detailed analysis follows the real schema."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    files = [{"path": str(p.relative_to(args.run)), "bytes": p.stat().st_size}
             for p in sorted((args.run / "profiler").rglob("*")) if p.is_file()]
    if not files:
        raise ValueError("profiler exported no files")
    (args.run / "profiler_files.json").write_text(json.dumps(files, indent=2) + "\n")
    print("Profiler artifacts:", len(files), "files;", sum(p["bytes"] for p in files), "bytes")


if __name__ == "__main__":
    main()
