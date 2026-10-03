"""Build an offline interactive host allocation / device-copy timeline."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    data = []
    for case in json.loads((args.run / "cases.json").read_text()):
        directory = args.run / case["case"]
        data.append(dict(case=case["case"], metadata=json.loads((directory / "run.json").read_text()),
                         evidence=json.loads((directory / "profile_evidence.json").read_text())))
    template = Path(__file__).with_name("viewer.html").read_text()
    html = template.replace("__P32_DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c"))
    (args.run / "index.html").write_text(html)
    print(args.run / "index.html")


if __name__ == "__main__":
    main()
