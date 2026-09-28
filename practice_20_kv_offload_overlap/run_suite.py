"""Six independent services; formal runs never enable the diagnostic observer."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--phase", choices=["benchmark", "diagnostic"], default="benchmark")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    modes = (["native", "serialized", "recompute", "recompute", "serialized", "native"]
             if a.phase == "benchmark" else ["native", "serialized", "recompute"])
    plan = dict(modes=modes, phase=a.phase, lengths=[1024, 3072],
                warmup=5 if a.phase == "benchmark" else 0,
                repeats=5 if a.phase == "benchmark" else 1)
    (a.output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    for i, mode in enumerate(modes):
        cmd = [sys.executable, "-u", str(Path(__file__).with_name("run_experiment.py")),
               "--mode", mode, "--phase", a.phase, "--port", str(a.port),
               "--output", str(a.output/("block-%02d-%s" % (i, mode))),
               "--block", str(0 if i < 3 else 1),
               "--warmup", str(plan["warmup"]), "--repeats", str(plan["repeats"])]
        print(json.dumps(dict(service=i, command=cmd)), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
