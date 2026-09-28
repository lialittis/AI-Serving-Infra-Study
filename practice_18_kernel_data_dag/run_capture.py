"""Run the reproducible Practice 13 eager request with richer access observation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "practice_13_operator_submission"))
from run_submission import main

if __name__ == "__main__":
    main(observer_dir=ROOT)
