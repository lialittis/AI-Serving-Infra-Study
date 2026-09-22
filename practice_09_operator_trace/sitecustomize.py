"""Opt-in annotations for this experiment's vLLM subprocesses only."""
import os

if os.environ.get("P09_TRACE_DIR"):
    from operator_trace import install

    install()
