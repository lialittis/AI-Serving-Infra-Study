"""Only instrument this practice's explicitly opted-in subprocesses."""
import os

if os.environ.get("P18_TRACE_DIR"):
    from dependency_trace import install
    install()
