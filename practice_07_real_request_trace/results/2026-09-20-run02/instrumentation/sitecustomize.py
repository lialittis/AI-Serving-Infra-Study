"""Enable tracing only for the explicitly opted-in experiment subprocesses."""

import os

if os.environ.get("P07_TRACE_DIR"):
    from trace_hooks import install

    install()
