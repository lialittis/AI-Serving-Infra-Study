"""Opt in only the Practice 08 server and its child processes."""

import os

if os.environ.get("P08_TRACE_DIR"):
    from kv_trace import install

    install()
