"""Opt in only in the explicitly launched service and its worker subprocesses."""
import os
if os.environ.get('P22_TRACE_DIR'):
    from stream_trace import install
    install()
