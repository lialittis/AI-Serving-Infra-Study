"""Activate observation only in the isolated Practice 12 subprocesses."""
import os
if os.environ.get('P12_TRACE_DIR'):
    from lifetime_trace import install
    install()
