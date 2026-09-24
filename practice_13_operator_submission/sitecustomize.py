"""Enable observation only in this experiment's subprocesses."""
import os
if os.environ.get('P13_TRACE_DIR'):
    from submission_trace import install
    install()
