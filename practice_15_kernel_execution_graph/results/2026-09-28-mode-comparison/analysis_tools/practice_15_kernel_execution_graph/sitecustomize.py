"""Opt-in observer for the Practice 15 model-mode experiment only."""
import os
if os.environ.get('P15_MODE_TRACE') and os.environ.get('P13_TRACE_DIR'):
    from graph_trace import install
    install()
