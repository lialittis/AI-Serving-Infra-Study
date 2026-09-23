"""Opt-in observation of this experiment's vLLM processes."""
import os

if os.environ.get("P11_TRACE_DIR"):
    from attention_trace import install

    install()
