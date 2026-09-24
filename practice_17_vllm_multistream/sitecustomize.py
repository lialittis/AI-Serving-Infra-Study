"""Enable Practice 17 observation only in explicitly configured subprocesses."""
import os

if os.environ.get("P17_TRACE_DIR"):
    from multistream_trace import install

    install()
