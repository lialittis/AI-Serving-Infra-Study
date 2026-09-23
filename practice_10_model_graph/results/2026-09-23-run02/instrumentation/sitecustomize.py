"""Enable this practice only in its own vLLM subprocesses."""
import os

if os.environ.get("P10_GRAPH_DIR"):
    from graph_capture import install

    install()
