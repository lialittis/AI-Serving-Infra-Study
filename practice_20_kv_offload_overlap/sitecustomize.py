"""Only the recompute diagnostic needs import-time observer activation."""
import os
import sys

if os.environ.get("P20_RECOMPUTE_OBSERVER") == "1":
    def ready(frame, event, arg):
        if (event == "return" and frame.f_code.co_name == "__init__"
                and frame.f_globals.get("__name__") == "vllm_ascend.worker.model_runner_v1"):
            sys.setprofile(None)
            from p20_connector import install_observer
            install_observer()
    sys.setprofile(ready)
