"""Only inject into explicitly configured P31 diagnostic service processes."""
import os
if os.environ.get("P31_OBSERVER_DIR"):
    from observer import install
    install()
