"""Only inject into explicitly configured P35 diagnostic processes."""
import os
if os.environ.get("P35_OBSERVER_DIR"):
    from observer import install
    install()
