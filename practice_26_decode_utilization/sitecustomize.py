"""Opt-in, process-local imports; no observer in unprofiled controls."""
import os
if os.environ.get('P26_PROFILE'):
    from observer import install
    install()
