import os

if os.environ.get('P14_OUTPUT'):
    from allocation_trace import install
    install()
