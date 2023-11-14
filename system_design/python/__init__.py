import os
only_python = os.environ.get("CELERITAS_ONLY_PYTHON", None)
if not only_python:
    try:
        from ._pyceleritas import *
    except ModuleNotFoundError:
        print("Bindings not installed")
