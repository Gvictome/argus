"""
Test-wide environment.

pytest imports this before any test module, and src.config builds its
settings object from the environment at import time, so anything set here
reaches the app.
"""

import os

# The detection worker opens a real camera at application startup. Tests
# drive the pipeline with stubs and must never grab the developer's webcam.
os.environ.setdefault("DETECTION_AUTOSTART", "false")

# An exported OpenVINO/NCNN model in models/ would otherwise be picked up
# automatically, making backend-dependent assertions pass or fail depending
# on what happens to be on the developer's disk.
os.environ.setdefault("OBJECT_BACKEND", "pt")


import pytest


@pytest.fixture(autouse=True)
def _reset_login_limiter():
    """Failed-login counters are process-wide; keep tests independent."""
    from src.api.rate_limit import reset_login_limiter

    reset_login_limiter()
    yield
    reset_login_limiter()
