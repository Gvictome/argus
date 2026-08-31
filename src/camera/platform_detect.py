"""
Which board are we actually on?

ARGUS previously decided this with:

    IS_RASPBERRY_PI = platform.machine().startswith('aarch') or ...

That is true on a Jetson Orin, which is also aarch64. The camera service
would therefore try to import `picamera2` on a Jetson, fail, and report
"camera initialization failed" -- a hardware-looking error with a purely
software cause.

`platform.machine()` describes the CPU architecture, not the board. The
board has to be read from the device tree.
"""

import logging
import platform
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEVICE_TREE_MODEL = Path("/proc/device-tree/model")
_TEGRA_RELEASE = Path("/etc/nv_tegra_release")


class Board(str, Enum):
    """The board ARGUS is running on, as far as it can tell."""
    RASPBERRY_PI = "raspberry_pi"
    JETSON = "jetson"
    GENERIC = "generic"          # laptop, desktop, CI, unknown SBC


def _device_tree_model() -> str:
    """The board's self-reported model string, or '' when unavailable."""
    try:
        if _DEVICE_TREE_MODEL.exists():
            # The device tree null-terminates these.
            return _DEVICE_TREE_MODEL.read_bytes().decode(
                "utf-8", errors="ignore"
            ).rstrip("\x00").strip()
    except OSError:
        pass
    return ""


def detect_board() -> Board:
    """
    Identify the board.

    Jetson is checked first: `/etc/nv_tegra_release` is written by the
    L4T BSP and exists on every JetPack install, which makes it a more
    reliable signal than the model string alone.
    """
    if _TEGRA_RELEASE.exists():
        return Board.JETSON

    model = _device_tree_model().lower()
    if "jetson" in model or "tegra" in model or "orin" in model:
        return Board.JETSON
    if "raspberry pi" in model:
        return Board.RASPBERRY_PI

    return Board.GENERIC


def board_description() -> str:
    """A human-readable board line for startup logs and /api/status."""
    model = _device_tree_model()
    if model:
        return f"{model} ({platform.machine()})"
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def jetson_model() -> Optional[str]:
    """The Jetson model string, or None when this is not a Jetson."""
    if detect_board() is not Board.JETSON:
        return None
    return _device_tree_model() or "NVIDIA Jetson"


BOARD = detect_board()

# Kept because existing code and tests import it. It now means what its
# name says -- an actual Raspberry Pi -- rather than "any ARM board".
IS_RASPBERRY_PI = BOARD is Board.RASPBERRY_PI
IS_JETSON = BOARD is Board.JETSON
