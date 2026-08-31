"""
Capture backends, one per board family.

Each backend answers the same three questions -- open a sensor, hand back
a BGR frame, close -- so `CameraService` does not branch on the platform
in every method. Adding a board means adding a class here.

    Board.RASPBERRY_PI  -> Picamera2Backend    (picamera2 / libcamera)
    Board.JETSON        -> JetsonCSIBackend    (GStreamer / nvarguscamerasrc)
    Board.GENERIC       -> OpenCVBackend       (V4L2 / DirectShow, dev machines)

The Jetson path goes through GStreamer because that is the only way to
reach the Orin's ISP: `nvarguscamerasrc` is the Argus camera source, and
it is what makes a CSI sensor usable at all. A plain `cv2.VideoCapture(0)`
on a Jetson opens USB/V4L2 devices and will not see a CSI camera.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CaptureBackend(ABC):
    """A camera we can open, read BGR frames from, and close."""

    name = "abstract"

    @abstractmethod
    def open(self) -> bool:
        """Acquire the device. False on failure, never raises."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """One BGR frame, or None if the device produced nothing."""

    @abstractmethod
    def close(self) -> None:
        """Release the device. Safe to call when already closed."""


# ---------------------------------------------------------------------------
# Jetson
# ---------------------------------------------------------------------------

def build_csi_pipeline(
    sensor_id: int = 0,
    capture_size: Tuple[int, int] = (1920, 1080),
    output_size: Optional[Tuple[int, int]] = None,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """
    A GStreamer pipeline string for one Orin CSI camera.

    Notes on the pieces that matter:

    `nvarguscamerasrc sensor-id=N` selects which CSI connector to read.
    This is the whole basis of multi-camera on the Orin -- two cameras
    are two pipelines differing only in this number.

    `(memory:NVMM)` keeps frames in the hardware-accessible memory pool,
    and `nvvidconv` is the hardware converter out of it. Skipping either
    forces a CPU copy of every frame and throws away the reason for
    using CSI.

    `drop=true max-buffers=1` on the sink means a slow consumer gets the
    *latest* frame rather than a growing backlog. For a security feed a
    late frame is worthless, so dropping is correct; without it, latency
    grows without bound whenever inference falls behind capture.
    """
    out_w, out_h = output_size or capture_size
    cap_w, cap_h = capture_size

    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){cap_w}, height=(int){cap_h}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){out_w}, height=(int){out_h}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def rotation_to_flip_method(rotation: int, hflip: bool = False, vflip: bool = False) -> int:
    """
    Map ARGUS's rotation/flip config onto nvvidconv's flip-method enum.

    nvvidconv numbers these in its own order, which does not match
    degrees; hence the table rather than arithmetic.
    """
    if hflip and not vflip:
        return 4          # horizontal mirror
    if vflip and not hflip:
        return 2          # vertical mirror
    if hflip and vflip:
        return 6          # both == 180 with mirror
    return {0: 0, 90: 1, 180: 2, 270: 3}.get(rotation % 360, 0)


class JetsonCSIBackend(CaptureBackend):
    """A CSI camera on a Jetson, via GStreamer and the Argus ISP."""

    name = "jetson-csi"

    def __init__(
        self,
        sensor_id: int = 0,
        resolution: Tuple[int, int] = (1920, 1080),
        framerate: int = 30,
        flip_method: int = 0,
        warmup_s: float = 0.5,
    ):
        self.sensor_id = sensor_id
        self.resolution = resolution
        self.framerate = framerate
        self.flip_method = flip_method
        self.warmup_s = warmup_s
        self._cap = None

    def open(self) -> bool:
        import cv2

        pipeline = build_csi_pipeline(
            sensor_id=self.sensor_id,
            capture_size=self.resolution,
            framerate=self.framerate,
            flip_method=self.flip_method,
        )
        logger.info("CSI sensor %d pipeline: %s", self.sensor_id, pipeline)

        try:
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        except Exception as exc:
            logger.error("CSI sensor %d: GStreamer open failed: %s", self.sensor_id, exc)
            return False

        if not self._cap or not self._cap.isOpened():
            logger.error(
                "CSI sensor %d did not open. Check: the sensor is detected "
                "(ls /dev/video*), the driver is loaded for this module, and "
                "OpenCV was built WITH GStreamer (cv2.getBuildInformation()).",
                self.sensor_id,
            )
            self._cap = None
            return False

        # Argus needs a moment before the first frames are usable.
        time.sleep(self.warmup_s)
        logger.info("CSI sensor %d opened", self.sensor_id)
        return True

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------------------
# Raspberry Pi
# ---------------------------------------------------------------------------

class Picamera2Backend(CaptureBackend):
    """A Pi camera via picamera2. Unchanged behavior, moved behind the ABC."""

    name = "picamera2"

    def __init__(
        self,
        camera_num: int = 0,
        resolution: Tuple[int, int] = (1920, 1080),
        framerate: int = 30,
        autofocus_mode: str = "continuous",
        fmt: str = "RGB888",
    ):
        self.camera_num = camera_num
        self.resolution = resolution
        self.framerate = framerate
        self.autofocus_mode = autofocus_mode
        self.format = fmt
        self._cam = None

    def open(self) -> bool:
        try:
            from picamera2 import Picamera2
            from libcamera import controls
        except ImportError:
            logger.error(
                "picamera2 not installed. On Pi OS: "
                "sudo apt install -y python3-picamera2, and create the venv "
                "with --system-site-packages so it is visible."
            )
            return False

        try:
            self._cam = Picamera2(camera_num=self.camera_num)
            cfg = self._cam.create_video_configuration(
                main={"size": self.resolution, "format": self.format},
                controls={"FrameRate": self.framerate},
            )
            self._cam.configure(cfg)

            try:
                if self.autofocus_mode == "continuous":
                    self._cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                elif self.autofocus_mode == "auto":
                    self._cam.set_controls({"AfMode": controls.AfModeEnum.Auto})
            except Exception:
                logger.info("Autofocus unavailable on this sensor")

            self._cam.start()
            time.sleep(0.5)
            return True
        except Exception as exc:
            logger.error("picamera2 init failed: %s", exc)
            self._cam = None
            return False

    def read(self) -> Optional[np.ndarray]:
        if self._cam is None:
            return None
        try:
            frame = self._cam.capture_array()
        except Exception as exc:
            logger.warning("picamera2 capture failed: %s", exc)
            return None
        # picamera2 hands back RGB; the rest of ARGUS is BGR throughout.
        if frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
            return frame[:, :, ::-1]
        return frame

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stop()
                self._cam.close()
            except Exception:
                pass
            self._cam = None


# ---------------------------------------------------------------------------
# Everything else
# ---------------------------------------------------------------------------

class OpenCVBackend(CaptureBackend):
    """A V4L2/DirectShow/UVC camera. Dev machines, and USB cams anywhere."""

    name = "opencv"

    def __init__(
        self,
        index: int = 0,
        resolution: Tuple[int, int] = (1920, 1080),
        framerate: int = 30,
    ):
        self.index = index
        self.resolution = resolution
        self.framerate = framerate
        self._cap = None

    def open(self) -> bool:
        import cv2

        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            logger.error("OpenCV: no camera at index %d", self.index)
            self._cap = None
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.framerate)
        return True

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def build_backend(board, sensor_id: int, config) -> CaptureBackend:
    """
    The right backend for this board and sensor.

    `sensor_id` means the CSI connector on a Jetson, the camera number on
    a Pi, and the V4L2/DirectShow index elsewhere -- which is the same
    idea in each case: which physical camera.
    """
    from src.camera.platform_detect import Board

    if board is Board.JETSON:
        return JetsonCSIBackend(
            sensor_id=sensor_id,
            resolution=config.resolution,
            framerate=config.framerate,
            flip_method=rotation_to_flip_method(
                config.rotation, config.hflip, config.vflip
            ),
        )

    if board is Board.RASPBERRY_PI:
        return Picamera2Backend(
            camera_num=sensor_id,
            resolution=config.resolution,
            framerate=config.framerate,
            autofocus_mode=config.autofocus_mode,
            fmt=config.format,
        )

    return OpenCVBackend(
        index=sensor_id,
        resolution=config.resolution,
        framerate=config.framerate,
    )
