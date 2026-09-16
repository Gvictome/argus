"""
Camera service.

Supports several boards through one interface:

    Raspberry Pi   picamera2 / libcamera
    Jetson Orin    GStreamer / nvarguscamerasrc (CSI)
    anything else  OpenCV (V4L2 / DirectShow)

The board is detected from the device tree, not from `platform.machine()`
-- see `src/camera/platform_detect.py` for why that distinction matters.

Multiple cameras are supported through `CameraRegistry`: the Orin has two
CSI connectors, and each is one `CameraService` differing only by
`sensor_id`. `camera_service` remains as the module-level singleton for
camera 0, so existing single-camera callers are unaffected.
"""

import io
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional

import numpy as np

from src.camera.backends import CaptureBackend, build_backend
from src.camera.platform_detect import (  # noqa: F401  (re-exported)
    BOARD,
    IS_JETSON,
    IS_RASPBERRY_PI,
    Board,
    board_description,
)

logger = logging.getLogger(__name__)

JPEG_QUALITY = 85


@dataclass
class CameraConfig:
    """Per-camera configuration."""
    resolution: tuple = (1920, 1080)
    framerate: int = 30
    rotation: int = 0            # 0, 90, 180, 270
    hflip: bool = False
    vflip: bool = False
    autofocus_mode: str = "continuous"
    hdr: bool = False
    format: str = "RGB888"
    # Which physical camera: CSI connector on Jetson, camera number on a
    # Pi, capture index elsewhere.
    sensor_id: int = 0
    # Human label for the dashboard ("Front door", "Back gate").
    name: str = "camera0"
    # Exchange red and blue on capture. Off by default: picamera2's
    # "RGB888" already delivers BGR, which is what the pipeline wants.
    swap_rb: bool = False
    # Video file path or capture index that overrides the board camera.
    source: str = ""
    # Pace a file source to its own frame rate, like a live camera.
    realtime: bool = True


class CameraService:
    """
    One camera: open it, read frames, stream, record.

    Frames are BGR numpy arrays everywhere internally, matching OpenCV and
    the detection pipeline. Backends convert to BGR at their edge so that
    conversion is not scattered through the callers.
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.backend: Optional[CaptureBackend] = None
        self.is_initialized = False
        self.is_streaming = False
        self.is_recording = False
        self._frame_lock = threading.Lock()
        self._current_frame: Optional[bytes] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        self._video_writer = None

    # -- lifecycle ------------------------------------------------------

    def initialize(self) -> bool:
        """Open the camera. False on failure; never raises."""
        if self.is_initialized:
            return True

        self.backend = build_backend(BOARD, self.config.sensor_id, self.config)
        try:
            opened = self.backend.open()
        except Exception as exc:
            logger.error("Camera %s init failed: %s", self.config.name, exc)
            opened = False

        if not opened:
            self.backend = None
            return False

        self.is_initialized = True
        logger.info(
            "Camera %r initialized (sensor %d, %s)",
            self.config.name, self.config.sensor_id, self.backend.name,
        )
        return True

    def shutdown(self) -> None:
        """Stop streaming and release the device."""
        logger.info("Shutting down camera %r", self.config.name)
        self.is_streaming = False

        if self.is_recording:
            self.stop_recording()

        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None

        if self.backend is not None:
            try:
                self.backend.close()
            except Exception:
                pass
            self.backend = None

        self.is_initialized = False

    # -- capture --------------------------------------------------------

    def get_frame_array(self) -> Optional[np.ndarray]:
        """One BGR frame as a numpy array, for the detection pipeline."""
        if not self.is_initialized or self.backend is None:
            return None
        try:
            return self.backend.read()
        except Exception as exc:
            logger.warning("Camera %s read failed: %s", self.config.name, exc)
            return None

    def get_frame(self) -> Optional[bytes]:
        """One frame as JPEG bytes."""
        frame = self.get_frame_array()
        if frame is None:
            return None
        return self._encode_jpeg(frame)

    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> Optional[bytes]:
        try:
            import cv2
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            return buf.tobytes() if ok else None
        except Exception:
            # OpenCV is always present in practice; PIL is the safety net.
            try:
                from PIL import Image
                img = Image.fromarray(frame[:, :, ::-1])
                out = io.BytesIO()
                img.save(out, format="JPEG", quality=JPEG_QUALITY)
                return out.getvalue()
            except Exception as exc:
                logger.warning("JPEG encode failed: %s", exc)
                return None

    def capture_snapshot(self, output_path: Optional[Path] = None) -> Optional[bytes]:
        frame = self.get_frame()
        if frame and output_path:
            output_path.write_bytes(frame)
            logger.info("Snapshot saved: %s", output_path)
        return frame

    # -- streaming ------------------------------------------------------

    def stream_mjpeg(self) -> Generator[bytes, None, None]:
        """Raw MJPEG, no overlays. See src/detection/stream.py for annotated."""
        self.is_streaming = True
        interval = 1.0 / max(1, self.config.framerate)
        while self.is_streaming:
            frame = self.get_frame()
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(interval)

    def start_background_capture(self, callback: Optional[Callable] = None) -> None:
        if self._stream_thread and self._stream_thread.is_alive():
            return
        if callback:
            self._callbacks.append(callback)
        self.is_streaming = True
        self._stream_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._stream_thread.start()

    def _capture_loop(self) -> None:
        interval = 1.0 / max(1, self.config.framerate)
        while self.is_streaming:
            frame = self.get_frame()
            if frame:
                with self._frame_lock:
                    self._current_frame = frame
                for cb in self._callbacks:
                    try:
                        cb(frame)
                    except Exception as exc:
                        logger.warning("Frame callback failed: %s", exc)
            time.sleep(interval)

    def get_current_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._current_frame

    def add_frame_callback(self, callback: Callable) -> None:
        self._callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    # -- recording ------------------------------------------------------

    def start_recording(self, output_path: Path) -> bool:
        """
        Record to a file.

        Uses OpenCV's writer on every board. picamera2's hardware H.264
        encoder is faster on a Pi, but it is bound to the picamera2 object
        the backend now owns, and one code path that works everywhere is
        worth more here than a per-board optimisation.
        """
        if not self.is_initialized or self.is_recording:
            return False
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                str(output_path), fourcc, self.config.framerate, self.config.resolution
            )
            if not self._video_writer.isOpened():
                logger.error("Could not open video writer for %s", output_path)
                self._video_writer = None
                return False

            self.add_frame_callback(self._write_frame)
            self.start_background_capture()
            self.is_recording = True
            logger.info("Recording to %s", output_path)
            return True
        except Exception as exc:
            logger.error("Failed to start recording: %s", exc)
            return False

    def _write_frame(self, _jpeg: bytes) -> None:
        if self._video_writer is None:
            return
        frame = self.get_frame_array()
        if frame is not None:
            self._video_writer.write(frame)

    def stop_recording(self) -> bool:
        if not self.is_recording:
            return False
        self.remove_frame_callback(self._write_frame)
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self.is_recording = False
        logger.info("Recording stopped")
        return True

    # -- status ---------------------------------------------------------

    def state(self) -> str:
        """
        One word for what this camera is doing.

        `/api/camera/status` had no `status` field while
        `/api/detection/status` did, so the two status endpoints disagreed
        about their own shape. tests/test_api.py has asserted this key
        since before the showcase work and had been failing for it.
        """
        if not self.is_initialized:
            return "stopped"
        if self.is_recording:
            return "recording"
        if self.is_streaming:
            return "streaming"
        return "ready"

    def get_status(self) -> dict:
        return {
            "status": self.state(),
            "name": self.config.name,
            "sensor_id": self.config.sensor_id,
            "initialized": self.is_initialized,
            "streaming": self.is_streaming,
            "recording": self.is_recording,
            "resolution": f"{self.config.resolution[0]}x{self.config.resolution[1]}",
            "fps": self.config.framerate,
            "platform": self.backend.name if self.backend else _default_backend_name(),
            "board": BOARD.value,
            "autofocus": self.config.autofocus_mode,
        }


def _default_backend_name() -> str:
    """What backend *would* be used, for status before initialize()."""
    return {
        Board.JETSON: "jetson-csi",
        Board.RASPBERRY_PI: "picamera2",
    }.get(BOARD, "opencv")


class CameraRegistry:
    """
    Every camera on this host, addressed by id.

    The Orin has two CSI connectors, so the demo runs two cameras from one
    process. They are kept in one place rather than as loose globals so
    that "start everything" and "shut everything down" are single calls --
    a camera left open after a failed startup blocks the next attempt.
    """

    def __init__(self) -> None:
        self._cameras: Dict[str, CameraService] = {}
        self._lock = threading.Lock()

    def add(self, config: CameraConfig) -> CameraService:
        with self._lock:
            cam = CameraService(config)
            self._cameras[config.name] = cam
            return cam

    def get(self, name: str) -> Optional[CameraService]:
        return self._cameras.get(name)

    def primary(self) -> Optional[CameraService]:
        """
        The camera the single-camera endpoints act on.

        Lowest sensor_id rather than insertion order, so the answer does
        not depend on configuration ordering.
        """
        if not self._cameras:
            return None
        return min(self._cameras.values(), key=lambda c: c.config.sensor_id)

    def names(self) -> List[str]:
        return list(self._cameras)

    def all(self) -> List[CameraService]:
        return list(self._cameras.values())

    def initialize_all(self) -> Dict[str, bool]:
        """
        Open every configured camera.

        Returns per-camera success rather than one boolean: with two
        cameras, "one of them failed" is the interesting case and a single
        False would hide which.
        """
        results = {}
        for name, cam in self._cameras.items():
            results[name] = cam.initialize()
            if not results[name]:
                logger.error("Camera %r failed to initialize", name)
        return results

    def shutdown_all(self) -> None:
        for cam in self._cameras.values():
            try:
                cam.shutdown()
            except Exception as exc:
                logger.warning("Shutdown failed for %r: %s", cam.config.name, exc)

    def status(self) -> dict:
        return {
            "board": BOARD.value,
            "board_description": board_description(),
            "count": len(self._cameras),
            "cameras": [c.get_status() for c in self._cameras.values()],
        }


def build_registry_from_settings(settings) -> CameraRegistry:
    """
    Construct the registry from CAMERA_SENSORS.

    Format: comma-separated `sensor_id:name`, or just `sensor_id`.
        CAMERA_SENSORS="0:front,1:back"
        CAMERA_SENSORS="0"
    """
    registry = CameraRegistry()
    spec = (getattr(settings, "CAMERA_SENSORS", "") or "").strip()

    if not spec:
        spec = str(getattr(settings, "CAMERA_INDEX", 0))

    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            raw_id, name = entry.split(":", 1)
        else:
            raw_id, name = entry, f"camera{entry.strip()}"
        try:
            sensor_id = int(raw_id)
        except ValueError:
            logger.warning("Skipping malformed CAMERA_SENSORS entry %r", entry)
            continue

        registry.add(CameraConfig(
            resolution=getattr(settings, "CAMERA_RESOLUTION", (1920, 1080)),
            framerate=getattr(settings, "CAMERA_FPS", 30),
            swap_rb=getattr(settings, "CAMERA_SWAP_RB", False),
            rotation=getattr(settings, "CAMERA_ROTATION", 0),
            sensor_id=sensor_id,
            name=name.strip(),
        ))

    return registry


# Module-level singleton for camera 0. Existing single-camera callers and
# the current API endpoints keep working unchanged; the registry is what
# multi-camera code uses.
camera_service = CameraService()

# Populated at startup by src/api/app.py.
camera_registry = CameraRegistry()
