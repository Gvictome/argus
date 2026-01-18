"""
Camera service module for Raspberry Pi Camera Module 3

Handles:
- Camera initialization via picamera2
- Frame capture and streaming
- Recording management
- MJPEG streaming for web interface
"""

import io
import time
import threading
from dataclasses import dataclass
from typing import Optional, Generator, Callable, List
from pathlib import Path
from datetime import datetime

# Platform detection
import platform
IS_RASPBERRY_PI = platform.machine().startswith('aarch') or platform.machine().startswith('arm')


@dataclass
class CameraConfig:
    """Camera configuration for Pi Camera Module 3"""
    resolution: tuple = (1920, 1080)  # Full HD
    framerate: int = 30
    rotation: int = 0  # 0, 90, 180, 270
    hflip: bool = False
    vflip: bool = False
    autofocus_mode: str = "continuous"  # continuous, manual, auto
    hdr: bool = False
    format: str = "RGB888"


class CameraService:
    """
    Camera service for Raspberry Pi Camera Module 3

    Uses picamera2 library (Pi 5 compatible)
    Falls back to OpenCV for development on other platforms
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.camera = None
        self.is_initialized = False
        self.is_streaming = False
        self.is_recording = False
        self._frame_lock = threading.Lock()
        self._current_frame: Optional[bytes] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        self._video_writer = None

    def initialize(self) -> bool:
        """
        Initialize the camera

        Returns True if successful, False otherwise
        """
        if self.is_initialized:
            return True

        try:
            if IS_RASPBERRY_PI:
                return self._init_picamera2()
            else:
                return self._init_opencv()
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            return False

    def _init_picamera2(self) -> bool:
        """Initialize using picamera2 (Raspberry Pi)"""
        try:
            from picamera2 import Picamera2
            from libcamera import controls

            self.camera = Picamera2()

            # Configure camera for video/streaming
            config = self.camera.create_video_configuration(
                main={"size": self.config.resolution, "format": self.config.format},
                controls={"FrameRate": self.config.framerate}
            )

            self.camera.configure(config)

            # Set autofocus mode for Camera Module 3
            try:
                if self.config.autofocus_mode == "continuous":
                    self.camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                elif self.config.autofocus_mode == "auto":
                    self.camera.set_controls({"AfMode": controls.AfModeEnum.Auto})
            except:
                print("Autofocus not available on this camera")

            self.camera.start()
            time.sleep(0.5)  # Allow camera to warm up
            self.is_initialized = True
            print("Camera initialized (picamera2)")
            return True

        except ImportError:
            print("picamera2 not installed. Install with: sudo apt install python3-picamera2")
            return False
        except Exception as e:
            print(f"picamera2 initialization failed: {e}")
            return False

    def _init_opencv(self) -> bool:
        """Initialize using OpenCV (fallback for development)"""
        try:
            import cv2

            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("OpenCV: No camera found")
                return False

            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.config.framerate)

            self.is_initialized = True
            print("Camera initialized (OpenCV fallback)")
            return True

        except ImportError:
            print("OpenCV not installed. Install with: pip install opencv-python")
            return False
        except Exception as e:
            print(f"OpenCV initialization failed: {e}")
            return False

    def get_frame(self) -> Optional[bytes]:
        """
        Capture and return a single frame as JPEG bytes

        Returns None if camera not initialized
        """
        if not self.is_initialized:
            return None

        try:
            if IS_RASPBERRY_PI:
                return self._get_frame_picamera2()
            else:
                return self._get_frame_opencv()
        except Exception as e:
            print(f"Frame capture failed: {e}")
            return None

    def _get_frame_picamera2(self) -> Optional[bytes]:
        """Get frame using picamera2"""
        from PIL import Image

        # Capture frame as numpy array
        frame = self.camera.capture_array()

        # Convert to JPEG
        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()

    def _get_frame_opencv(self) -> Optional[bytes]:
        """Get frame using OpenCV"""
        import cv2

        ret, frame = self.camera.read()
        if not ret:
            return None

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()

    def get_frame_array(self):
        """Get frame as numpy array (for detection processing)"""
        if not self.is_initialized:
            return None

        try:
            if IS_RASPBERRY_PI:
                return self.camera.capture_array()
            else:
                import cv2
                ret, frame = self.camera.read()
                return frame if ret else None
        except Exception as e:
            print(f"Frame array capture failed: {e}")
            return None

    def stream_mjpeg(self) -> Generator[bytes, None, None]:
        """
        Generator that yields MJPEG frames for HTTP streaming

        Usage in FastAPI:
            return StreamingResponse(
                camera.stream_mjpeg(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
        """
        self.is_streaming = True
        while self.is_streaming:
            frame = self.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1.0 / self.config.framerate)

    def start_background_capture(self, callback: Optional[Callable] = None):
        """
        Start background frame capture thread

        Args:
            callback: Function to call with each frame (receives bytes)
        """
        if self._stream_thread and self._stream_thread.is_alive():
            return

        if callback:
            self._callbacks.append(callback)

        self.is_streaming = True
        self._stream_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._stream_thread.start()
        print("Background capture started")

    def _capture_loop(self):
        """Background capture loop"""
        while self.is_streaming:
            frame = self.get_frame()
            if frame:
                with self._frame_lock:
                    self._current_frame = frame

                # Call registered callbacks
                for callback in self._callbacks:
                    try:
                        callback(frame)
                    except Exception as e:
                        print(f"Callback error: {e}")

            time.sleep(1.0 / self.config.framerate)

    def get_current_frame(self) -> Optional[bytes]:
        """Get the most recent frame from background capture"""
        with self._frame_lock:
            return self._current_frame

    def start_recording(self, output_path: Path) -> bool:
        """
        Start video recording

        Args:
            output_path: Path to save video file

        Returns True if recording started
        """
        if not self.is_initialized or self.is_recording:
            return False

        try:
            if IS_RASPBERRY_PI:
                from picamera2.encoders import H264Encoder
                from picamera2.outputs import FfmpegOutput

                encoder = H264Encoder(bitrate=10000000)
                output = FfmpegOutput(str(output_path))
                self.camera.start_recording(encoder, output)
            else:
                import cv2
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self._video_writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    self.config.framerate,
                    self.config.resolution
                )

            self.is_recording = True
            print(f"Recording started: {output_path}")
            return True

        except Exception as e:
            print(f"Failed to start recording: {e}")
            return False

    def stop_recording(self) -> bool:
        """Stop video recording"""
        if not self.is_recording:
            return False

        try:
            if IS_RASPBERRY_PI:
                self.camera.stop_recording()
            else:
                if self._video_writer:
                    self._video_writer.release()
                    self._video_writer = None

            self.is_recording = False
            print("Recording stopped")
            return True

        except Exception as e:
            print(f"Failed to stop recording: {e}")
            return False

    def capture_snapshot(self, output_path: Optional[Path] = None) -> Optional[bytes]:
        """
        Capture a snapshot

        Args:
            output_path: Optional path to save image

        Returns:
            JPEG bytes if successful, None otherwise
        """
        frame = self.get_frame()
        if frame and output_path:
            output_path.write_bytes(frame)
            print(f"Snapshot saved: {output_path}")
        return frame

    def add_frame_callback(self, callback: Callable):
        """Add a callback to be called with each captured frame"""
        self._callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable):
        """Remove a frame callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_status(self) -> dict:
        """Get camera status"""
        return {
            "initialized": self.is_initialized,
            "streaming": self.is_streaming,
            "recording": self.is_recording,
            "resolution": f"{self.config.resolution[0]}x{self.config.resolution[1]}",
            "fps": self.config.framerate,
            "platform": "picamera2" if IS_RASPBERRY_PI else "opencv",
            "autofocus": self.config.autofocus_mode,
        }

    def shutdown(self):
        """Clean shutdown of camera"""
        print("Shutting down camera...")
        self.is_streaming = False
        self.is_recording = False

        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)

        if self.camera:
            try:
                if IS_RASPBERRY_PI:
                    self.camera.stop()
                    self.camera.close()
                else:
                    self.camera.release()
            except:
                pass

        self.is_initialized = False
        print("Camera shutdown complete")


# Singleton instance
camera_service = CameraService()
