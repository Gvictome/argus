import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import requests
import threading
import time
import io
import numpy as np
import hailo

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

# --- Configuration ---
ARGUS_API = "http://localhost:5000/api/detections"
RECORD_API = "http://localhost:5000/api/record"
FRAME_API = "http://localhost:5000/api/frame"
PIR_GPIO_PIN = 17
PERSON_CONFIDENCE_THRESHOLD = 0.60
MOTION_COOLDOWN = 30  # seconds to keep running after last motion
SOFTWARE_MOTION_THRESHOLD = 5.0  # mean pixel difference to trigger


# --- PIR Motion Sensor (hardware) ---
class PIRSensor:
    def __init__(self, pin):
        self.pin = pin
        self.motion_detected = False
        self.last_motion_time = 0
        self._running = True

        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            GPIO.add_event_detect(self.pin, GPIO.RISING, callback=self._on_motion, bouncetime=2000)
            print(f"[PIR] Initialized on GPIO {self.pin}")
        except (ImportError, RuntimeError) as e:
            print(f"[PIR] GPIO not available ({e}), PIR disabled — using software motion only")
            self.GPIO = None
            self.motion_detected = True  # always on if no PIR

    def _on_motion(self, channel):
        self.motion_detected = True
        self.last_motion_time = time.time()
        print("[PIR] Motion detected!")

    def is_active(self):
        if self.GPIO is None:
            return True
        if self.motion_detected:
            if time.time() - self.last_motion_time > MOTION_COOLDOWN:
                self.motion_detected = False
                print("[PIR] Motion cooldown expired, going idle")
                return False
            return True
        return False

    def cleanup(self):
        if self.GPIO:
            self.GPIO.cleanup()


# --- Software Motion Detection ---
class SoftwareMotionDetector:
    def __init__(self):
        self.prev_frame = None
        self.motion_detected = False
        self.last_motion_time = 0

    def check(self, frame):
        if frame is None:
            return False

        # Convert to grayscale by averaging channels
        gray = np.mean(frame, axis=2).astype(np.float32)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        # Calculate frame difference
        diff = np.abs(gray - self.prev_frame)
        mean_diff = np.mean(diff)
        self.prev_frame = gray

        if mean_diff > SOFTWARE_MOTION_THRESHOLD:
            self.motion_detected = True
            self.last_motion_time = time.time()
            return True

        # Keep active during cooldown
        if self.motion_detected and time.time() - self.last_motion_time < MOTION_COOLDOWN:
            return True

        self.motion_detected = False
        return False


# --- Main Callback ---
class ArgusCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_count = 0
        self.pir = PIRSensor(PIR_GPIO_PIN)
        self.sw_motion = SoftwareMotionDetector()
        self.recording = False
        self.use_frame = True  # enable frame access for motion detection


def app_callback(element, buffer, user_data):
    if buffer is None:
        return

    user_data.frame_count += 1

    # Always grab and stream frames for live view (every 3rd frame)
    pad = element.get_static_pad("src")
    fmt, width, height = get_caps_from_pad(pad)
    frame = None
    if fmt and width and height and user_data.frame_count % 3 == 0:
        try:
            frame = get_numpy_from_buffer(buffer, fmt, width, height)
        except Exception:
            pass

    if frame is not None:
        try:
            from PIL import Image
            img = Image.fromarray(frame)
            buf_io = io.BytesIO()
            img.save(buf_io, format='JPEG', quality=70)
            threading.Thread(
                target=lambda d: requests.post(FRAME_API, data=d, headers={"Content-Type": "image/jpeg"}, timeout=0.3),
                args=(buf_io.getvalue(),),
                daemon=True
            ).start()
        except Exception:
            pass

    # Check PIR — if no motion, skip detection (but stream still runs above)
    if not user_data.pir.is_active():
        if user_data.frame_count % 100 == 0:
            print("[ARGUS] Idle — waiting for PIR motion...")
        return

    # Check software motion detection
    if not user_data.sw_motion.check(frame):
        if user_data.frame_count % 50 == 0:
            print("[ARGUS] PIR active but no software motion detected")
        return

    # Only process every 10th frame for YOLO detections
    if user_data.frame_count % 10 != 0:
        return

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    person_detected = False

    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()
        bbox = detection.get_bbox()

        if confidence > 0.5:
            det_data = {
                "label": label,
                "confidence": round(float(confidence), 2),
                "bbox": {
                    "xmin": round(float(bbox.xmin()), 3),
                    "ymin": round(float(bbox.ymin()), 3),
                    "xmax": round(float(bbox.xmax()), 3),
                    "ymax": round(float(bbox.ymax()), 3),
                },
                "frame": user_data.frame_count,
            }

            try:
                requests.post(ARGUS_API, json=det_data, timeout=0.5)
            except Exception:
                pass

            print(f"[ARGUS] {label} ({confidence:.0%})")

            # Check for person with >= 60% confidence
            if label == "person" and confidence >= PERSON_CONFIDENCE_THRESHOLD:
                person_detected = True

    # Trigger recording when person detected
    if person_detected and not user_data.recording:
        user_data.recording = True
        print("[ARGUS] PERSON DETECTED — triggering recording!")
        try:
            requests.post(RECORD_API, json={
                "trigger": "person",
                "confidence": PERSON_CONFIDENCE_THRESHOLD,
                "frame": user_data.frame_count,
            }, timeout=1)
        except Exception:
            pass

    # Stop recording after cooldown with no person
    if not person_detected and user_data.recording:
        if not user_data.sw_motion.motion_detected:
            user_data.recording = False
            print("[ARGUS] No person — stopping recording")


def main():
    user_data = ArgusCallback()
    try:
        app = GStreamerDetectionApp(app_callback, user_data)
        app.run()
    finally:
        user_data.pir.cleanup()


if __name__ == "__main__":
    main()
