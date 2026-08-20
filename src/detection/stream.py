"""
Annotated MJPEG streaming for the live demo (showcase P0-2).

Pulls frames as numpy arrays straight from the camera, runs them through
the detection cascade, burns boxes and labels in, and yields JPEG parts.
Frames arrive as arrays rather than encoded JPEG, so the stream costs one
encode per frame with no decode.
"""

import logging
import time
from typing import Generator, List

import cv2

from src.detection import Detection
from src.detection.annotate import draw_detections

logger = logging.getLogger(__name__)

BOUNDARY = b"frame"


def stream_annotated_mjpeg(
    camera,
    detector,
    detect_every: int = 3,
    jpeg_quality: int = 80,
) -> Generator[bytes, None, None]:
    """
    Yield multipart MJPEG parts with detection overlays burned in.

    Args:
        camera: CameraService (needs get_frame_array, is_streaming, config).
        detector: DetectionService (needs process_frame).
        detect_every: Run the detector every Nth frame, reusing the previous
            boxes in between. The YOLO plus ArcFace cascade is far more
            expensive than an encode, so running it on every frame is what
            collapses the frame rate. Boxes persist between runs so they do
            not flicker.
        jpeg_quality: 0-100, passed to the JPEG encoder.

    Yields:
        Complete multipart parts, ready for StreamingResponse.
    """
    camera.is_streaming = True
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    interval = 1.0 / max(1, getattr(camera.config, "framerate", 30))
    detections: List[Detection] = []
    frame_index = 0

    while camera.is_streaming:
        frame = camera.get_frame_array()
        if frame is None:
            # A dropped capture is normal under load. Pace the loop anyway so
            # a persistently failing camera cannot spin the CPU.
            time.sleep(interval)
            continue

        if frame_index % max(1, detect_every) == 0:
            try:
                detections = detector.process_frame(frame)
            except Exception as exc:
                # Video is the demo. A model failure degrades to a raw feed
                # rather than a dead stream.
                logger.warning("Detection failed on frame %d: %s", frame_index, exc)
                detections = []
        frame_index += 1

        annotated = draw_detections(frame, detections)

        ok, buffer = cv2.imencode(".jpg", annotated, encode_params)
        if not ok:
            logger.warning("JPEG encode failed on frame %d", frame_index)
            time.sleep(interval)
            continue

        yield (
            b"--" + BOUNDARY + b"\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        time.sleep(interval)
