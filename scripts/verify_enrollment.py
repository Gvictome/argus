#!/usr/bin/env python3
"""
Hardware verification for live-frame face enrollment (showcase G-3 / P0-4).

Enrollment is "done in code" but had only ever been exercised on a laptop.
This drives the real path on the Pi -- camera capture, ArcFace embedding,
database write, then a fresh capture matched back against it -- and reports
the similarity score, because "it worked" without a number tells you nothing
about how much margin you have before a judge sees a miss.

Usage:
    python scripts/verify_enrollment.py --name "Giovanny"
    python scripts/verify_enrollment.py --name "Giovanny" --keep

Exit codes:
    0  enrollment and re-recognition both succeeded
    1  verification failed (see the printed reason)
    2  prerequisite missing (camera, insightface, or database unavailable)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

PASS = "PASS"
FAIL = "FAIL"


def _say(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def _capture(camera, attempts: int = 10, delay: float = 0.3):
    """
    Grab a frame, retrying briefly.

    The first frames after camera init are routinely dropped or badly
    exposed while auto-exposure settles, so a single failed grab is not
    evidence the camera is broken.
    """
    for _ in range(attempts):
        frame = camera.get_frame_array()
        if frame is not None:
            return frame
        time.sleep(delay)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify live-frame face enrollment on hardware.")
    parser.add_argument("--name", default="VERIFY_TEST", help="Label to enroll under.")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the enrolment instead of deleting it. Off by default so "
             "repeated runs do not pollute the known-faces database.",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=2.0,
        help="Seconds to wait after camera init for auto-exposure to settle.",
    )
    args = parser.parse_args()

    from src.camera import camera_service
    from src.database import Database

    # --- Prerequisites -----------------------------------------------------
    try:
        from src.detection.face_recognition import FaceRecognitionService
    except Exception as exc:
        _say(FAIL, f"Could not import face recognition: {exc}")
        return 2

    db = Database(settings.DB_PATH)
    if not db.initialize():
        _say(FAIL, f"Database failed to initialize at {settings.DB_PATH}")
        return 2

    if not camera_service.initialize():
        _say(FAIL, "Camera failed to initialize")
        return 2
    platform = getattr(getattr(camera_service, "config", None), "platform", "unknown")
    _say(PASS, f"Camera initialized (platform={platform})")

    try:
        recognizer = FaceRecognitionService(
            db=db, similarity_threshold=settings.FACE_SIMILARITY_THRESHOLD
        )
    except Exception as exc:
        _say(FAIL, f"Face recognition unavailable: {exc}")
        _say(FAIL, "Install with: pip install insightface onnxruntime")
        camera_service.shutdown()
        return 2
    _say(PASS, f"ArcFace loaded (threshold={settings.FACE_SIMILARITY_THRESHOLD})")

    face_id = None
    exit_code = 1
    try:
        print(f"\nLook at the camera. Enrolling in {args.settle:.0f}s...")
        time.sleep(args.settle)

        # --- Enrol ---------------------------------------------------------
        frame = _capture(camera_service)
        if frame is None:
            _say(FAIL, "Camera returned no frame for enrollment")
            return 1

        face_id = recognizer.enroll_face(frame, args.name)
        if face_id is None:
            _say(FAIL, "No face detected in the enrollment frame")
            _say(FAIL, "Check lighting and that the face fills a reasonable part of the frame")
            return 1
        _say(PASS, f"Enrolled '{args.name}' as {face_id}")

        # --- Confirm it persisted -------------------------------------------
        if db.get_face(face_id) is None:
            _say(FAIL, "Face was not written to the database")
            return 1
        _say(PASS, "Embedding persisted to the database")

        # --- Recognize from a fresh capture ---------------------------------
        print("\nHold still. Re-capturing to verify recognition...")
        time.sleep(1.5)

        frame = _capture(camera_service)
        if frame is None:
            _say(FAIL, "Camera returned no frame for recognition")
            return 1

        matches = recognizer.recognize(frame)
        if not matches:
            _say(FAIL, "No face detected in the verification frame")
            return 1

        best = max(matches, key=lambda m: m.confidence)
        print(f"\n  faces in frame: {len(matches)}")
        print(f"  best similarity: {best.confidence:.4f}")
        print(f"  threshold:       {settings.FACE_SIMILARITY_THRESHOLD:.4f}")
        print(f"  margin:          {best.confidence - settings.FACE_SIMILARITY_THRESHOLD:+.4f}")

        if best.face_id != face_id:
            _say(FAIL, f"Recognized as {best.name or 'unknown'}, expected '{args.name}'")
            _say(FAIL, "Similarity fell below the threshold, or another enrolled face scored higher")
            return 1

        _say(PASS, f"Recognized '{best.name}' at {best.confidence:.4f}")
        exit_code = 0
        return 0

    finally:
        if face_id is not None and not args.keep:
            recognizer.remove_face(face_id)
            _say(PASS, f"Removed test enrollment {face_id}")
        camera_service.shutdown()
        db.shutdown()
        print(f"\n{'VERIFICATION PASSED' if exit_code == 0 else 'VERIFICATION FAILED'}")


if __name__ == "__main__":
    sys.exit(main())
