#!/usr/bin/env python3
"""
Run the full node against a recording, with an isolated mock database.

Everything is real -- capture, motion, YOLO, tracking, event storage, the
dashboard API, federated rounds -- except the camera, which is a video
file, and the data directory, which is data/mock so the test never touches
real events or a real model.

    python scripts/mock_demo.py --reset                # media/vtest.avi
    python scripts/mock_demo.py --source 0             # a webcam instead
    python scripts/mock_demo.py --threat               # include the threat stage

Then point the dashboard at it (NEXT_PUBLIC_API_BASE_URL=http://localhost:8000)
and log in as admin with --password (default: argus-demo).
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(REPO / "media" / "vtest.avi"),
                    help="video file path, or a capture index such as 0")
    ap.add_argument("--data", default="data/mock")
    ap.add_argument("--password", default="argus-demo",
                    help="admin password, applied when the mock DB is created")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--origins", default="http://localhost:3000")
    ap.add_argument("--reset", action="store_true", help="wipe the mock data first")
    ap.add_argument("--threat", action="store_true", help="enable the threat stage")
    ap.add_argument("--record", action="store_true", help="write event clips")
    args = ap.parse_args()

    source = args.source
    if not source.isdigit() and not Path(source).exists():
        print(f"no such video: {source}")
        return 2

    data = (REPO / args.data).resolve()
    if args.reset and data.exists():
        shutil.rmtree(data)

    # Must be set before src.config is imported: settings are read once.
    os.environ.update({
        "ARGUS_DATA_DIR": str(data),
        "CAMERA_SOURCE": source,
        "CAMERA_SOURCE_REALTIME": "true",
        "ARGUS_ADMIN_PASSWORD": args.password,
        "CORS_ORIGINS": args.origins,
        "AUTH_REQUIRED": "false",
        "FL_ENABLED": "false",
        "DETECTION_AUTOSTART": "true",
        "FACE_RECOGNITION_ENABLED": "false",
        "THREAT_ENABLED": "true" if args.threat else "false",
        "RECORD_EVENTS": "true" if args.record else "false",
        "CAMERA_ZONE": os.environ.get("CAMERA_ZONE", "Mock camera"),
    })

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import uvicorn

    print(f"mock node: source={source} data={data} http://{args.host}:{args.port}")
    uvicorn.run("src.api.app:create_app", factory=True, host=args.host,
                port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
