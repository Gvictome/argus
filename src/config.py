"""
Configuration settings for THE EYE
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MEDIA_DIR = BASE_DIR / "media"
CONFIG_DIR = BASE_DIR / "config"


@dataclass
class Settings:
    """Application settings"""

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    SECRET_KEY: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "dev-secret-change-in-production"))

    # Camera
    CAMERA_INDEX: int = 0
    CAMERA_RESOLUTION: tuple = (1920, 1080)
    CAMERA_FPS: int = 30
    CAMERA_ROTATION: int = 0

    # Detection
    MOTION_SENSITIVITY: int = 25
    DETECTION_THRESHOLD: float = 0.5
    # Haar-cascade face *detection* confidence placeholder (no identity).
    FACE_RECOGNITION_THRESHOLD: float = 0.6
    # ArcFace cosine-similarity threshold for face *identity* matching.
    # Typical usable range is 0.3-0.5; higher is stricter. Raising this
    # trades false accepts for false rejects, which is the safer direction
    # for a live demo. See FaceRecognitionService.similarity_threshold.
    FACE_SIMILARITY_THRESHOLD: float = 0.4

    # Threat detection (dangerous-person cascade stage).
    # Disabling this returns the pipeline to pure COCO detection + faces,
    # with no code change -- the demo's escape hatch if the stage misbehaves.
    THREAT_ENABLED: bool = True
    THREAT_MODEL_PATH: Path = BASE_DIR / "models" / "threat-yolo11n.pt"
    # Tuned, not inherited. The upstream model card demonstrates at 0.35;
    # swept against the 314-image labeled test set plus 128 ordinary COCO
    # images as negatives (scripts/tune_threat_threshold.py):
    #
    #   thresh  precision  recall     F1   false alarms per 100 ordinary scenes
    #     0.35      0.901   0.800  0.847   29.7
    #     0.55      0.919   0.768  0.837   18.8
    #     0.70      0.931   0.700  0.799   10.2
    #
    # F1 peaks near 0.30, but F1 is the wrong objective here. At 0.35
    # nearly one ordinary scene in three flags someone as armed. Pointed
    # at a booth visitor that is both a credibility problem and an
    # ethical one. 0.55 gives up 0.03 recall to cut false alarms by 37%.
    # Raise toward 0.70 for a crowded room; lower only if a missed
    # detection genuinely costs more than a false accusation.
    THREAT_CONFIDENCE: float = 0.55
    # Trained at 832, but that costs ~110ms/frame on x86 and materially more
    # on a Pi 5 CPU. 416 roughly halves it. Raise on the Orin Nano.
    THREAT_IMGSZ: int = 416

    # Storage
    DATA_DIR: Path = DATA_DIR
    MEDIA_DIR: Path = MEDIA_DIR
    DB_PATH: Path = DATA_DIR / "database.db"

    # Security
    TOKEN_EXPIRY: int = 3600  # seconds
    MAX_LOGIN_ATTEMPTS: int = 5
    # Require a bearer token on every endpoint that reads the camera or
    # mutates state.
    #
    # Defaults to false so the existing LAN demo flow is unchanged. That
    # is only defensible behind a router: with this off, anyone who can
    # reach the port can watch the camera and delete enrolled faces.
    # Anything that exposes ARGUS beyond the LAN must set this true --
    # scripts/run_tunnel.sh refuses to start without it.
    AUTH_REQUIRED: bool = False
    # Comma-separated origins allowed to call the API cross-origin. Empty
    # by default: the built-in dashboard is same-origin and needs none.
    # Set this only for a separately-hosted frontend.
    CORS_ORIGINS: str = ""

    # Automation
    AUTOMATION_ENABLED: bool = True

    # Federated Learning
    FL_ENABLED: bool = False
    FL_SERVER_URL: str = "localhost:8080"
    FL_LOCAL_EPOCHS: int = 5
    FL_MIN_SAMPLES: int = 50
    FL_ROUND_HOUR: int = 2  # 2 AM
    FL_TRAINING_DIR: Optional[Path] = None  # set in __post_init__

    def __post_init__(self):
        """Create directories if they don't exist"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MEDIA_DIR.mkdir(parents=True, exist_ok=True)

        # Set FL training dir under DATA_DIR and create it
        if self.FL_TRAINING_DIR is None:
            self.FL_TRAINING_DIR = self.DATA_DIR / "training"
        self.FL_TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables"""
        return cls(
            HOST=os.getenv("HOST", "0.0.0.0"),
            PORT=int(os.getenv("PORT", 8000)),
            DEBUG=os.getenv("DEBUG", "false").lower() == "true",
            SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret-change-in-production"),
            CAMERA_INDEX=int(os.getenv("CAMERA_INDEX", 0)),
            MOTION_SENSITIVITY=int(os.getenv("MOTION_SENSITIVITY", 25)),
            DETECTION_THRESHOLD=float(os.getenv("DETECTION_THRESHOLD", 0.5)),
            FACE_RECOGNITION_THRESHOLD=float(os.getenv("FACE_RECOGNITION_THRESHOLD", 0.6)),
            FACE_SIMILARITY_THRESHOLD=float(os.getenv("FACE_SIMILARITY_THRESHOLD", 0.4)),
            # Threat detection
            THREAT_ENABLED=os.getenv("THREAT_ENABLED", "true").lower() == "true",
            THREAT_MODEL_PATH=Path(
                os.getenv("THREAT_MODEL_PATH", BASE_DIR / "models" / "threat-yolo11n.pt")
            ),
            THREAT_CONFIDENCE=float(os.getenv("THREAT_CONFIDENCE", 0.55)),
            THREAT_IMGSZ=int(os.getenv("THREAT_IMGSZ", 416)),
            # Security
            AUTH_REQUIRED=os.getenv("AUTH_REQUIRED", "false").lower() == "true",
            CORS_ORIGINS=os.getenv("CORS_ORIGINS", ""),
            # Federated Learning
            FL_ENABLED=os.getenv("FL_ENABLED", "false").lower() == "true",
            FL_SERVER_URL=os.getenv("FL_SERVER_URL", "localhost:8080"),
            FL_LOCAL_EPOCHS=int(os.getenv("FL_LOCAL_EPOCHS", 5)),
            FL_MIN_SAMPLES=int(os.getenv("FL_MIN_SAMPLES", 50)),
            FL_ROUND_HOUR=int(os.getenv("FL_ROUND_HOUR", 2)),
        )


# Global settings instance
settings = Settings.from_env()
