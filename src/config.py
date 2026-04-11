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
    FACE_RECOGNITION_THRESHOLD: float = 0.6

    # Storage
    DATA_DIR: Path = DATA_DIR
    MEDIA_DIR: Path = MEDIA_DIR
    DB_PATH: Path = DATA_DIR / "database.db"

    # Security
    TOKEN_EXPIRY: int = 3600  # seconds
    MAX_LOGIN_ATTEMPTS: int = 5

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
            # Federated Learning
            FL_ENABLED=os.getenv("FL_ENABLED", "false").lower() == "true",
            FL_SERVER_URL=os.getenv("FL_SERVER_URL", "localhost:8080"),
            FL_LOCAL_EPOCHS=int(os.getenv("FL_LOCAL_EPOCHS", 5)),
            FL_MIN_SAMPLES=int(os.getenv("FL_MIN_SAMPLES", 50)),
            FL_ROUND_HOUR=int(os.getenv("FL_ROUND_HOUR", 2)),
        )


# Global settings instance
settings = Settings.from_env()
