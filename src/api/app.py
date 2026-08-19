"""
FastAPI application factory
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import settings, BASE_DIR
from src.api.routes import router as api_router
from src.detection.models import ModelManager
from src.training import LocalTrainer
from src.federated.client import ArgusFlowerClient
from src.federated.scheduler import FLScheduler

logger = logging.getLogger(__name__)

# Module-level reference for route access
_face_recognizer = None


def get_face_recognizer():
    """Get the initialized FaceRecognitionService (or None)."""
    return _face_recognizer


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    global _face_recognizer

    app = FastAPI(
        title="THE EYE",
        description="Offline-First Smart Home Security & Automation System",
        version="0.1.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )

    # CORS middleware for local network access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files
    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include API routes
    app.include_router(api_router)

    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup"""
        global _face_recognizer

        # Initialize database
        from src.database import Database
        db = Database(settings.DB_PATH)
        db.initialize()
        app.state.db = db

        # Initialize detection service (motion -> YOLO -> faces)
        from src.detection import detection_service
        detection_service.initialize()
        app.state.detection_service = detection_service

        # Initialize face recognition service
        try:
            from src.detection.face_recognition import FaceRecognitionService
            _face_recognizer = FaceRecognitionService(
                db=db,
                similarity_threshold=settings.FACE_SIMILARITY_THRESHOLD,
            )
            app.state.face_recognizer = _face_recognizer
            # Without this the pipeline silently falls back to Haar cascade
            # detection and never resolves identity.
            detection_service.attach_face_recognizer(_face_recognizer)
        except Exception as exc:
            # insightface/onnxruntime missing, or model download failed.
            # Detection still runs; identity matching is disabled.
            _face_recognizer = None
            app.state.face_recognizer = None
            logger.warning("Face recognition unavailable: %s", exc)

        # Start Federated Learning scheduler if enabled
        if settings.FL_ENABLED:
            model_manager = ModelManager()
            model_manager.load_model()
            trainer = LocalTrainer(model_manager, settings)
            client = ArgusFlowerClient(model_manager, trainer, settings)
            app.state.fl_scheduler = FLScheduler(client, settings)
            await app.state.fl_scheduler.start()

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        # Shut down detection service
        if hasattr(app.state, "detection_service"):
            app.state.detection_service.shutdown()

        # Close database
        if hasattr(app.state, "db"):
            app.state.db.shutdown()

        # Stop FL scheduler if running
        if hasattr(app.state, "fl_scheduler"):
            await app.state.fl_scheduler.stop()

    return app
