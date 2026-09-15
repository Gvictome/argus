"""
FastAPI application factory
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import settings, BASE_DIR
from src.api.routes import router as api_router
from src.api.trial_routes import router as trial_router
from src.api.dashboard_routes import router as dashboard_router
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
    # allow_origins=["*"] with allow_credentials=True is rejected by every
    # browser, and would be unsafe if it were not. The dashboard is served
    # from this same origin, so it needs no CORS grant at all; CORS_ORIGINS
    # exists for a separately-hosted frontend (the Next.js dashboard).
    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # Static files
    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include API routes
    app.include_router(api_router)
    # Federated-trial endpoints and the dashboard compatibility aliases.
    app.include_router(trial_router)
    # adam9798/argus-dashboard backend contract.
    app.include_router(dashboard_router)

    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup"""
        global _face_recognizer

        # Initialize database
        from src.database import Database
        db = Database(settings.DB_PATH)
        db.initialize()
        app.state.db = db

        # Authentication. Must come before anything serves a route, and
        # it bootstraps the admin account on a fresh install.
        from src.api.auth import initialize_auth
        initialize_auth(db)

        # Cameras. The registry holds every configured sensor; on the
        # Orin that is both CSI connectors. Devices are not opened here --
        # startup must not block on hardware, and a camera that is absent
        # should not stop the API from serving.
        import src.camera as camera_module
        from src.camera import build_registry_from_settings
        camera_module.camera_registry = build_registry_from_settings(settings)
        app.state.camera_registry = camera_module.camera_registry
        logger.info(
            "Camera registry: %s on %s",
            camera_module.camera_registry.names() or "none configured",
            camera_module.BOARD.value,
        )

        # Initialize detection service (motion -> YOLO -> faces -> threat)
        from src.detection import detection_service, DetectionConfig
        # Without this the env vars in Settings never reach the pipeline:
        # DetectionConfig's dataclass defaults silently won every time.
        detection_service.config = DetectionConfig(
            motion_threshold=settings.MOTION_SENSITIVITY,
            detection_threshold=settings.DETECTION_THRESHOLD,
            face_recognition_threshold=settings.FACE_RECOGNITION_THRESHOLD,
        )
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

        # Initialize threat classification (dangerous-person stage)
        app.state.threat_classifier = None
        if settings.THREAT_ENABLED:
            try:
                from src.detection.threat import ThreatClassifier
                _threat = ThreatClassifier(
                    model_path=settings.THREAT_MODEL_PATH,
                    confidence=settings.THREAT_CONFIDENCE,
                    imgsz=settings.THREAT_IMGSZ,
                )
                app.state.threat_classifier = _threat
                detection_service.attach_threat_classifier(_threat)
            except Exception as exc:
                # Weights not fetched, or ultralytics missing. Object
                # detection and face recognition carry the demo unchanged.
                logger.warning("Threat detection unavailable: %s", exc)
        else:
            logger.info("Threat detection disabled via THREAT_ENABLED")

        # Event-triggered recording. Cameras run 24/7; footage is kept
        # only around detections.
        app.state.event_recorder = None
        if settings.RECORD_EVENTS:
            try:
                from src.detection.event_recorder import EventRecorder, RecorderConfig
                app.state.event_recorder = EventRecorder(
                    RecorderConfig(
                        pre_roll_s=settings.RECORD_PRE_ROLL_S,
                        post_roll_s=settings.RECORD_POST_ROLL_S,
                        max_clip_s=settings.RECORD_MAX_CLIP_S,
                    ),
                    camera_name="primary",
                )
                logger.info(
                    "Event recording on — %.0fs pre-roll, %.0fs post-roll",
                    settings.RECORD_PRE_ROLL_S, settings.RECORD_POST_ROLL_S,
                )
            except Exception as exc:
                logger.warning("Event recording unavailable: %s", exc)
        else:
            logger.info("Event recording disabled via RECORD_EVENTS")

        # Federated head, sample store, and the collector that feeds them.
        # These come up regardless of FL_ENABLED: capturing and labelling
        # samples is useful long before any round is scheduled, and the
        # trial endpoints 503 without them.
        try:
            from src.federated.head import FederatedHead
            from src.federated.store import SampleStore
            from src.federated.collector import EventCollector

            app.state.fl_head = FederatedHead(seed=0)
            app.state.fl_store = SampleStore(
                Path(settings.FL_TRAINING_DIR) / "samples.jsonl"
            )
            app.state.fl_collector = EventCollector(
                store=app.state.fl_store, db=db,
            )
            detection_service.attach_collector(app.state.fl_collector)
            logger.info(
                "Federated head ready: %d params, %d samples on disk",
                app.state.fl_head.n_params,
                app.state.fl_store.stats()["total"],
            )
        except Exception as exc:
            # The detection pipeline must still run if this fails.
            logger.warning("Federated head/collector unavailable: %s", exc)
            app.state.fl_head = None
            app.state.fl_store = None
            app.state.fl_collector = None

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

        # Finalize any clip in progress, or it is left truncated.
        recorder = getattr(app.state, "event_recorder", None)
        if recorder is not None:
            recorder.close()

        # Release every camera. A CSI sensor left open blocks the next
        # process from acquiring it, which turns one crash into a reboot.
        if hasattr(app.state, "camera_registry"):
            app.state.camera_registry.shutdown_all()

        # Close database
        if hasattr(app.state, "db"):
            app.state.db.shutdown()

        # Stop FL scheduler if running
        if hasattr(app.state, "fl_scheduler"):
            await app.state.fl_scheduler.stop()

    return app
