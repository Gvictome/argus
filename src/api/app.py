"""
FastAPI application factory
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import settings, BASE_DIR
from src.api.routes import router as api_router
from src.detection.models import ModelManager
from src.training import LocalTrainer
from src.federated.client import ArgusFlowerClient
from src.federated.scheduler import FLScheduler


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""

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
        # TODO: Initialize camera service
        # TODO: Initialize detection service
        # TODO: Initialize automation service
        # TODO: Initialize database

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
        # TODO: Stop camera stream
        # TODO: Close database connections

        # Stop FL scheduler if running
        if hasattr(app.state, "fl_scheduler"):
            await app.state.fl_scheduler.stop()

    return app
