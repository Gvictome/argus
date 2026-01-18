"""
THE EYE - Smart Home Security System
Main application entry point
"""

import uvicorn
from src.api.app import create_app
from src.config import settings

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )
