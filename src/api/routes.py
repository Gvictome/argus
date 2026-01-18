"""
API Routes for THE EYE
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()


# ============================================================================
# Models
# ============================================================================

class StatusResponse(BaseModel):
    status: str
    version: str
    uptime: Optional[str] = None
    services: dict


class DeviceState(BaseModel):
    id: str
    name: str
    type: str
    state: dict
    online: bool


class EventResponse(BaseModel):
    id: str
    type: str
    timestamp: datetime
    source: str
    data: dict


# ============================================================================
# Health & Status
# ============================================================================

@router.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {"status": "online", "service": "THE EYE"}


@router.get("/api/status", response_model=StatusResponse, tags=["Health"])
async def get_status():
    """Get system status"""
    return StatusResponse(
        status="operational",
        version="0.1.0",
        services={
            "camera": "stopped",
            "detection": "stopped",
            "automation": "stopped",
            "security": "active"
        }
    )


# ============================================================================
# Camera
# ============================================================================

@router.get("/api/camera/status", tags=["Camera"])
async def camera_status():
    """Get camera status"""
    from src.camera import camera_service
    return camera_service.get_status()


@router.post("/api/camera/init", tags=["Camera"])
async def camera_init():
    """Initialize the camera"""
    from src.camera import camera_service
    success = camera_service.initialize()
    if success:
        return {"message": "Camera initialized", "status": camera_service.get_status()}
    raise HTTPException(status_code=503, detail="Camera initialization failed")


@router.get("/api/camera/snapshot", tags=["Camera"])
async def camera_snapshot():
    """Capture and return current frame as JPEG"""
    from fastapi.responses import Response
    from src.camera import camera_service

    if not camera_service.is_initialized:
        raise HTTPException(status_code=503, detail="Camera not initialized. Call /api/camera/init first")

    frame = camera_service.get_frame()
    if frame:
        return Response(content=frame, media_type="image/jpeg")
    raise HTTPException(status_code=500, detail="Failed to capture frame")


@router.get("/api/camera/stream", tags=["Camera"])
async def camera_stream():
    """Live MJPEG video stream"""
    from fastapi.responses import StreamingResponse
    from src.camera import camera_service

    if not camera_service.is_initialized:
        raise HTTPException(status_code=503, detail="Camera not initialized. Call /api/camera/init first")

    return StreamingResponse(
        camera_service.stream_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/api/camera/record", tags=["Camera"])
async def camera_record(start: bool = True, filename: Optional[str] = None):
    """Start or stop recording"""
    from pathlib import Path
    from datetime import datetime
    from src.camera import camera_service
    from src.config import settings

    if not camera_service.is_initialized:
        raise HTTPException(status_code=503, detail="Camera not initialized")

    if start:
        if not filename:
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = settings.MEDIA_DIR / filename
        success = camera_service.start_recording(output_path)
        if success:
            return {"recording": True, "file": str(output_path)}
        raise HTTPException(status_code=500, detail="Failed to start recording")
    else:
        camera_service.stop_recording()
        return {"recording": False, "message": "Recording stopped"}


@router.post("/api/camera/shutdown", tags=["Camera"])
async def camera_shutdown():
    """Shutdown the camera"""
    from src.camera import camera_service
    camera_service.shutdown()
    return {"message": "Camera shutdown complete"}


# ============================================================================
# Detection
# ============================================================================

@router.get("/api/detection/status", tags=["Detection"])
async def detection_status():
    """Get detection service status"""
    return {
        "status": "stopped",
        "motion_detection": False,
        "face_recognition": False,
        "object_detection": False
    }


@router.get("/api/faces", tags=["Detection"])
async def list_faces():
    """List all known faces"""
    # TODO: Query database for faces
    return {"faces": [], "count": 0}


@router.post("/api/faces", tags=["Detection"])
async def add_face(name: str):
    """Add a new face to the database"""
    # TODO: Implement face enrollment
    return {"message": f"Face enrollment started for {name}"}


# ============================================================================
# Automation
# ============================================================================

@router.get("/api/devices", response_model=List[DeviceState], tags=["Automation"])
async def list_devices():
    """List all connected devices"""
    # TODO: Query device registry
    return []


@router.get("/api/devices/{device_id}", tags=["Automation"])
async def get_device(device_id: str):
    """Get device details"""
    # TODO: Query device by ID
    raise HTTPException(status_code=404, detail=f"Device {device_id} not found")


@router.post("/api/devices/{device_id}", tags=["Automation"])
async def control_device(device_id: str, action: str, value: Optional[str] = None):
    """Control a device"""
    # TODO: Implement device control
    return {"device_id": device_id, "action": action, "value": value, "success": False}


@router.get("/api/automations", tags=["Automation"])
async def list_automations():
    """List all automation rules"""
    return {"automations": [], "count": 0}


@router.post("/api/automations", tags=["Automation"])
async def create_automation(name: str, trigger: dict, action: dict):
    """Create a new automation rule"""
    # TODO: Implement automation creation
    return {"message": f"Automation '{name}' created", "id": "auto-001"}


# ============================================================================
# Events
# ============================================================================

@router.get("/api/events", tags=["Events"])
async def list_events(limit: int = 50, offset: int = 0, event_type: Optional[str] = None):
    """Get event history"""
    # TODO: Query events from database
    return {"events": [], "count": 0, "limit": limit, "offset": offset}


@router.get("/api/events/{event_id}", tags=["Events"])
async def get_event(event_id: str):
    """Get event details"""
    # TODO: Query event by ID
    raise HTTPException(status_code=404, detail=f"Event {event_id} not found")


# ============================================================================
# Security
# ============================================================================

@router.post("/api/auth/login", tags=["Security"])
async def login(username: str, password: str):
    """Authenticate user"""
    # TODO: Implement authentication
    return {"message": "Authentication not implemented", "token": None}


@router.post("/api/auth/logout", tags=["Security"])
async def logout():
    """End session"""
    return {"message": "Logged out"}


@router.get("/api/logs", tags=["Security"])
async def get_audit_logs(limit: int = 100):
    """Get audit logs"""
    # TODO: Query audit log
    return {"logs": [], "count": 0}


# ============================================================================
# Error Router (Agent System)
# ============================================================================

class ErrorRequest(BaseModel):
    error_message: str
    auto_fix: bool = False


class ErrorAnalysis(BaseModel):
    category: str
    subcategory: Optional[str]
    file_path: Optional[str]
    line_number: Optional[int]
    suggested_fix: Optional[str]
    auto_fixable: bool


class FixResponse(BaseModel):
    success: bool
    action_taken: str
    output: Optional[str]
    error: Optional[str]


@router.post("/api/errors/analyze", response_model=ErrorAnalysis, tags=["Error Agents"])
async def analyze_error(request: ErrorRequest):
    """
    Analyze an error message and get fix suggestions

    The error router identifies the error type and delegates
    to specialized agents for targeted solutions.
    """
    try:
        from src.agents.error_router import error_router
        report = error_router.analyze(request.error_message)
        return ErrorAnalysis(
            category=report.category.value,
            subcategory=report.subcategory,
            file_path=report.file_path,
            line_number=report.line_number,
            suggested_fix=report.suggested_fix,
            auto_fixable=report.auto_fixable
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/errors/fix", response_model=FixResponse, tags=["Error Agents"])
async def fix_error(request: ErrorRequest):
    """
    Attempt to automatically fix an error

    Only works for errors marked as auto_fixable.
    Set auto_fix=True to allow the agent to execute fixes.
    """
    try:
        from src.agents.error_router import error_router
        result = error_router.fix(request.error_message, auto_approve=request.auto_fix)
        return FixResponse(
            success=result.success,
            action_taken=result.action_taken,
            output=result.output,
            error=result.error
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/errors/history", tags=["Error Agents"])
async def get_error_history(limit: int = 10):
    """Get recent error analysis history"""
    try:
        from src.agents.error_router import error_router
        history = error_router.get_history(limit)
        return {
            "errors": [
                {
                    "category": e.category.value,
                    "subcategory": e.subcategory,
                    "suggested_fix": e.suggested_fix,
                    "auto_fixable": e.auto_fixable,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in history
            ],
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/agents/status", tags=["Error Agents"])
async def get_agents_status():
    """Get status of all error-solving agents"""
    return {
        "agents": [
            {"name": "DependencyAgent", "category": "dependency", "status": "active"},
            {"name": "NetworkAgent", "category": "network", "status": "active"},
            {"name": "SyntaxAgent", "category": "syntax", "status": "active"},
            {"name": "HardwareAgent", "category": "hardware", "status": "active"},
        ],
        "router": "active"
    }
