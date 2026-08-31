"""
API Routes for THE EYE
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List

from src.api.auth import optional_auth, require_auth

router = APIRouter()

# Applied to everything that reads the camera or mutates state. A no-op
# while AUTH_REQUIRED is false, so the LAN demo flow is unchanged.
_PROTECTED = [Depends(require_auth)]


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

def _dashboard_path():
    from src.config import BASE_DIR
    return BASE_DIR / "static" / "index.html"


@router.get("/", tags=["Health"])
async def root(request: Request):
    """
    The dashboard for a browser, the health payload for anything else.

    Content-negotiated rather than split across two paths: at a booth the
    useful behavior is that typing the Pi's address shows the demo, not
    raw JSON. API clients and the existing health checks send `*/*` or
    `application/json` and are unaffected.
    """
    from fastapi.responses import FileResponse

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        page = _dashboard_path()
        if page.exists():
            return FileResponse(page, media_type="text/html")

    return {"status": "online", "service": "THE EYE"}


@router.get("/dashboard", tags=["Health"], include_in_schema=False)
async def dashboard():
    """The dashboard at a stable path, regardless of Accept headers."""
    from fastapi.responses import FileResponse

    page = _dashboard_path()
    if not page.exists():
        raise HTTPException(status_code=404, detail="Dashboard not installed")
    return FileResponse(page, media_type="text/html")


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


@router.post("/api/camera/init", tags=["Camera"], dependencies=_PROTECTED)
async def camera_init():
    """Initialize the camera"""
    from src.camera import camera_service
    success = camera_service.initialize()
    if success:
        return {"message": "Camera initialized", "status": camera_service.get_status()}
    raise HTTPException(status_code=503, detail="Camera initialization failed")


@router.get("/api/camera/snapshot", tags=["Camera"], dependencies=_PROTECTED)
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


@router.get("/api/camera/stream", tags=["Camera"], dependencies=_PROTECTED)
async def camera_stream(request: Request, detect_every: int = 3, quality: int = 80):
    """
    Live MJPEG video stream with detection overlays burned in.

    Args:
        detect_every: Run the detection cascade every Nth frame. Raise this
            if the frame rate sags at the booth; boxes persist between runs.
        quality: JPEG quality, 0-100.
    """
    from fastapi.responses import StreamingResponse
    from src.camera import camera_service
    from src.detection import detection_service
    from src.detection.stream import stream_annotated_mjpeg

    if not camera_service.is_initialized:
        raise HTTPException(status_code=503, detail="Camera not initialized. Call /api/camera/init first")

    return StreamingResponse(
        stream_annotated_mjpeg(
            camera_service,
            detection_service,
            detect_every=detect_every,
            jpeg_quality=quality,
            recorder=getattr(request.app.state, "event_recorder", None),
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/api/camera/record", tags=["Camera"], dependencies=_PROTECTED)
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


@router.post("/api/camera/shutdown", tags=["Camera"], dependencies=_PROTECTED)
async def camera_shutdown():
    """Shutdown the camera"""
    from src.camera import camera_service
    camera_service.shutdown()
    return {"message": "Camera shutdown complete"}


# ============================================================================
# Cameras (multi-sensor)
# ============================================================================
#
# The single-camera endpoints above act on the primary sensor and are
# unchanged. These address a specific camera by name, which is what the
# Orin's two CSI connectors need.

def _registry():
    import src.camera as camera_module
    return camera_module.camera_registry


def _require_camera(name: str):
    cam = _registry().get(name)
    if cam is None:
        known = _registry().names()
        raise HTTPException(
            status_code=404,
            detail=f"No camera {name!r}. Configured: {known or 'none'}",
        )
    return cam


@router.get("/api/cameras", tags=["Camera"])
async def list_cameras():
    """Every configured camera, and what board we are on."""
    return _registry().status()


@router.post("/api/cameras/init", tags=["Camera"], dependencies=_PROTECTED)
async def init_all_cameras():
    """
    Open every configured camera.

    Reports per-camera success: with two sensors, "one failed" is the
    interesting outcome and a single boolean would hide which.
    """
    results = _registry().initialize_all()
    return {
        "results": results,
        "ok": all(results.values()) if results else False,
        "cameras": _registry().status()["cameras"],
    }


@router.get("/api/cameras/{name}/status", tags=["Camera"])
async def camera_status_by_name(name: str):
    return _require_camera(name).get_status()


@router.post("/api/cameras/{name}/init", tags=["Camera"], dependencies=_PROTECTED)
async def init_camera(name: str):
    cam = _require_camera(name)
    if cam.initialize():
        return {"message": f"Camera {name!r} initialized", "status": cam.get_status()}
    raise HTTPException(status_code=503, detail=f"Camera {name!r} failed to initialize")


@router.get("/api/cameras/{name}/snapshot", tags=["Camera"], dependencies=_PROTECTED)
async def camera_snapshot_by_name(name: str):
    from fastapi.responses import Response

    cam = _require_camera(name)
    if not cam.is_initialized:
        raise HTTPException(status_code=503, detail=f"Camera {name!r} not initialized")
    frame = cam.get_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame")
    return Response(content=frame, media_type="image/jpeg")


@router.get("/api/cameras/{name}/stream", tags=["Camera"], dependencies=_PROTECTED)
async def camera_stream_by_name(request: Request, name: str, detect_every: int = 3, quality: int = 80):
    """
    Annotated MJPEG for one named camera.

    All cameras share the one DetectionService, so its motion history and
    FPS window are shared too. That is correct for the demo -- one
    pipeline, several views -- but it does mean two simultaneous streams
    interleave their motion state. Run one at a time for a clean reading.
    """
    from fastapi.responses import StreamingResponse
    from src.detection import detection_service
    from src.detection.stream import stream_annotated_mjpeg

    cam = _require_camera(name)
    if not cam.is_initialized:
        raise HTTPException(status_code=503, detail=f"Camera {name!r} not initialized")

    return StreamingResponse(
        stream_annotated_mjpeg(cam, detection_service,
                               detect_every=detect_every, jpeg_quality=quality,
                               recorder=getattr(request.app.state, "event_recorder", None)),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/api/cameras/{name}/shutdown", tags=["Camera"], dependencies=_PROTECTED)
async def shutdown_camera(name: str):
    _require_camera(name).shutdown()
    return {"message": f"Camera {name!r} shut down"}


# ============================================================================
# Federated Learning
# ============================================================================

@router.get("/api/recordings", tags=["Camera"], dependencies=_PROTECTED)
async def list_recordings(request: Request, limit: int = 20):
    """
    Event clips saved so far.

    Cameras run continuously; footage is written only around detections,
    so this is the list of things that actually happened rather than a
    directory of hours nobody will watch.
    """
    recorder = getattr(request.app.state, "event_recorder", None)
    if recorder is None:
        from src.config import settings
        return {
            "enabled": settings.RECORD_EVENTS,
            "clips": [],
            "detail": "Event recording is not active",
        }
    return {
        "enabled": True,
        "status": recorder.status(),
        "clips": recorder.recent_clips(limit),
    }


@router.get("/api/federated/status", tags=["Federated"])
async def federated_status(request: Request):
    """
    FL schedule state: when the last round ran and when the next is due.

    Backs the poster claim that rounds happen on a cadence. Without a
    visible next-due time, "it runs every two weeks" is unfalsifiable.
    """
    scheduler = getattr(request.app.state, "fl_scheduler", None)
    if scheduler is None:
        from src.config import settings
        return {
            "enabled": settings.FL_ENABLED,
            "running": False,
            "interval_days": settings.FL_ROUND_INTERVAL_DAYS,
            "detail": "Scheduler not started (FL_ENABLED is false)",
        }
    return scheduler.status()


# ============================================================================
# Detection
# ============================================================================

@router.get("/api/detection/status", tags=["Detection"])
async def detection_status():
    """Get detection service status: FPS, backend, and which models are live."""
    from src.detection import detection_service
    return detection_service.status()


@router.get("/api/faces", tags=["Detection"], dependencies=_PROTECTED)
async def list_faces():
    """List all known faces"""
    try:
        from src.api.app import get_face_recognizer
        recognizer = get_face_recognizer()
        if recognizer is None:
            return {"faces": [], "count": 0, "error": "Face recognition not initialized"}
        faces = recognizer.list_known_faces()
        return {"faces": faces, "count": len(faces)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/faces", tags=["Detection"], dependencies=_PROTECTED)
async def add_face(name: str):
    """Enroll a new face by capturing a frame from the camera"""
    try:
        from src.camera import camera_service
        from src.api.app import get_face_recognizer

        recognizer = get_face_recognizer()
        if recognizer is None:
            raise HTTPException(status_code=503, detail="Face recognition not initialized")

        if not camera_service.is_initialized:
            raise HTTPException(status_code=503, detail="Camera not initialized")

        import numpy as np
        frame_bytes = camera_service.get_frame()
        if frame_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to capture frame")

        # Decode JPEG bytes to numpy array
        frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = __import__("cv2").imdecode(frame, __import__("cv2").IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to decode frame")

        face_id = recognizer.enroll_face(frame, name)
        if face_id is None:
            raise HTTPException(status_code=400, detail="No face detected in frame")

        return {"message": f"Face enrolled for {name}", "face_id": face_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/faces/reset", tags=["Detection"], dependencies=_PROTECTED)
async def reset_faces():
    """
    Clear every enrolled face (showcase P1-6).

    Declared ahead of the /api/faces/{face_id} route so "reset" is never
    read as a face id.
    """
    try:
        from src.api.app import get_face_recognizer
        recognizer = get_face_recognizer()
        if recognizer is None:
            raise HTTPException(status_code=503, detail="Face recognition not initialized")
        removed = recognizer.reset()
        return {"message": f"Cleared {removed} enrolled face(s)", "removed": removed}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/faces/{face_id}", tags=["Detection"], dependencies=_PROTECTED)
async def remove_face(face_id: str):
    """Remove a face from the database"""
    try:
        from src.api.app import get_face_recognizer
        recognizer = get_face_recognizer()
        if recognizer is None:
            raise HTTPException(status_code=503, detail="Face recognition not initialized")
        removed = recognizer.remove_face(face_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Face {face_id} not found")
        return {"message": f"Face {face_id} removed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@router.post("/api/devices/{device_id}", tags=["Automation"], dependencies=_PROTECTED)
async def control_device(device_id: str, action: str, value: Optional[str] = None):
    """Control a device"""
    # TODO: Implement device control
    return {"device_id": device_id, "action": action, "value": value, "success": False}


@router.get("/api/automations", tags=["Automation"])
async def list_automations():
    """List all automation rules"""
    return {"automations": [], "count": 0}


@router.post("/api/automations", tags=["Automation"], dependencies=_PROTECTED)
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

class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/api/auth/login", tags=["Security"])
async def login(body: LoginRequest, request: Request):
    """
    Exchange credentials for a bearer token.

    Credentials come in a JSON body, not query parameters: query strings
    land in server logs, browser history, and proxy logs, and this one
    would carry the password in plain text.
    """
    from src.api.auth import get_security

    security = get_security()
    if security is None:
        raise HTTPException(status_code=503, detail="Authentication is not initialized")

    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    user = db.get_user_by_username(body.username)
    client = request.client.host if request.client else None

    # One message and one code for "no such user" and "wrong password".
    # Distinguishing them tells an attacker which usernames are real.
    if user is None or not security.verify_password(body.password, user["password_hash"]):
        try:
            db.log_audit(body.username, "login_failed", "bad credentials", client)
        except Exception:
            pass
        raise HTTPException(status_code=401, detail="Invalid username or password")

    from src.config import settings
    token = security.generate_token(
        user_id=user["id"],
        expiry_hours=max(1, settings.TOKEN_EXPIRY // 3600),
    )
    try:
        db.log_audit(body.username, "login", "success", client)
    except Exception:
        pass

    return {
        "token": token,
        "token_type": "bearer",
        "username": user["username"],
        "role": user.get("role", "user"),
        "expires_in": settings.TOKEN_EXPIRY,
    }


@router.post("/api/auth/logout", tags=["Security"])
async def logout(request: Request):
    """Revoke the presented token."""
    from src.api.auth import get_security

    security = get_security()
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else None
    token = token or request.query_params.get("token")

    if security is not None and token:
        security.revoke_token(token)

    return {"message": "Logged out"}


@router.get("/api/auth/me", tags=["Security"])
async def whoami(user_id: Optional[str] = Depends(optional_auth)):
    """
    Whether auth is enforced, and whether the caller is authenticated.

    Deliberately never returns 401. This endpoint's whole job is to let a
    caller discover the auth state, and guarding it would make "auth is
    off" indistinguishable from "your token expired" -- which is exactly
    the confusion the dashboard and the tunnel preflight need to resolve.
    """
    from src.config import settings

    return {
        "auth_required": settings.AUTH_REQUIRED,
        "authenticated": user_id is not None,
        "user_id": user_id,
    }


@router.get("/api/logs", tags=["Security"], dependencies=_PROTECTED)
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
