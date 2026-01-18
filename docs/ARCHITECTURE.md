# System Architecture

## Overview

THE EYE is a modular, event-driven security and automation system. All processing happens locally on the Raspberry Pi 5, with optional hardware acceleration via the AI Hat+ 2.

## System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                          │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │   Streamlit   │  │   REST API    │  │  WebSocket    │        │
│  │   Dashboard   │  │   Endpoints   │  │   Streams     │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
├─────────────────────────────────────────────────────────────────┤
│                      ORCHESTRATION LAYER                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Sentinel Prime                         │    │
│  │              (Master Agent Orchestrator)                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │              │              │              │             │
│  ┌────┴────┐   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐        │
│  │Engineer │   │  Code   │   │Sentinel │   │  Docu   │        │
│  │   Bot   │   │  Mage   │   │  Guard  │   │   Bot   │        │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘        │
├─────────────────────────────────────────────────────────────────┤
│                       SERVICE LAYER                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Camera  │ │Detection │ │Automation│ │ Security │           │
│  │ Service  │ │ Service  │ │ Service  │ │ Service  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
├─────────────────────────────────────────────────────────────────┤
│                        DATA LAYER                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  SQLite  │ │  Media   │ │  Config  │ │   Logs   │           │
│  │    DB    │ │  Store   │ │  Store   │ │  Store   │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
├─────────────────────────────────────────────────────────────────┤
│                      HARDWARE LAYER                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Camera  │ │   GPIO   │ │ AI Hat+  │ │ Network  │           │
│  │ Module 3 │ │   Pins   │ │    2     │ │Interface │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Agent System

### Sentinel Prime (Master Orchestrator)

**Role:** Central coordinator for all operations

**Responsibilities:**
- Receives events from services
- Delegates tasks to sub-agents
- Aggregates responses
- Makes final decisions
- Maintains system state

**Communication Protocol:**
```python
class AgentMessage:
    source: str          # Sending agent
    target: str          # Receiving agent
    action: str          # Task type
    payload: dict        # Task data
    priority: int        # 1-10 (10 = urgent)
    timestamp: datetime
```

### Sub-Agents

#### Engineer-Bot
- **Domain:** Hardware, GPIO, sensors, power
- **Triggers:** Sensor readings, hardware events
- **Actions:** Read sensors, control relays, monitor power

#### Code-Mage
- **Domain:** Python, FastAPI, AI models
- **Triggers:** Detection events, API requests
- **Actions:** Process frames, run inference, update models

#### Sentinel-Guard
- **Domain:** Security, encryption, access control
- **Triggers:** Auth requests, suspicious activity
- **Actions:** Encrypt data, validate access, log events

#### DocuBot
- **Domain:** Documentation, logging, reporting
- **Triggers:** System events, user requests
- **Actions:** Generate reports, update logs, create docs

## Data Flow

### Camera Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Camera    │────▶│   Frame     │────▶│   Motion    │
│   Capture   │     │   Buffer    │     │  Detection  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                    ┌─────────────────────────────────────┐
                    │         Motion Detected?             │
                    └─────────────────────────────────────┘
                           │ YES                    │ NO
                           ▼                        ▼
                    ┌─────────────┐          ┌─────────────┐
                    │   Object    │          │   Continue  │
                    │Classification│          │  Monitoring │
                    └─────────────┘          └─────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │    Human    │ │   Animal    │ │   Vehicle   │
    │  Detected   │ │  Detected   │ │  Detected   │
    └─────────────┘ └─────────────┘ └─────────────┘
           │               │               │
           ▼               │               │
    ┌─────────────┐        │               │
    │    Face     │        │               │
    │ Recognition │        │               │
    └─────────────┘        │               │
           │               │               │
    ┌──────┴──────┐        │               │
    │             │        │               │
    ▼             ▼        ▼               ▼
┌───────┐   ┌───────┐ ┌───────┐      ┌───────┐
│ Known │   │Unknown│ │  Log  │      │  Log  │
│  Log  │   │ Alert │ │ Type  │      │ Event │
└───────┘   └───────┘ └───────┘      └───────┘
```

### Event Processing

```python
# Event types
class EventType(Enum):
    MOTION_DETECTED = "motion_detected"
    FACE_RECOGNIZED = "face_recognized"
    FACE_UNKNOWN = "face_unknown"
    ANIMAL_DETECTED = "animal_detected"
    VEHICLE_DETECTED = "vehicle_detected"
    AUTOMATION_TRIGGER = "automation_trigger"
    SECURITY_ALERT = "security_alert"
    SYSTEM_STATUS = "system_status"

# Event structure
class Event:
    id: str
    type: EventType
    timestamp: datetime
    source: str
    data: dict
    media: Optional[bytes]
```

## Service Definitions

### Camera Service (`src/camera/`)

**Purpose:** Capture and stream video frames

**Key Functions:**
- `start_capture()` - Initialize camera
- `get_frame()` - Retrieve current frame
- `start_stream()` - Begin continuous streaming
- `save_snapshot()` - Save single image
- `record_clip()` - Save video segment

**Configuration:**
```python
CAMERA_CONFIG = {
    "resolution": (1920, 1080),
    "framerate": 30,
    "format": "RGB",
    "rotation": 0,
    "autofocus": True,
    "hdr": False
}
```

### Detection Service (`src/detection/`)

**Purpose:** Analyze frames for motion, objects, faces

**Models:**
| Model | Purpose | Source |
|-------|---------|--------|
| MobileNet-SSD | Object detection | TensorFlow Hub |
| FaceNet | Face recognition | Hugging Face |
| Custom CNN | Motion detection | Local |

**Key Functions:**
- `detect_motion(frame, prev_frame)` - Compare frames
- `detect_objects(frame)` - Classify objects in frame
- `recognize_face(face_crop)` - Match face to database
- `encode_face(face_crop)` - Generate face embedding

### Automation Service (`src/automation/`)

**Purpose:** Control home devices

**Supported Devices:**
- Lights (on/off, brightness, color)
- Locks (lock/unlock, status)
- Thermostats (temperature, mode)
- Cameras (PTZ, recording)

**Integration Options:**
- Direct GPIO control
- HTTP/REST APIs
- MQTT messaging
- Home Assistant API

**Key Functions:**
- `get_devices()` - List all devices
- `get_device_state(id)` - Get device status
- `set_device_state(id, state)` - Control device
- `create_automation(trigger, action)` - Define rule

### Security Service (`src/security/`)

**Purpose:** Protect data and access

**Features:**
- AES-256 encryption for stored data
- Bcrypt hashing for passwords
- JWT tokens for API auth
- Rate limiting
- Audit logging

**Key Functions:**
- `encrypt(data)` - Encrypt sensitive data
- `decrypt(data)` - Decrypt data
- `hash_password(password)` - Secure password storage
- `verify_token(token)` - Validate API token
- `log_event(event)` - Audit trail

## API Endpoints

### Core Routes

```
GET  /                      # Health check
GET  /api/status            # System status

# Camera
GET  /api/camera/stream     # Live video stream (MJPEG)
GET  /api/camera/snapshot   # Current frame
POST /api/camera/record     # Start/stop recording

# Detection
GET  /api/detection/status  # Detection service status
POST /api/detection/analyze # Analyze uploaded image
GET  /api/faces             # List known faces
POST /api/faces             # Add new face

# Automation
GET  /api/devices           # List devices
GET  /api/devices/{id}      # Device details
POST /api/devices/{id}      # Control device
GET  /api/automations       # List automations
POST /api/automations       # Create automation

# Events
GET  /api/events            # Event history
GET  /api/events/{id}       # Event details
WS   /api/events/stream     # Real-time events

# Security
POST /api/auth/login        # Authenticate
POST /api/auth/logout       # End session
GET  /api/logs              # Audit logs
```

## Database Schema

### SQLite Tables

```sql
-- Known faces
CREATE TABLE faces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP
);

-- Events log
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source TEXT,
    data JSON,
    media_path TEXT
);

-- Devices
CREATE TABLE devices (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    config JSON,
    state JSON,
    last_updated TIMESTAMP
);

-- Automations
CREATE TABLE automations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    trigger JSON NOT NULL,
    action JSON NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP
);

-- Audit log
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user TEXT,
    action TEXT,
    details JSON
);
```

## Configuration

### Environment Variables

```bash
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Camera
CAMERA_INDEX=0
CAMERA_RESOLUTION=1080p
CAMERA_FPS=30

# Detection
DETECTION_THRESHOLD=0.5
MOTION_SENSITIVITY=25
FACE_RECOGNITION_THRESHOLD=0.6

# Security
SECRET_KEY=<generate-random-key>
TOKEN_EXPIRY=3600
ENCRYPTION_KEY=<generate-random-key>

# Storage
DATA_DIR=/home/pi/the-eye/data
MEDIA_DIR=/home/pi/the-eye/media
DB_PATH=/home/pi/the-eye/database.db
```

## Deployment

### Local LAN

1. Configure static IP
2. Open port 8000 in firewall
3. Run FastAPI with uvicorn
4. Access via `http://<pi-ip>:8000`

### Systemd Service

```ini
[Unit]
Description=THE EYE Security System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/the-eye
ExecStart=/home/pi/the-eye/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Performance Considerations

### CPU Optimization
- Use threading for I/O operations
- Batch inference requests
- Reduce frame resolution for detection
- Skip frames during low activity

### Memory Management
- Stream frames instead of buffering
- Clear old media files periodically
- Use generators for large datasets
- Monitor with `htop` / `free -m`

### AI Hat+ 2 Acceleration
- Offload TensorFlow models to NPU
- Use Hailo Model Zoo for optimized models
- Benchmark CPU vs NPU inference times
