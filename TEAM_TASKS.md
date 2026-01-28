# ARGUS - Team Task Delegation
## Senior Design Project - Task Assignments

---

## Team Members

| Name | Role | Focus Areas |
|------|------|-------------|
| **Christian** | Electrical Engineer | Hardware, power, sensors, PCB, wiring |
| **Giovanny** | Computer Engineer | AI/ML, detection pipeline, optimization |
| **Mohammed** | Computer Engineer | Backend, API, integrations, database |

---

## Phase 1: Foundation (Weeks 1-4)

### Christian (Electrical Engineer)
- [ ] **Hardware Assembly & Setup**
  - [ ] Raspberry Pi 5 initial setup and configuration
  - [ ] AI HAT+ physical installation and thermal management
  - [ ] Camera Module 3 mounting and cable routing
  - [ ] Design power distribution system (27W PSU requirements)
  - [ ] GPIO pin mapping and documentation

- [ ] **Sensor Integration**
  - [ ] PIR sensor (HC-SR501) wiring to GPIO 17
  - [ ] Door/window sensor (MC-38) wiring to GPIO 27
  - [ ] IR illuminator circuit design (850nm LED array)
  - [ ] Create wiring schematic diagram

- [ ] **Enclosure Design**
  - [ ] Evaluate Argon ONE V3 vs custom enclosure
  - [ ] Design mounting solution for camera angle adjustment
  - [ ] Ensure proper ventilation for thermal management
  - [ ] Create 3D model/CAD drawings if custom

**Deliverables Week 4:**
- Working hardware assembly
- Wiring schematic (PDF)
- GPIO pin documentation
- Thermal performance report

---

### Giovanny (Computer Engineer - AI/ML Focus)
- [ ] **Development Environment**
  - [ ] Set up Python 3.11+ virtual environment
  - [ ] Install Hailo SDK and runtime
  - [ ] Configure Picamera2 library
  - [ ] Set up VS Code Remote development

- [ ] **Camera Pipeline**
  - [ ] Implement basic camera capture (Picamera2)
  - [ ] Create frame buffer management system
  - [ ] Implement resolution/FPS configuration
  - [ ] Build basic streaming proof-of-concept

- [ ] **AI Model Research**
  - [ ] Evaluate YOLOv8n vs YOLOv8s for Hailo
  - [ ] Research ONNX to HEF model conversion
  - [ ] Document Hailo model zoo options
  - [ ] Benchmark baseline CPU inference speeds

**Deliverables Week 4:**
- Camera streaming working at 30 FPS
- Model evaluation report
- Development environment documentation

---

### Mohammed (Computer Engineer - Backend Focus)
- [ ] **Project Infrastructure**
  - [ ] Initialize Git repository structure
  - [ ] Create project scaffolding (see directory structure)
  - [ ] Set up requirements.txt and setup.py
  - [ ] Configure logging framework

- [ ] **Database Design**
  - [ ] Design SQLite schema for events
  - [ ] Create models for: events, faces, zones, settings
  - [ ] Implement database migration system
  - [ ] Build basic CRUD operations

- [ ] **Configuration System**
  - [ ] Implement YAML configuration parser
  - [ ] Create argus.yaml template
  - [ ] Build zones.yaml for detection areas
  - [ ] Environment variable management

**Deliverables Week 4:**
- Git repo with proper structure
- Database schema documentation
- Configuration system working
- Basic logging implemented

---

## Phase 2: AI Integration (Weeks 5-8)

### Christian (Electrical Engineer)
- [ ] **Power Optimization**
  - [ ] Measure power consumption at idle/load
  - [ ] Implement power monitoring (if hardware available)
  - [ ] Design UPS/battery backup solution (optional)
  - [ ] Document thermal throttling thresholds

- [ ] **Additional Sensors**
  - [ ] Integrate I2S MEMS microphone (if using)
  - [ ] Set up speaker/siren circuit for deterrent
  - [ ] Test IR illuminator with camera night mode
  - [ ] Create sensor calibration procedures

- [ ] **Hardware Testing**
  - [ ] Stress test under sustained AI load
  - [ ] Verify GPIO interrupt handling
  - [ ] Test camera in various lighting conditions
  - [ ] Document any hardware issues/limitations

**Deliverables Week 8:**
- Power consumption report
- Sensor calibration guide
- Hardware stress test results

---

### Giovanny (Computer Engineer - AI/ML Focus)
- [ ] **Object Detection Implementation**
  - [ ] Convert YOLOv8 model to Hailo format (.hef)
  - [ ] Implement Hailo runtime inference pipeline
  - [ ] Build detection post-processing (NMS, filtering)
  - [ ] Achieve target: 30 FPS inference

- [ ] **Face Recognition**
  - [ ] Integrate face detection model
  - [ ] Implement ArcFace/InsightFace for embeddings
  - [ ] Build known faces database system
  - [ ] Create face enrollment workflow

- [ ] **Object Tracking**
  - [ ] Implement SORT or DeepSORT tracker
  - [ ] Track objects across frames with IDs
  - [ ] Calculate movement vectors and speed
  - [ ] Handle occlusion and re-identification

- [ ] **Zone Detection**
  - [ ] Build polygon zone definition system
  - [ ] Implement zone intrusion detection
  - [ ] Create zone-based filtering logic
  - [ ] Support multiple zone types (entry, perimeter, restricted)

**Deliverables Week 8:**
- YOLOv8 running at 30+ FPS on Hailo
- Face recognition with known/unknown classification
- Object tracking with persistent IDs
- Zone detection system

---

### Mohammed (Computer Engineer - Backend Focus)
- [ ] **Event System**
  - [ ] Design event bus architecture
  - [ ] Implement pub/sub pattern for detections
  - [ ] Create event classification engine
  - [ ] Build event queue for async processing

- [ ] **Recording Service**
  - [ ] Implement H.264 video clip recording
  - [ ] Create pre/post event buffer (5s before, 10s after)
  - [ ] Build thumbnail generation from clips
  - [ ] Implement storage management (rotation, cleanup)

- [ ] **API Foundation**
  - [ ] Set up FastAPI application structure
  - [ ] Create WebSocket handler for live events
  - [ ] Implement basic authentication
  - [ ] Build health check endpoints

**Deliverables Week 8:**
- Event system with pub/sub
- Video recording service
- Basic API endpoints working
- WebSocket live feed

---

## Phase 3: Automation & Alerts (Weeks 9-12)

### Christian (Electrical Engineer)
- [ ] **Smart Home Hardware**
  - [ ] Research relay modules for light control
  - [ ] Design siren/alarm circuit
  - [ ] Test smart lock integration options
  - [ ] Document supported hardware list

- [ ] **Physical Installation Guide**
  - [ ] Create mounting instructions
  - [ ] Design weatherproofing for outdoor use
  - [ ] Cable management recommendations
  - [ ] Create installation checklist

**Deliverables Week 12:**
- Smart home hardware integration guide
- Physical installation documentation

---

### Giovanny (Computer Engineer - AI/ML Focus)
- [ ] **Anomaly Detection**
  - [ ] Implement loitering detection (person stationary too long)
  - [ ] Build unusual activity patterns detection
  - [ ] Create time-based behavior analysis
  - [ ] Reduce false positives with ML filtering

- [ ] **Performance Optimization**
  - [ ] Profile inference pipeline bottlenecks
  - [ ] Optimize frame preprocessing
  - [ ] Implement model batching if beneficial
  - [ ] Achieve <100ms detection latency

- [ ] **Model Fine-tuning**
  - [ ] Collect edge cases from testing
  - [ ] Fine-tune models for home environment
  - [ ] Test in various lighting/weather conditions
  - [ ] Document accuracy metrics

**Deliverables Week 12:**
- Anomaly detection features
- Performance optimization report
- Model accuracy benchmarks

---

### Mohammed (Computer Engineer - Backend Focus)
- [ ] **Alert System**
  - [ ] Implement Telegram bot notifications
  - [ ] Add Discord webhook integration
  - [ ] Create push notification service
  - [ ] Build alert throttling/cooldown logic

- [ ] **Automation Engine**
  - [ ] Design rule engine for triggers
  - [ ] Implement time-based schedules
  - [ ] Create presence modes (Home/Away/Night)
  - [ ] Build action execution system

- [ ] **Home Assistant Integration**
  - [ ] Create MQTT event publishing
  - [ ] Build Home Assistant custom component
  - [ ] Implement entity discovery
  - [ ] Test bidirectional control

- [ ] **API Completion**
  - [ ] Event history endpoints
  - [ ] Settings/configuration API
  - [ ] Zone management API
  - [ ] Face enrollment API

**Deliverables Week 12:**
- Full notification system
- Automation engine with rules
- Home Assistant integration
- Complete REST API

---

## Phase 4: Web App & Cloud (Weeks 13-14)

### Platform: Cloudflare (Primary) / Render (Backup)

```
Frontend: Next.js 14 + TailwindCSS + shadcn/ui
Backend:  Cloudflare Workers (Hono.js) or FastAPI
Database: Cloudflare D1 or PostgreSQL
Tunnel:   Cloudflare Tunnel (Pi to Cloud)
```

---

### Christian (Electrical Engineer)
- [ ] **Cloudflare Tunnel Setup (on Pi)**
  - [ ] Install cloudflared daemon on Raspberry Pi
  - [ ] Configure tunnel for local API access
  - [ ] Set up tunnel as systemd service
  - [ ] Test remote connectivity

- [ ] **Hardware Documentation**
  - [ ] Create final bill of materials (BOM)
  - [ ] Write hardware assembly guide
  - [ ] Document troubleshooting procedures
  - [ ] Create maintenance checklist

- [ ] **Device Provisioning**
  - [ ] Create device registration flow
  - [ ] Generate unique device API keys
  - [ ] Build first-time setup wizard

**Deliverables Week 14:**
- Cloudflare Tunnel working
- Device provisioning script
- Hardware documentation complete

---

### Giovanny (Computer Engineer - AI/ML Focus)
- [ ] **Live Streaming Pipeline**
  - [ ] Implement HLS streaming server (FFmpeg)
  - [ ] Add WebRTC option for low-latency
  - [ ] Create adaptive bitrate streams (360p, 720p, 1080p)
  - [ ] Optimize encoding for mobile viewing

- [ ] **Stream Overlays**
  - [ ] Draw bounding boxes on live stream
  - [ ] Display face recognition labels
  - [ ] Show zone boundary overlays
  - [ ] Add timestamp and confidence scores

- [ ] **Cloud Video Storage**
  - [ ] Implement clip upload to Cloudflare R2/S3
  - [ ] Generate thumbnail from clips
  - [ ] Create signed URLs for secure access
  - [ ] Build retention policy (auto-delete old clips)

**Deliverables Week 14:**
- HLS streaming at 720p+
- Detection overlays on stream
- Cloud video clip storage

---

### Mohammed (Computer Engineer - Backend Focus)
- [ ] **Cloudflare Workers API**
  - [ ] Set up Hono.js project structure
  - [ ] Create D1 database schema
  - [ ] Implement authentication (JWT)
  - [ ] Build all REST endpoints

- [ ] **API Endpoints**
  - [ ] `/api/auth/*` - Authentication
  - [ ] `/api/devices/*` - Device management
  - [ ] `/api/events/*` - Event history
  - [ ] `/api/zones/*` - Zone configuration
  - [ ] `/api/faces/*` - Face enrollment
  - [ ] `/api/rules/*` - Automation rules

- [ ] **WebSocket Handler**
  - [ ] Implement Durable Objects for WebSocket
  - [ ] Real-time event push to dashboard
  - [ ] Device status updates (online/offline)
  - [ ] Live detection notifications

- [ ] **Next.js Frontend**
  - [ ] Set up Next.js 14 + TailwindCSS + shadcn/ui
  - [ ] Create authentication pages (login/register)
  - [ ] Build main dashboard with live feed
  - [ ] Implement event timeline view
  - [ ] Create zone configuration UI (polygon drawing)
  - [ ] Build face enrollment interface
  - [ ] Add settings and profile pages
  - [ ] PWA configuration for mobile install

- [ ] **Deployment**
  - [ ] Deploy frontend to Cloudflare Pages
  - [ ] Deploy API to Cloudflare Workers
  - [ ] Configure custom domain
  - [ ] Set up CI/CD with GitHub Actions

**Deliverables Week 14:**
- Full REST API deployed
- WebSocket real-time events
- Complete web dashboard
- Mobile-responsive PWA

---

## Phase 5: Testing & Documentation (Weeks 15-16)

### Christian (Electrical Engineer)
- [ ] **Hardware Testing**
  - [ ] 48-hour continuous operation test
  - [ ] Temperature cycling test
  - [ ] Power failure recovery test
  - [ ] Create test report

- [ ] **Final Documentation**
  - [ ] Complete hardware schematics
  - [ ] Installation video/photos
  - [ ] Safety considerations document

---

### Giovanny (Computer Engineer - AI/ML Focus)
- [ ] **AI Testing**
  - [ ] Create test dataset (100+ scenarios)
  - [ ] Calculate mAP, precision, recall
  - [ ] Test edge cases (night, rain, crowds)
  - [ ] Performance benchmarking report

- [ ] **Optimization Report**
  - [ ] Document final FPS/latency metrics
  - [ ] Compare Hailo vs CPU performance
  - [ ] Power consumption analysis

---

### Mohammed (Computer Engineer - Backend Focus)
- [ ] **System Testing**
  - [ ] Unit tests for all modules
  - [ ] Integration tests for API
  - [ ] Load testing for concurrent users
  - [ ] Security audit (authentication, XSS, etc.)

- [ ] **Documentation**
  - [ ] API documentation (OpenAPI/Swagger)
  - [ ] User manual
  - [ ] Developer setup guide
  - [ ] Deployment instructions

---

## Weekly Sync Meetings

| Day | Time | Purpose |
|-----|------|---------|
| Monday | 6:00 PM | Sprint planning, task review |
| Thursday | 6:00 PM | Progress check, blockers |

---

## Communication Channels

- **GitHub:** Code, issues, pull requests
- **Discord/Slack:** Daily communication
- **Notion:** Documentation, meeting notes
- **Prometheus:** Agent status, automation

---

## Task Priority Legend

| Priority | Description |
|----------|-------------|
| P0 | Critical - blocks other work |
| P1 | High - required for milestone |
| P2 | Medium - important but flexible |
| P3 | Low - nice to have |

---

## Git Workflow

```
main
  └── develop
        ├── feature/christian-hardware-setup
        ├── feature/giovanny-detection-pipeline
        ├── feature/mohammed-api-backend
        └── feature/[name]-[feature]
```

**Branch naming:** `feature/[name]-[short-description]`
**Commit format:** `[type]: description` (feat, fix, docs, test, refactor)

---

*Last Updated: January 2026*
*Project: ARGUS Senior Design*
