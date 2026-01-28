# ARGUS - AI-Powered Home Security & Automation System
## Senior Design Project Proposal

---

## Executive Summary

**ARGUS** (Autonomous Residential Guardian & Utility System) is an AI-powered home security and automation platform built on Raspberry Pi 5 hardware. The system provides real-time threat detection, intelligent automation, and seamless integration with existing smart home ecosystems.

---

## 1. Project Overview

### 1.1 Problem Statement
Traditional home security systems are:
- Reactive rather than proactive
- Expensive with monthly subscription fees
- Limited in customization and automation
- Prone to false alarms without intelligent filtering

### 1.2 Proposed Solution
ARGUS leverages edge AI computing to provide:
- Real-time person/object detection and recognition
- Behavioral anomaly detection
- Intelligent automation based on presence and context
- Local processing for privacy and low latency
- No cloud dependency or subscription fees

### 1.3 Project Objectives
1. Design and implement a functional AI security system
2. Achieve <500ms detection-to-alert latency
3. Maintain >95% detection accuracy with <5% false positive rate
4. Create intuitive mobile/web dashboard
5. Enable voice control integration

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ARGUS SYSTEM                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   CAPTURE    │    │  PROCESSING  │    │      RESPONSE        │  │
│  │    LAYER     │───▶│    LAYER     │───▶│       LAYER          │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                      │                │
│         ▼                   ▼                      ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ • Camera     │    │ • AI Engine  │    │ • Alerts/Notifs      │  │
│  │ • Sensors    │    │ • Detection  │    │ • Automation         │  │
│  │ • Audio      │    │ • Tracking   │    │ • Recording          │  │
│  │ • Motion     │    │ • Analysis   │    │ • Dashboard          │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    INTEGRATION LAYER                         │   │
│  │  HomeAssistant │ MQTT │ Telegram │ Discord │ Local API      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Layered Architecture Detail

```
┌─────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  Web UI     │ │  Mobile App │ │ Voice Ctrl  │ │  REST API   │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                         SERVICE LAYER                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  Detection  │ │  Automation │ │  Recording  │ │  Alerting   │   │
│  │  Service    │ │  Service    │ │  Service    │ │  Service    │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                        AI/ML LAYER                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  YOLOv8     │ │  Face Rec   │ │  Pose Est   │ │  Anomaly    │   │
│  │  Detection  │ │  (ArcFace)  │ │  (MoveNet)  │ │  Detection  │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                       HARDWARE LAYER                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │ Raspberry   │ │  AI HAT+    │ │   Camera    │ │  Sensors    │   │
│  │ Pi 5 8GB    │ │  (Hailo-8L) │ │  Module 3   │ │  (PIR/Door) │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Hardware Components

### 3.1 Core Components

| Component | Model | Purpose | Est. Cost |
|-----------|-------|---------|-----------|
| **SBC** | Raspberry Pi 5 (8GB) | Main processing unit | $80 |
| **AI Accelerator** | Raspberry Pi AI HAT+ | 26 TOPS neural acceleration | $70 |
| **Camera** | Raspberry Pi Camera Module 3 | 12MP, HDR, autofocus | $35 |
| **Storage** | Samsung EVO 128GB microSD | OS and recordings | $20 |
| **Power** | Official Pi 5 27W PSU | Stable power delivery | $15 |
| **Case** | Argon ONE V3 or Custom | Cooling + protection | $25 |
| **Total** | | | **~$245** |

### 3.2 Optional/Extended Components

| Component | Model | Purpose | Est. Cost |
|-----------|-------|---------|-----------|
| PIR Sensor | HC-SR501 | Motion detection backup | $5 |
| Door Sensor | MC-38 | Entry point monitoring | $8 |
| IR Illuminator | 850nm LED Array | Night vision enhancement | $15 |
| Microphone | USB or I2S MEMS | Audio detection/voice | $10 |
| Speaker | 3W amplified | Alerts/deterrent | $8 |
| NVMe SSD | 256GB M.2 | Extended recording storage | $40 |

### 3.3 Hardware Block Diagram

```
                           ┌─────────────────────┐
                           │   Power Supply      │
                           │   27W USB-C         │
                           └──────────┬──────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RASPBERRY PI 5 (8GB)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   BCM2712   │  │   8GB       │  │   Peripherals           │  │
│  │   Quad A76  │  │   LPDDR4X   │  │   USB 3.0, GPIO, PCIe   │  │
│  │   2.4GHz    │  │             │  │   Ethernet, WiFi, BT    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                            │                                     │
│                     ┌──────┴──────┐                             │
│                     │   PCIe x4   │                             │
│                     └──────┬──────┘                             │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI HAT+ (Hailo-8L)                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              26 TOPS Neural Processing Unit              │    │
│  │   • YOLOv8 inference at 30+ FPS                         │    │
│  │   • Multiple concurrent models                          │    │
│  │   • Low power consumption (~3W typical)                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Camera Module  │ │   PIR Sensor    │ │   Door/Window   │
│  • 12MP Sony    │ │   • HC-SR501    │ │   • MC-38       │
│  • 120° FOV     │ │   • GPIO 17     │ │   • GPIO 27     │
│  • HDR + AF     │ │                 │ │                 │
│  • CSI-2        │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## 4. Software Architecture

### 4.1 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **OS** | Raspberry Pi OS (64-bit) | Optimized Linux base |
| **Runtime** | Python 3.11+ | Core application logic |
| **AI Framework** | Hailo Runtime + ONNX | Model inference |
| **Detection** | YOLOv8n/s (Hailo-optimized) | Object detection |
| **Face Recognition** | ArcFace/InsightFace | Known person identification |
| **Video** | Picamera2 + libcamera | Camera interface |
| **Streaming** | FFmpeg + HLS | Live view streaming |
| **Database** | SQLite + Redis | Events and caching |
| **API** | FastAPI | REST endpoints |
| **Frontend** | React + TailwindCSS | Web dashboard |
| **Messaging** | MQTT (Mosquitto) | Event distribution |
| **Integration** | Home Assistant | Smart home ecosystem |

### 4.2 Software Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           ARGUS CORE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    MAIN CONTROLLER                           │   │
│  │              (argus/core/controller.py)                      │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
│         ┌───────────────────┼───────────────────┐                  │
│         ▼                   ▼                   ▼                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │   CAPTURE   │     │  DETECTION  │     │   ACTION    │          │
│  │   MODULE    │────▶│   ENGINE    │────▶│   ENGINE    │          │
│  └─────────────┘     └─────────────┘     └─────────────┘          │
│         │                   │                   │                  │
│         ▼                   ▼                   ▼                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │ • Camera    │     │ • YOLO      │     │ • Alerts    │          │
│  │ • Sensors   │     │ • FaceRec   │     │ • Recording │          │
│  │ • Audio     │     │ • Tracking  │     │ • Automation│          │
│  │ • Streaming │     │ • Anomaly   │     │ • Logging   │          │
│  └─────────────┘     └─────────────┘     └─────────────┘          │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        DATA LAYER                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │   SQLite    │     │    Redis    │     │ File System │          │
│  │   Events    │     │   Cache     │     │  Recordings │          │
│  └─────────────┘     └─────────────┘     └─────────────┘          │
├─────────────────────────────────────────────────────────────────────┤
│                      INTEGRATION LAYER                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐   │
│  │    MQTT     │ │  REST API   │ │  WebSocket  │ │ HomeAssist │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 AI Pipeline

```
┌───────────────────────────────────────────────────────────────────┐
│                      AI DETECTION PIPELINE                        │
└───────────────────────────────────────────────────────────────────┘

  Frame Input                Processing                    Output
  ──────────                 ──────────                    ──────
       │                          │                           │
       ▼                          ▼                           ▼
┌─────────────┐           ┌─────────────┐           ┌─────────────┐
│   Camera    │           │   Hailo-8L  │           │   Action    │
│   1080p     │──────────▶│   NPU       │──────────▶│   Engine    │
│   30 FPS    │           │             │           │             │
└─────────────┘           └─────────────┘           └─────────────┘
       │                          │                           │
       │                          │                           │
       ▼                          ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   STAGE 1: DETECTION          STAGE 2: RECOGNITION             │
│   ──────────────────          ────────────────────             │
│                                                                 │
│   ┌─────────────────┐         ┌─────────────────┐              │
│   │    YOLOv8n      │         │    ArcFace      │              │
│   │    ────────     │         │    ────────     │              │
│   │  • Person       │   ───▶  │  • Known faces  │              │
│   │  • Vehicle      │         │  • Strangers    │              │
│   │  • Animal       │         │  • Confidence   │              │
│   │  • Package      │         │                 │              │
│   └─────────────────┘         └─────────────────┘              │
│           │                           │                        │
│           │         STAGE 3: ANALYSIS │                        │
│           │         ─────────────────                          │
│           │                           │                        │
│           ▼                           ▼                        │
│   ┌─────────────────────────────────────────────────┐          │
│   │              DECISION ENGINE                     │          │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │          │
│   │  │ Zone Check  │  │ Time Rules  │  │ Anomaly │  │          │
│   │  │ • Entry     │  │ • Schedule  │  │ • Loiter│  │          │
│   │  │ • Perimeter │  │ • Away mode │  │ • Speed │  │          │
│   │  │ • Restricted│  │ • Night     │  │ • Count │  │          │
│   │  └─────────────┘  └─────────────┘  └─────────┘  │          │
│   └─────────────────────────────────────────────────┘          │
│                               │                                │
│                               ▼                                │
│   ┌─────────────────────────────────────────────────┐          │
│   │              OUTPUT ACTIONS                      │          │
│   │  • Push Notification (Telegram/Discord/App)     │          │
│   │  • Start Recording (H.264 clip)                 │          │
│   │  • Trigger Automation (lights, siren, lock)     │          │
│   │  • Log Event (SQLite + timestamp + thumbnail)   │          │
│   │  • Stream Alert (WebSocket to dashboard)        │          │
│   └─────────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

PERFORMANCE TARGETS:
┌─────────────────────────────────────────────────────────────────┐
│  • Frame Rate:        30 FPS input, 15-30 FPS inference        │
│  • Latency:           <100ms detection, <500ms end-to-end      │
│  • Accuracy:          >95% mAP for person detection            │
│  • False Positive:    <5% with zone + time filtering           │
│  • Power:             <15W typical system consumption          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Development Pipeline

### 5.1 Project Timeline (16 Weeks)

```
PHASE 1: FOUNDATION (Weeks 1-4)
═══════════════════════════════════════════════════════════════════
Week 1-2: Hardware Setup & Environment
  ├─ Raspberry Pi 5 initial setup
  ├─ AI HAT+ installation and driver configuration
  ├─ Camera Module 3 integration
  ├─ Development environment (VS Code Remote, Git)
  └─ Baseline performance benchmarks

Week 3-4: Core Infrastructure
  ├─ Project structure and architecture
  ├─ Camera capture pipeline (Picamera2)
  ├─ Basic frame processing loop
  ├─ SQLite database schema
  └─ Configuration management system

PHASE 2: AI INTEGRATION (Weeks 5-8)
═══════════════════════════════════════════════════════════════════
Week 5-6: Object Detection
  ├─ YOLOv8 model optimization for Hailo
  ├─ Model conversion (ONNX → HEF)
  ├─ Hailo runtime integration
  ├─ Detection pipeline implementation
  └─ Performance optimization (target: 30 FPS)

Week 7-8: Advanced Recognition
  ├─ Face detection integration
  ├─ Face embedding extraction (ArcFace)
  ├─ Known faces database
  ├─ Tracking between frames (SORT/DeepSORT)
  └─ Zone definition and monitoring

PHASE 3: AUTOMATION & ALERTS (Weeks 9-12)
═══════════════════════════════════════════════════════════════════
Week 9-10: Alert System
  ├─ Event classification engine
  ├─ Push notification integration (Telegram/Discord)
  ├─ Recording service (H.264 clips)
  ├─ Thumbnail generation
  └─ Event logging and storage

Week 11-12: Automation Engine
  ├─ Rule engine design
  ├─ Time-based schedules
  ├─ Presence detection modes
  ├─ Home Assistant integration
  └─ MQTT event publishing

PHASE 4: USER INTERFACE (Weeks 13-14)
═══════════════════════════════════════════════════════════════════
Week 13: Backend API
  ├─ FastAPI REST endpoints
  ├─ WebSocket live streaming
  ├─ Authentication/authorization
  ├─ Settings and configuration API
  └─ Event history API

Week 14: Frontend Dashboard
  ├─ React dashboard scaffolding
  ├─ Live video feed component
  ├─ Event timeline view
  ├─ Zone configuration UI
  └─ Mobile-responsive design

PHASE 5: TESTING & DOCUMENTATION (Weeks 15-16)
═══════════════════════════════════════════════════════════════════
Week 15: Testing & Optimization
  ├─ Unit and integration tests
  ├─ Performance benchmarking
  ├─ Edge case handling
  ├─ Power consumption optimization
  └─ Memory leak detection

Week 16: Documentation & Presentation
  ├─ Technical documentation
  ├─ User manual
  ├─ API documentation
  ├─ Presentation preparation
  └─ Demo video production
```

### 5.2 Gantt Chart

```
Week:        1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
             ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤

PHASE 1      ████████████████
Hardware     ████████
Core Infra           ████████

PHASE 2                      ████████████████
Detection                    ████████
Recognition                          ████████

PHASE 3                                      ████████████████
Alerts                                       ████████
Automation                                           ████████

PHASE 4                                                      ████████
Backend                                                      ████
Frontend                                                         ████

PHASE 5                                                              ████████
Testing                                                              ████
Docs                                                                     ████

MILESTONES:
  M1 (Week 4):  Hardware functional, camera streaming ◆
  M2 (Week 8):  AI detection working at 30 FPS       ◆
  M3 (Week 12): Full automation and alerts           ◆
  M4 (Week 14): Complete dashboard                   ◆
  M5 (Week 16): Final presentation ready             ◆
```

---

## 6. Directory Structure

```
argus/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── argus.yaml              # Main configuration
│   ├── zones.yaml              # Detection zones
│   ├── faces/                  # Known faces database
│   └── models/                 # AI model files (.hef)
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── core/
│   │   ├── controller.py       # Main controller
│   │   ├── config.py           # Configuration management
│   │   └── events.py           # Event bus
│   ├── capture/
│   │   ├── camera.py           # Camera interface
│   │   ├── sensors.py          # GPIO sensors
│   │   └── stream.py           # HLS streaming
│   ├── detection/
│   │   ├── detector.py         # YOLO detection
│   │   ├── faces.py            # Face recognition
│   │   ├── tracker.py          # Object tracking
│   │   └── zones.py            # Zone management
│   ├── actions/
│   │   ├── alerts.py           # Notification service
│   │   ├── recording.py        # Video recording
│   │   ├── automation.py       # Home automation
│   │   └── logging.py          # Event logging
│   ├── api/
│   │   ├── app.py              # FastAPI application
│   │   ├── routes/             # API endpoints
│   │   └── websocket.py        # WebSocket handler
│   └── integrations/
│       ├── mqtt.py             # MQTT client
│       ├── homeassistant.py    # HA integration
│       ├── telegram.py         # Telegram bot
│       └── discord.py          # Discord webhook
├── web/                        # React frontend
│   ├── src/
│   ├── public/
│   └── package.json
├── tests/
│   ├── test_detection.py
│   ├── test_camera.py
│   └── test_api.py
├── scripts/
│   ├── install.sh              # Installation script
│   ├── convert_model.py        # ONNX to HEF conversion
│   └── benchmark.py            # Performance testing
└── docs/
    ├── architecture.md
    ├── api.md
    └── user_guide.md
```

---

## 7. Key Features

### 7.1 Core Security Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Person Detection** | Detect humans in frame with bounding boxes | P0 |
| **Face Recognition** | Identify known vs unknown persons | P0 |
| **Motion Zones** | Define areas of interest for monitoring | P0 |
| **Intrusion Alerts** | Push notifications on unauthorized entry | P0 |
| **Video Recording** | Capture clips on detection events | P0 |
| **Live Streaming** | Real-time video feed via HLS/WebRTC | P1 |
| **Night Vision** | IR-enhanced low-light detection | P1 |
| **Package Detection** | Identify deliveries at door | P2 |
| **Vehicle Detection** | Monitor driveway/parking | P2 |

### 7.2 Automation Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Presence Modes** | Home/Away/Night automation profiles | P1 |
| **Smart Lighting** | Trigger lights on detection | P1 |
| **Schedule Rules** | Time-based automation triggers | P1 |
| **Geofencing** | Auto arm/disarm based on phone location | P2 |
| **Voice Control** | Integration with voice assistants | P2 |

---

## 8. Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Hailo SDK compatibility issues | High | Medium | Early testing, fallback to CPU inference |
| Insufficient processing power | High | Low | Model optimization, reduce resolution |
| Camera latency | Medium | Medium | Direct CSI, optimize pipeline |
| Power consumption too high | Medium | Low | Power profiling, sleep modes |
| False positive alerts | Medium | Medium | ML filtering, zone refinement |
| Network reliability | Low | Medium | Local-first design, offline capability |

---

## 9. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Detection Latency | <100ms | Timestamp analysis |
| End-to-End Alert Time | <500ms | Event correlation |
| Detection Accuracy (mAP) | >95% | Test dataset evaluation |
| False Positive Rate | <5% | Production monitoring |
| System Uptime | >99.5% | Health monitoring |
| Power Consumption | <15W idle, <20W active | Power meter |
| Frame Rate | 30 FPS | Performance profiler |

---

## 10. Budget Summary

| Category | Items | Cost |
|----------|-------|------|
| **Hardware (Core)** | Pi 5, AI HAT+, Camera, Storage, PSU, Case | $245 |
| **Hardware (Extended)** | Sensors, IR, Mic, Speaker | $46 |
| **Software** | Open source (free) | $0 |
| **Contingency** | 15% buffer | $44 |
| **Total** | | **~$335** |

---

## 11. Team Responsibilities

| Role | Responsibilities |
|------|-----------------|
| **Hardware Lead** | Pi setup, HAT integration, sensor wiring, power management |
| **AI/ML Lead** | Model selection, optimization, Hailo integration, accuracy tuning |
| **Backend Lead** | API development, database design, event system, integrations |
| **Frontend Lead** | Dashboard UI, mobile responsiveness, live streaming |
| **Integration Lead** | Home Assistant, MQTT, notification services |

---

## 12. References

- [Raspberry Pi 5 Documentation](https://www.raspberrypi.com/documentation/)
- [Raspberry Pi AI HAT+ Documentation](https://www.raspberrypi.com/documentation/accessories/ai-hat.html)
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- [YOLOv8 by Ultralytics](https://docs.ultralytics.com/)
- [Picamera2 Documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [Home Assistant Integration](https://www.home-assistant.io/integrations/)

---

## Appendix A: Prometheus Integration

ARGUS integrates with the Prometheus AI Orchestrator for:
- Voice command control ("Hey Prometheus, arm Argus")
- Agent status monitoring
- Cross-project coordination
- Event logging to Notion

```python
# Example Prometheus command
prometheus> spawn argus agent
prometheus> argus status
prometheus> set argus mode away
```

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Project: ARGUS Senior Design*
