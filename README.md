# ARGUS
## Autonomous Residential Guardian & Utility System

AI-Powered Home Security & Automation built on Raspberry Pi 5

---

## Overview

ARGUS is a senior design project that creates an intelligent home security system using edge AI computing. The system provides real-time threat detection, face recognition, and smart home automation without cloud dependencies.

## Hardware

| Component | Model | Description |
|-----------|-------|-------------|
| SBC | Raspberry Pi 5 (8GB) | Main compute unit |
| AI Accelerator | Raspberry Pi AI HAT+ (Hailo-8L, 26 TOPS) | Neural network accelerator |
| Camera | Raspberry Pi Camera Module 3 (12MP) | Primary vision input |
| Enclosure | HighPi Pro 5S Case | Enclosure with ventilation |
| Power | USB-C PD Power Supply | 27W power delivery |
| Cooling | Active Cooler | Thermal management |

## Features

- Real-time person/object detection (YOLOv8)
- Face recognition (known vs unknown)
- Motion zones with alerts
- Video recording on events
- Live streaming (HLS/WebRTC)
- Home Assistant integration
- Push notifications (Telegram/Discord)
- 100% offline operation - no cloud dependencies

## Team

| Name | Role | Focus |
|------|------|-------|
| Christian | Electrical Engineer | Hardware, sensors, power |
| Giovanny | Computer Engineer | AI/ML, detection, optimization |
| Mohammed | Computer Engineer | Backend, API, integrations |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         ARGUS                                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Camera  │  │   AI    │  │  Home   │  │ Security│        │
│  │ Module  │──│Detection│──│Automaton│──│  Layer  │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│       │            │            │            │              │
│  ┌────┴────────────┴────────────┴────────────┴────┐        │
│  │              FastAPI Backend                    │        │
│  └────────────────────┬───────────────────────────┘        │
│                       │                                     │
│  ┌────────────────────┴───────────────────────────┐        │
│  │              Web Dashboard (Next.js)            │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Detection Pipeline

```
Camera Stream → Motion Detection → Object Classification
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
               Known Face            Unknown Human           Object
                    │                     │                     │
                Log Entry            Alert + Record         Log + Save
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/Gvictome/argus.git
cd argus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run ARGUS
python -m src.main
```

## Project Structure

```
argus/
├── config/          # Configuration files
├── src/             # Source code
│   ├── core/        # Main controller
│   ├── capture/     # Camera & sensors
│   ├── detection/   # AI models
│   ├── actions/     # Alerts & automation
│   └── api/         # REST API
├── web/             # Next.js dashboard
├── tests/           # Test suite
└── docs/            # Documentation
```

## Documentation

- [Project Proposal](docs/PROPOSAL.md)
- [Team Tasks](TEAM_TASKS.md)
- [Web App Architecture](docs/WEB_APP_ARCHITECTURE.md)
- [API Documentation](docs/api.md)

## Timeline

- **Weeks 1-4:** Hardware setup & core infrastructure
- **Weeks 5-8:** AI integration (YOLO + Face Recognition)
- **Weeks 9-12:** Automation & alert system
- **Weeks 13-14:** Web app & cloud deployment
- **Weeks 15-16:** Testing & documentation

## Security Features

- All data stored locally (no cloud required)
- Encrypted biometric storage
- Secure API endpoints
- Audit logging for all events

## License

MIT License - Senior Design Project 2026

---

*Powered by Prometheus AI Orchestrator*
