# THE EYE

**Offline-First Smart Home Security & Automation System**

A self-hosted, privacy-focused smart home security system built on Raspberry Pi 5. Designed to replace Ring, Alexa, and other cloud-dependent services with fully local AI-powered surveillance and home automation.

## Overview

THE EYE is a modular security and automation platform that:
- Detects motion, recognizes faces, and classifies objects using on-device AI
- Controls home automation (lights, locks, thermostats, cameras)
- Operates 100% offline with no cloud dependencies
- Provides a web interface for monitoring and control

## Hardware

| Component | SKU | Description |
|-----------|-----|-------------|
| Raspberry Pi 5 | - | Main compute unit |
| Camera Module 3 | 46-1 | Primary vision input |
| HighPi Pro 5S Case | 9015 | Enclosure with ventilation |
| USB-C PD Power Supply | 492-2 | 27W power delivery |
| Active Cooler | 374-1 | Thermal management |
| MicroSD Card | 1350-1 | 32GB with Raspberry Pi OS |
| AI Hat+ 2 | - | Neural network accelerator (optional) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        THE EYE                               │
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
│  │           Streamlit / Web UI                    │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
THE EYE/
├── src/
│   ├── api/            # FastAPI routes and endpoints
│   ├── agents/         # AI agent orchestration
│   ├── camera/         # Camera capture and streaming
│   ├── detection/      # Motion, face, object detection
│   ├── automation/     # Home automation controls
│   ├── security/       # Encryption, auth, logging
│   └── database/       # Local data storage
├── docs/               # Documentation
├── tests/              # Unit and integration tests
├── config/             # Configuration files
├── static/             # Web assets
├── requirements.txt    # Python dependencies
└── main.py            # Application entry point
```

## Quick Start

### 1. Hardware Setup
See [docs/HARDWARE.md](docs/HARDWARE.md) for assembly instructions.

### 2. Software Installation
```bash
# Clone and enter project
cd THE_EYE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the system
python main.py
```

### 3. Access the Dashboard
Open `http://<pi-ip>:8000` in your browser.

## Agent System

THE EYE uses a multi-agent architecture:

| Agent | Role |
|-------|------|
| **Sentinel Prime** | Lead orchestrator, delegates tasks |
| **Engineer-Bot** | GPIO, sensors, power management |
| **Code-Mage** | Python, FastAPI, AI model integration |
| **Sentinel-Guard** | Security, encryption, firewalling |
| **DocuBot** | Documentation, API references |

## Detection Pipeline

```
Camera Stream → Motion Detection → Object Classification
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
               Known Face            Unknown Human           Animal
                    │                     │                     │
                Log Entry            Alert + Chat           Log + Save
```

## Security Features

- All data stored locally (no cloud)
- Encrypted biometric storage
- Firewall-hardened endpoints
- No external API dependencies
- Audit logging for all events

## License

Private / Internal Use

## Contributing

This is a personal project. Contact the owner for collaboration.
