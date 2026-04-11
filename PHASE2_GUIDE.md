# ARGUS - Phase 2: Federated Learning & AI Optimization

This document outlines the steps to test and run Phase 2 of the ARGUS project, focusing on Federated Learning with Flower and AI optimization for the Raspberry Pi 5.

## 1. Prerequisites

### Central PC (FL Server)
- Python 3.10+
- `flwr`, `torch`, `torchvision`, `ultralytics` installed.
- Network access (know your local IPv4 address).

### Raspberry Pi 5 (FL Client / Node)
- Raspberry Pi AI HAT+ installed and configured (`hailo-all` package).
- Camera Module 3 connected.
- `flwr`, `torch`, `torchvision`, `ultralytics` installed in a virtual environment.

## 2. Setup Instructions

### On the Central PC
1. **Clone and Install**:
   ```bash
   git clone https://github.com/Gvictome/argus.git
   cd argus
   python -m venv venv
   # Activate venv and install:
   pip install -r requirements.txt
   ```
2. **Start the FL Server**:
   ```bash
   python -m src.federated.server
   ```

### On the Raspberry Pi 5
1. **Clone and Install**:
   ```bash
   git clone https://github.com/Gvictome/argus.git
   cd argus
   python -m venv venv --system-site-packages
   # Activate venv and install:
   pip install -r requirements.txt
   ```
2. **Optimize for 30 FPS (AI HAT+)**:
   ```bash
   # Export YOLO model to Hailo format
   yolo export model=yolov8n.pt format=hailo
   # Ensure it's named for ARGUS to find:
   mv yolov8n_hailo_model.hef yolov8n_hailo_model.hef
   ```
3. **Configure Connection**:
   Edit `src/config.py` and set `FL_SERVER_URL` to your PC's IP (e.g., `192.168.1.15:8080`).

## 3. Running Phase 2 with Prometheus

Launch the Prometheus orchestrator on the Raspberry Pi:
```bash
python -m prometheus.main
```

### Commands:
- **"switch to argus"**: Sets the active project.
- **"spawn detection agent"**: Starts 30 FPS tracking using the AI HAT+.
- **"spawn federated agent"**: Connects to the central PC to begin federated training rounds.

## 4. Benchmarking
To verify your 30 FPS target on any device, run:
```bash
python -m src.detection.benchmark yolov8n.pt
```
*(Note: On the Pi, use the .hef model for best results)*
