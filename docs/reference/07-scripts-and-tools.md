# 7 · Scripts, tools and legacy code

---

## 7.1 Runners (`scripts/`)

| Script | Machine | What it does |
|---|---|---|
| `run_node.sh` | PI | Start the node with demo defaults. Flags: `--every-frame`, `--threat`, `--model`, `--video`, `--no-record`, `--port` |
| `node_status.sh [--once]` | PI | One status line per second from `/api/detection/status` and `/api/status` |
| `fl_trial.sh [--rounds N] [--epochs N] [--samples N] [--no-bootstrap]` | PI | Local aggregator + seed + local train + one round + gate decision |
| `run_central_server.ps1 [-Rounds] [-MinClients] [-FlPort] [-ApiPort] [-Token]` | LAPTOP | The aggregator, kept up between sessions; prints the lines to paste on the Pi |
| `run_dashboard.ps1 -Node <ip>` | LAPTOP | Clone, install and start the Next.js dashboard pointed at a node |
| `run_tunnel.sh [--check]` | PI | Cloudflare tunnel, refused unless auth is on and proven |
| `mock_demo.py [--reset] [--source] [--threat] [--record]` | ANY | The full node against a video, data isolated in `data/mock` |

## 7.2 Data and labelling

| Script | What it does |
|---|---|
| `ingest_video.py <video> [--label L] [--every-frame] [--max-frames N] [--zone Z] [--data-dir D]` | Replays a recording through the real pipeline and labels every event. Uses the video's own clock, so speed and dwell are right even when it replays faster than real time. Stop the node first |
| `check_classes.py --camera / <video> / <image> [--save out.jpg]` | Shows ARGUS class and raw COCO label side by side |
| `set_password.py [--user U] [--create --role R]` | Reset or create an account |

## 7.3 Models: fetch, export, accelerate

| Script | Runs on | What it does |
|---|---|---|
| `fetch_models.py [--variant] [--list]` | ANY | Download threat weights, verify SHA-256 |
| `export_cpu_models.py --format ncnn / openvino` | the target | CPU-optimised YOLO export into `models/` |
| `export_hailo.py --hw-arch hailo8 / hailo10h [--check]` | **x86_64 Linux only** | Hailo export directory. Refuses on the wrong platform rather than failing halfway through a long build |
| `export_tensorrt.py [--model object] [--no-half] [--check]` | **the Jetson itself** | TensorRT engines; not portable across devices or JetPack versions |

## 7.4 Measurement

| Script | What it measures |
|---|---|
| `benchmark_classes.py --images <dir>` or `--dataset <split> [--limit N]` | Per-class confidence: mean, median, 5th percentile, spread, share within 0.10 of the threshold. With a dataset, threat precision and recall |
| `sweep_threat_config.py --dataset <split> [--negatives <dir>] [--weights ...]` | Threat input size × threshold, across variants |
| `tune_threat_threshold.py --dataset <split> [--negatives <dir>]` | Threat threshold only, both error counts |
| `benchmark_face_accuracy.py --faces <dir>` | Face verification accuracy and TAR@FAR (deprecated stage) |
| `verify_enrollment.py --name N [--keep]` | Live face enrollment on the Pi (deprecated stage) |

## 7.5 Documentation

| Script | What it does |
|---|---|
| `build_docs_pdf.py --master [--out-dir D]` | One PDF: this reference, the eight guides, the quick start, the simulator notes |
| `build_docs_pdf.py --guides` | The eight guides, one PDF each |
| `build_docs_pdf.py [--docs NAME ...]` | Selected docs from `docs/`, one PDF each |

Uses headless Chrome rather than a print library (weasyprint needs GTK on
Windows; wkhtmltopdf is unmaintained). No web fonts, so an offline machine
renders the same tables.

---

## 7.6 The simulator (`sim/`)

Described in [section 5.12](05-federated-learning.md). Entry points:
`sim/fl.py serve|client` and `sim/run_node.py`. Its own notes are in
`sim/README.md`, included at the end of the master PDF.

---

## 7.7 Legacy code: present, tested in places, not on the running path

These files are from the earlier design, where federated learning meant
fine-tuning YOLO's own weights on each Pi. That design was replaced by the
19-feature head: YOLOv8n is 3.2M parameters against the head's ~200K
(constraint CON-3), and the node runs YOLO from a compiled export, so
averaged PyTorch weights could not reach the Hailo without recompiling the
`.hef` on an x86 machine after every round.

| File | Was | Now |
|---|---|---|
| `run_client.py` | Started a YOLO-weights Flower client | Not used. The node uses `src/federated/node_client.py` |
| `src/federated/client.py` | `ArgusFlowerClient` exchanging YOLO weights | Not used by the node; `FLScheduler` falls back to it only if no runner is given |
| `src/federated/server.py` | Minimal Flower server | Not used |
| `src/federated/utils.py` | Compress / delta helpers for weight lists | Not used by the running path |
| `src/training/__init__.py`, `dataset.py` | `LocalTrainer`, `LocalDetectionDataset`: YOLO crops on disk, export to YOLO format, `model.train()` | Not used |
| `src/detection/models.py` | `ModelManager`: YOLO weight get / set / delta | Used only by the files above |
| `server/app.py`, `server/config.py`, `server/strategy.py` | FedAvg / FedProx server driven by `STRATEGY`, `NUM_ROUNDS`, `MIN_CLIENTS`, `FL_PORT` | Not used by the node; `central_server/` replaced it |
| `src/detection/stream.py` | Detect-inside-the-stream MJPEG generator | Replaced by `stream_from_worker`; kept for tests |
| `src/detection/face_recognition.py` | ArcFace identity | Deprecated 2026-09-14, off by default |
| `src/detection/benchmark.py` | Bare YOLO FPS loop | Superseded by the scripts and the simulator |
| `src/automation/__init__.py` | Devices and rules (lights, locks, GPIO) | Stub behind `/api/devices`, `/api/automations` |

Do not delete them casually: some are still imported by tests, and the
proposal describes parts of them. Do not extend them either.

---

## 7.8 Research prototypes (`research/`)

Standalone experiments, not imported by the node.

| File | What it is |
|---|---|
| `wifi_csi.py` | Presence and motion from WiFi channel state information (ESP32 or Nexmon backends, plus a mock), with an experimental pose estimator |
| `sensor_fusion.py` | Merges WiFi CSI presence with camera YOLO detections (Bayesian confidence merging; WiFi can wake the camera, the camera can veto WiFi false positives) |
| `argus_detect.py` | Early Hailo detection callback with PIR sensor and software motion |
| `federated.py` | First federated prototype with Prometheus metrics and node registration; the ideas moved into `central_server/` |
| `fl_config.json`, `csi_config.json` | Their configs |

---

## 7.9 Developer tooling inside the app

| Package | What it is |
|---|---|
| `src/agents/` | `ErrorRouter` classifies an error report and routes it to `SyntaxAgent`, `DependencyAgent`, `HardwareAgent` or `NetworkAgent`, which suggest or apply a fix; `FederatedAgent` is a separate early helper. Exposed at `/api/errors/*` and `/api/agents/status`. A development aid, not part of detection |
| `prometheus/` | **Not** the metrics system. A separate voice-activated orchestrator ("Hey Prometheus", "switch to argus") used during development; `python -m prometheus.main`. Documented in `docs/PROMETHEUS.md` |

The Prometheus *metrics* system (the time-series database scraping
`/metrics`) is a different thing entirely and lives in
`central_server/prometheus.yml`.
