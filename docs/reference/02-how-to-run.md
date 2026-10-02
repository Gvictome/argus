# 2 · How to run everything

Every block is labelled with the machine it runs on. **PI** is a shell on
the Raspberry Pi. **LAPTOP** is Windows (Git Bash or PowerShell, as
marked). **ANY** runs on either.

Unless a step says otherwise, you are in the repository root on branch
`feat/threat-detection`.

---

## 2.1 One-time setup

### LAPTOP (Git Bash)

```bash
git clone https://github.com/Gvictome/argus.git
cd argus
git checkout feat/threat-detection
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

`requirements.txt` includes `ultralytics`, `flwr` and `pytest-asyncio`.
Leaving any of them out shows up later in confusing ways: without
`ultralytics` the detector silently finds nothing, and without
`pytest-asyncio` four scheduler tests fail.

Fetch the threat-model weights (not committed to git; checksum-verified):

```bash
python scripts/fetch_models.py
```

Get the sample video the mock demo and one test use:

```bash
mkdir -p media
curl -fsSL -o media/vtest.avi https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi
```

### PI

Full walk-through with explanations: guide 2, *Setting up the Pi*.

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev python3-picamera2 python3-opencv build-essential cmake
git clone https://github.com/Gvictome/argus.git ~/argus
cd ~/argus && git checkout feat/threat-detection
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "from picamera2 import Picamera2; print('camera library ok')"
python scripts/export_cpu_models.py --format ncnn
python scripts/fetch_models.py
python scripts/set_password.py --user admin
```

`--system-site-packages` is required: `picamera2` comes from `apt`, and a
plain venv cannot see it. If the camera import complains about numpy,
`pip install --ignore-installed "numpy>=2.2,<3"`.

---

## 2.2 The node

### PI

```bash
./scripts/run_node.sh
```

| Flag | Effect |
|---|---|
| `--every-frame` | Run YOLO on every frame, not only on motion (classifies subjects that hold still) |
| `--threat` | Turn on the dangerous-person stage (about 100 ms per frame on the Pi CPU) |
| `--model yolov8s` | A bigger, slower detector; needs its own export to be accelerated |
| `--video media/vtest.avi` | Use a recording instead of the camera |
| `--no-record` | Do not write event clips |
| `--port 8001` | Serve on a different port |

`run_node.sh` sets demo-friendly defaults: `AUTH_REQUIRED=false`,
`FL_ENABLED=false`, `THREAT_ENABLED=false`, `RECORD_EVENTS=true`,
`MOTION_GATE=true`. Note that `THREAT_ENABLED` defaults to **true** in
`src/config.py`; `run_node.sh` is what turns it off. Running `python main.py`
directly gives you the config defaults instead.

Watch it from a second shell:

```bash
./scripts/node_status.sh          # one line per second
./scripts/node_status.sh --once
```

Open `http://<pi-address>:8000/dashboard` in a browser. Find the address
with `hostname -I`.

### ANY: the node against a video, with isolated data

```bash
python scripts/mock_demo.py --reset            # media/vtest.avi, data in data/mock
python scripts/mock_demo.py --source 0         # a webcam
python scripts/mock_demo.py --threat --record  # threat stage and clips on
```

Everything is real except the camera (a file) and the data directory
(`data/mock`). The admin password is `argus-demo` unless `--password` is
given. Use this to work on the pipeline without a Pi.

---

## 2.3 Checking what the detector sees

### ANY

```bash
python scripts/check_classes.py --camera --save /tmp/seen.jpg   # through the running node's API
python scripts/check_classes.py media/vtest.avi                  # a video
python scripts/check_classes.py photo.jpg                        # an image
```

Prints the ARGUS class and the raw COCO label behind it. This separates
"the model missed it" from "the model saw it and it maps somewhere else".
`--camera` goes through the node's API because the detection worker owns
the camera and a second reader gets no frames.

---

## 2.4 Labelling and training on the node

Events land unlabelled. Labels come from a person, through the dashboard
(confirm or correct an event) or through the API:

```bash
curl -s localhost:8000/api/federated/samples | python -m json.tool | head -40
curl -sX POST localhost:8000/api/federated/samples/<sample_id>/label \
     -H 'Content-Type: application/json' -d '{"label": 0}'
```

`label` is an integer: 0 `routine_person`, 1 `routine_vehicle`,
2 `routine_animal`, 3 `routine_package`, 4 `anomaly`. The dashboard route
`POST /api/events/{n}/label` takes `{"action": "confirm"}` or
`{"action": "correct", "label": "<free text>"}` instead, and reads intent
from the words (see [section 6](06-api-data-security.md)).

Turn a saved clip into labelled events in one go (**stop the node first**,
both write the same store):

```bash
python scripts/ingest_video.py media/events/some_clip.mp4 --label routine_person
python scripts/ingest_video.py doorbell.mp4 --label anomaly --every-frame
```

Train the head locally and gate it in (needs 200 labelled samples by
default):

```bash
curl -sX POST localhost:8000/api/federated/head/train \
     -H 'Content-Type: application/json' -d '{"epochs": 6}' | python -m json.tool
```

Body fields: `epochs` (default 8), `min_samples` (200), `tolerance` (0.01).

Not enough labels yet? Seed a synthetic corpus. These rows are tagged
`synthetic` and never appear in the event list; say so when presenting.

```bash
curl -sX POST localhost:8000/api/federated/samples/bootstrap \
     -H 'Content-Type: application/json' -d '{"site": "driveway", "n": 4000}'
```

---

## 2.5 A federated round on one node

### PI (node already running in another shell)

```bash
./scripts/fl_trial.sh                  # 3 rounds, seeded corpus
./scripts/fl_trial.sh --rounds 5
./scripts/fl_trial.sh --no-bootstrap   # only real labelled events
```

It starts a local aggregator (`sim/fl.py serve`), seeds samples, trains,
joins the round, and prints the gate's decision. `model_version` moves to
`fl-r1`, `fl-r2`, ... only when the gate accepts the averaged model.

With one node, FedAvg averages that node with itself. This proves the
plumbing, not federation. Federation needs two or more clients: see 2.7.

---

## 2.6 The central aggregator and scheduled rounds

### LAPTOP (PowerShell)

```powershell
.\scripts\run_central_server.ps1
.\scripts\run_central_server.ps1 -Rounds 3 -MinClients 2
```

It prints the two lines to paste on the Pi, keeps every global model in
`central_server/checkpoints`, and stays up between sessions
(`ARGUS_SERVE_FOREVER=true`). If the Pi cannot connect, Windows Firewall is
the usual cause; the script prints the one-line rule to allow it.

| Watch | URL |
|---|---|
| Server state | `http://localhost:8090/api/status` |
| Rounds, loss, accuracy | `http://localhost:8090/api/training/summary` |
| Registered nodes | `http://localhost:8090/api/nodes` |
| Prometheus metrics | `http://localhost:8090/metrics` |

The full stack with Prometheus and Grafana runs under Docker:

```bash
cd central_server && docker compose up -d     # Grafana on :3000 (admin/argus)
```

### PI: join on a schedule

```bash
export FL_ENABLED=true
export FL_SERVER_URL=<laptop-ip>:8080
export FL_CENTRAL_API=http://<laptop-ip>:8090
./scripts/run_node.sh
curl -s localhost:8000/api/federated/status | python -m json.tool
```

Rounds then run every 14 days at 02:00, and catch up promptly if the Pi
was off across the due date. The scheduler has an "only while idle" check,
but nothing feeds it motion yet (`notify_motion()` has no caller), so today
it always reads idle; see [section 10](10-status-and-gaps.md). Training
runs in a separate one-core process either way, so detection keeps going.
Trigger a round by hand at any time:

```bash
curl -sX POST localhost:8000/api/federated/round -H 'Content-Type: application/json' \
     -d '{"server": "<laptop-ip>:8080", "epochs": 4}'
curl -s localhost:8000/api/federated/round | python -m json.tool
```

---

## 2.7 A live multi-client simulation (server + Flower clients)

Real Flower gRPC server and clients as separate processes on one machine.
Each client is a synthetic site with its own non-IID data (`sim/workload.py`).

### ANY: four terminals

```bash
python sim/fl.py serve --rounds 5 --clients 3          # terminal 1, waits for 3 clients
python sim/fl.py client --site driveway                # terminal 2
python sim/fl.py client --site backdoor                # terminal 3
python sim/fl.py client --site yard                    # terminal 4
```

Sites: `driveway`, `backdoor`, `yard`. Client flags: `--samples` (default
6000), `--epochs` (4), `--seed`, `--address`. The server prints per-round
distributed loss, accuracy and balanced accuracy at the end.

Verified on 2026-10-02 (3 clients, 3 rounds, 3000 samples, 2 epochs):

| Round | Loss | Accuracy | Balanced accuracy |
|---|---|---|---|
| 1 | 0.327 | 0.918 | 0.820 |
| 2 | 0.286 | 0.919 | 0.829 |
| 3 | 0.279 | 0.923 | 0.838 |

These are synthetic sites whose generator and model were both written by
the team. They show the federation works end to end; they are not evidence
about real-world accuracy.

### ANY: one node under load while a round runs

```bash
python sim/run_node.py                                # one node at 12 FPS, one round
python sim/run_node.py --clients 3 --rounds 6         # real FedAvg across 3 clients
python sim/run_node.py --profile pi5-threat-on-cpu    # the risky configuration
python sim/run_node.py --pipelined --cores 4          # overlap the accelerator, Pi core count
```

It runs the detection loop at a target frame rate, fires a real Flower
round against it, and reports what training did to the frame rate.
Metrics are on `http://127.0.0.1:9108/metrics` (`--port 0` disables).
Profiles and their sources: [section 5](05-federated-learning.md) and
`sim/README.md`.

---

## 2.8 Making detection faster

| Where | Command | Result |
|---|---|---|
| PI | `python scripts/export_cpu_models.py --format ncnn` | NCNN export in `models/`, picked up automatically |
| x86 | `python scripts/export_cpu_models.py --format openvino` | OpenVINO export, the fast x86 CPU path |
| x86_64 Linux / WSL2 only | `python scripts/export_hailo.py --hw-arch hailo10h` | Hailo export directory for the AI HAT+ 2; copy the **whole directory** to the Pi's `models/` |
| Jetson Orin only | `python scripts/export_tensorrt.py` | TensorRT engine, built on the device itself |

The backend is chosen at startup in this order: TensorRT engine, Hailo
export, CPU export (NCNN on ARM, OpenVINO on x86), plain PyTorch. Check
which one loaded with `curl -s localhost:8000/api/detection/status`.

An export has a fixed input size. The node forces `OBJECT_IMGSZ` to the
export's own size, because asking a 640 export for 416 returns no boxes at
all, quickly, which looks like a huge speedup.

---

## 2.9 Exposing the node beyond the LAN

```bash
AUTH_REQUIRED=true ARGUS_ADMIN_PASSWORD='<a real password>' SECRET_KEY='<random>' python main.py
./scripts/run_tunnel.sh --check     # preflight only
./scripts/run_tunnel.sh             # preflight, then cloudflared
```

The tunnel script refuses to start unless the running node reports
`AUTH_REQUIRED` on **and** an unauthenticated request to a protected
endpoint returns 401. Add a Cloudflare Access policy as the first gate.
Details: `docs/CLOUDFLARE_TUNNEL.md`.

---

## 2.10 The product dashboard

### LAPTOP (PowerShell)

```powershell
.\scripts\run_dashboard.ps1 -Node 10.0.0.140
```

Clones `adam9798/argus-dashboard` if missing, installs it, points it at the
node, and starts it on `http://localhost:3000` (or 3001). If it says
SYSTEM OFFLINE, the browser cannot reach the Pi, or the port Next.js chose
is missing from `CORS_ORIGINS`. The node's own page,
`http://<pi>:8000/dashboard`, has no such dependency.

---

## 2.11 Tests

### ANY

```bash
python -m pytest -q
python -m pytest tests/test_threat.py -q          # one file
python -m pytest -q -k "gate"                     # by name
```

303 tests, about 100 seconds on the development laptop. Gate any push on
the exit code, not on reading the output. Details: [section 9](09-tests.md).

---

## 2.12 Building the documentation

### ANY (needs Google Chrome installed)

```bash
python scripts/build_docs_pdf.py --master                     # this reference + guides, one PDF
python scripts/build_docs_pdf.py --guides                     # the 8 guides, one PDF each
python scripts/build_docs_pdf.py                              # quick start, overview, runbook...
python scripts/build_docs_pdf.py --master --out-dir "C:/Users/you/Dropbox/argus"
```

Markdown in `docs/` is the source of truth. PDFs are rebuilt from it.
