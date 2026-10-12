# 1 · System overview

ARGUS is a camera node that detects what is in front of it, decides
whether that is routine for the place it is watching, records the moments
that matter, and learns from other nodes without any of them sharing
footage.

This reference describes the code on `feat/threat-detection` as of
2026-10-05 (commit `a5f2c39` plus the working-tree changes listed in
section 10.4). That branch is the real system. `main` and
`master` on GitHub are older and do not have most of what follows.

---

## What it does, in one pass

```
 camera ──► motion ──► YOLO (COCO) ──► threat model ──► overlay ──► live stream
              │            │                 │
              │            └──── detections ─┴──► event recorder ──► .mp4 clips
              │                                │
              └────────────────────────────────┴──► event collector
                                                       │ one row per track
                                                       ▼
                                             sample store (JSONL)
                                                       │ labelled by a person
                                                       ▼
                                        federated head (19 → 5 MLP)
                                                       │ Flower round
                                                       ▼
                                  aggregator averages weights from every node
                                                       │
                                                       ▼
                                  validation gate ──► accepted model goes live
```

Three kinds of model run on a node, and only one of them learns:

| Model | Question it answers | Where it comes from | Learns on the node? |
|---|---|---|---|
| YOLOv8n (COCO) | What is in the frame: person, vehicle, animal, package? | Ultralytics pretrained weights | No, frozen |
| Threat model (YOLO11s, 2 classes) | Is this person carrying a weapon? | Third-party thesis weights, AGPL-3.0 | No, frozen |
| Federated head (numpy MLP, ~200K params) | Is this detection routine *for this site*? | Trained on this node's labelled events, averaged across nodes | **Yes** |

The federated head is the project's research contribution. YOLO already
says "person"; the head learns that a person at the side gate at 03:00 is
not routine here, even if it is routine somewhere else.

---

## The four ARGUS classes

YOLO emits 80 COCO classes. ARGUS keeps the ones that matter at a front
door and folds them into four (`src/detection/__init__.py`, `_YOLO_CLASS_MAP`):

| ARGUS class | COCO classes behind it |
|---|---|
| person | person |
| vehicle | bicycle, car, motorcycle, bus, train, truck |
| animal | bird, cat, dog, horse, sheep, cow |
| package | backpack, handbag, suitcase |

COCO has no parcel class. A cardboard box detects as nothing; a backpack
detects as a package. Real parcel detection needs a custom-trained model.

The head then labels each event with one of five outcomes:
`routine_person`, `routine_vehicle`, `routine_animal`, `routine_package`,
`anomaly`.

---

## Processes and threads on a running node

`python main.py` (or `scripts/run_node.sh`, which wraps it) starts one
FastAPI process. Inside it:

| Thread / task | Code | Job |
|---|---|---|
| uvicorn event loop | `src/api/app.py` | Serves the API, the built-in dashboard, the MJPEG stream, the WebSocket |
| `argus-capture` | `src/detection/worker.py` | Reads the camera into a one-frame slot; feeds every frame to the recorder |
| `argus-detect` | `src/detection/worker.py` | Runs the detection cascade on the newest frame only |
| FL scheduler task | `src/federated/scheduler.py` | Wakes hourly; runs a round every 14 days at 02:00 (only if `FL_ENABLED`) |

A federated round never trains inside that process. It launches
`python -m src.federated.node_client` as a separate OS process so training
competes with detection through the OS scheduler, pinned to one BLAS
thread, and a crash in training cannot take the camera down.

---

## The machines

| Machine | Runs | Ports |
|---|---|---|
| Raspberry Pi 5 + AI HAT+ 2 (the node) | `main.py` via `scripts/run_node.sh` | 8000 (API + dashboard) |
| Laptop (the aggregator) | `central_server/fl_aggregator.py` via `scripts/run_central_server.ps1` | 8080 (Flower gRPC), 8090 (REST + `/metrics`) |
| Laptop (optional) | The Next.js product dashboard, `adam9798/argus-dashboard` | 3000 / 3001 |
| Any machine | The node against a video file (`scripts/mock_demo.py`) or the simulator (`sim/`) | 8000, 8080, 9108 |

Only weights cross the network to the aggregator. No video, no images.

---

## Three Flower servers, and which to use

The repository has three ways to start a Flower server. They are not
interchangeable.

| Server | File | Strategy | Use it for |
|---|---|---|---|
| **Central aggregator** | `central_server/fl_aggregator.py` | `ArgusStrategy` (robust aggregation + checkpoints + metrics), Flask REST API on 8090 | Real rounds between real nodes. This is the one a deployed node talks to. |
| **Simulator server** | `sim/fl.py serve` | Robust aggregation (`--aggregation fedavg` for the baseline) | Quick local rounds, `scripts/fl_trial.sh`, and multi-client simulations |
| Legacy server | `server/app.py` | FedAvg or FedProx | The original YOLO-weights design. Not used by the current node. |

The first two, and the minimal `src/federated/server.py`, share one
strategy class: `RobustFedAvg` in `src/federated/robust_strategy.py`
(section 5.10). It screens every update through `src/federated/robust.py`
before averaging. Plain FedAvg stays available behind one switch
(`aggregation="fedavg"`), so the comparison study runs both arms through
identical server code.

---

## Repository layout

```
main.py                 node entry point (uvicorn on src.api.app)
src/
  config.py             every setting, read from environment variables
  camera/               board detection and capture backends
  detection/            motion, YOLO, threat, overlay, worker, recorder
  federated/            features, collector, store, head, gate, rounds,
                        node client, scheduler, robust aggregation
  api/                  routes, auth, dashboard contract, trial endpoints
  database/             SQLite: faces, events, devices, automations, audit, users
  security/             PBKDF2 hashing, bearer tokens, audit log
  automation/           device and rule models (stubbed)
  agents/               error-routing helper agents (dev tooling)
  training/             LEGACY: YOLO fine-tuning dataset and trainer
central_server/         the aggregator, Prometheus, Grafana, tunnel config
server/                 LEGACY Flower server
sim/                    node simulator and synthetic multi-site workload
scripts/                runners, exports, benchmarks, tools
deploy/                 systemd unit and settings template for a node
static/index.html       the built-in live dashboard
tests/                  328 tests, pytest
docs/                   this reference, guides, runbooks, proposal
research/               WiFi CSI, sensor fusion, early FL prototype
prometheus/             voice assistant orchestrator (separate tool)
```

Legacy code is described in [section 7](07-scripts-and-tools.md) so it is
not mistaken for the running path.
