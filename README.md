# ARGUS
## Autonomous Residential Guardian & Utility System

An edge camera node that detects what is in front of it, decides whether
that is routine for the place it is watching, records the moments that
matter, and learns from other nodes without any of them sharing footage.

Senior Design 2026. Showcase 2026-11-22.

> **Work on `feat/threat-detection`.** That branch is the real system.
> `master` and `main` are older and do not have most of what is described
> here.

---

## What runs on a node

```
camera → motion → YOLOv8n (person / vehicle / animal / package) → threat model (armed person?)
           │                         │
           │                         ├─► live annotated stream and dashboard
           │                         └─► event clips with 5 s pre-roll
           └─► one event per track → 19 features → labelled by a person
                                         → federated head (is this routine HERE?)
                                         → Flower rounds, weights only → validation gate
```

| Model | Job | Learns? |
|---|---|---|
| YOLOv8n (COCO) | What is in frame | No, frozen |
| Threat model (YOLO11s, AGPL-3.0, third party) | Is this person carrying a weapon | No, frozen |
| Federated head (numpy MLP, ~200K params) | Is this detection routine for this site | **Yes**, on each node, averaged across nodes |

## Hardware

| Component | Model |
|---|---|
| Board | Raspberry Pi 5 (8 GB) |
| Accelerator | Raspberry Pi AI HAT+ 2 (Hailo-10H, 40 TOPS INT8). Not yet used: needs a `.hef` compiled on x86_64 Linux |
| Camera | Raspberry Pi Camera Module 3 |
| Aggregator | Any laptop (Windows scripts provided) |

The camera layer also supports Jetson Orin (CSI over GStreamer) and any
OpenCV camera or video file.

## Quick start

```bash
git clone https://github.com/Gvictome/argus.git && cd argus
git checkout feat/threat-detection
python -m venv .venv && source .venv/bin/activate     # Windows: .venv/Scripts/activate
pip install -r requirements.txt
python scripts/fetch_models.py                         # threat-model weights

./scripts/run_node.sh                                  # on the Pi
python scripts/mock_demo.py --reset                    # anywhere, against media/vtest.avi
python -m pytest -q                                    # 303 tests
```

Open `http://<node>:8000/dashboard`.

A federated simulation with a real Flower server and three clients:

```bash
python sim/fl.py serve --rounds 5 --clients 3
python sim/fl.py client --site driveway     # and --site backdoor, --site yard, in their own shells
```

Every command, by machine: [`docs/reference/02-how-to-run.md`](docs/reference/02-how-to-run.md).

## Documentation

**Start with the master reference PDF:**
[`docs/pdf/ARGUS_Master_Reference.pdf`](docs/pdf/ARGUS_Master_Reference.pdf).
It contains everything below in one file. Rebuild it with
`python scripts/build_docs_pdf.py --master`.

| | |
|---|---|
| [`docs/README.md`](docs/README.md) | Map of every doc folder: reference, guides, operations, design, research, project, archive |
| [`docs/research/datasets.md`](docs/research/datasets.md) | The dataset plan: MEVA replaces CIFAR-10, plus Stanford Cars and ShanghaiTech |
| [`docs/reference/`](docs/reference/) | The technical reference: overview, how to run, computer vision, threat classification, federated learning, API and data, scripts, configuration, tests, status and gaps |
| [`docs/guides/`](docs/guides/) | Eight plain-language guides, one topic each |
| [`docs/operations/QUICKSTART.md`](docs/operations/QUICKSTART.md) | Every demo command, labelled PI or LAPTOP |
| [`sim/README.md`](sim/README.md) | The node simulator, and what it can and cannot prove |
| [`docs/design/THREAT_DETECTION.md`](docs/design/THREAT_DETECTION.md), [`docs/operations/HAILO_PIPELINE.md`](docs/operations/HAILO_PIPELINE.md), [`docs/operations/CLOUDFLARE_TUNNEL.md`](docs/operations/CLOUDFLARE_TUNNEL.md) | Deep dives |
| [`docs/project/PROPOSAL.md`](docs/project/PROPOSAL.md) | The project proposal |

## Repository layout

```
main.py              node entry point
src/camera/          board detection and capture backends
src/detection/       motion, YOLO, threat, overlay, worker, recorder
src/federated/       features, collector, store, head, gate, rounds, client, scheduler, robust aggregation
src/api/             routes, auth, dashboard contract, federated endpoints
central_server/      Flower aggregator + REST + Prometheus/Grafana
sim/                 node simulator and synthetic multi-site workload
scripts/             runners, exports, benchmarks, tools
tests/               pytest suite
docs/                reference, guides, runbooks, proposal
```

## Team

| Name | Role | Focus |
|---|---|---|
| Christian | Electrical Engineer | Hardware, power, sensors |
| Giovanny | Computer Engineer | AI/ML, detection, federated learning |
| Mohammed | Computer Engineer | Backend, API, cloud |
| Saifeddine (Adam) | Computer Science | Frontend, architecture, security audits |

---

## License

**AGPL-3.0**, see [`LICENSE`](LICENSE).

ARGUS links Ultralytics and ships a third-party threat-detection model,
both AGPL-3.0. AGPL's network clause reaches ARGUS because it serves video
over a network. Third-party models, datasets, and their terms are recorded
in [`NOTICE.md`](NOTICE.md).
