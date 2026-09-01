# Hailo Pipeline

Running ARGUS on Raspberry Pi 5 with a Hailo accelerator, one Pi per
camera, targeting 8–12 FPS.

```
   [Pi 5 + AI HAT+ 2 + camera]  ──┐
                                  │   weights only, every 14 days
   [Pi 5 + AI HAT+ 2 + camera]  ──┼──▶  central server (global model)
                                  │
   [Pi 5 + AI HAT+ 2 + camera]  ──┘
```

Each camera node is a complete ARGUS install: capture, inference, face
recognition, event recording, and one federated-learning client.

---

## 1. Know which board you have

The AI HAT+ 2 is a **different chip** from the AI HAT+, not a faster
revision. A model compiled for one will not load on the other, and the
runtime packages differ.

| Board | Accelerator | Vision perf | Memory | Runtime package |
|-------|-------------|-------------|--------|-----------------|
| AI Kit / AI HAT+ 13 TOPS | Hailo-8L | 13 TOPS | Pi RAM | `hailo-all` |
| AI HAT+ 26 TOPS | Hailo-8 | 26 TOPS | Pi RAM | `hailo-all` |
| **AI HAT+ 2** | **Hailo-10H** | 40 TOPS INT8 / 26 TOPS INT4 | **8 GB onboard** | `hailo-h10-all` |

```bash
sudo apt install hailo-h10-all      # AI HAT+ 2
sudo reboot
hailortcli fw-control identify      # must see the board
```

If `fw-control identify` finds nothing, stop here. Nothing downstream
works until the board enumerates.

---

## 2. Compilation happens on a workstation, not the Pi

**The Hailo Dataflow Compiler is x86_64 Linux only.** You cannot compile
on the Pi. This is the exact opposite of the TensorRT flow, where the
engine *must* be built on the target — easy to get backwards.

```bash
# On an x86_64 Linux box (or WSL2), with the Hailo DFC installed:
python scripts/export_hailo.py --hw-arch hailo10h     # AI HAT+ 2
python scripts/export_hailo.py --hw-arch hailo8       # AI HAT+
python scripts/export_hailo.py --check                # what exists
```

The script verifies the platform and refuses rather than failing halfway
through a build that takes tens of minutes.

Then copy the **directory**, not just the `.hef`:

```bash
scp -r models/yolov8n_hailo_model pi@<host>:~/argus/models/
```

### Why a directory

`format="hailo"` produces an export directory holding the HEF *plus*
metadata describing input size, classes, and NMS. Ultralytics loads the
directory: `YOLO("yolov8n_hailo_model")`. The HEF alone is not enough.

ARGUS previously probed for a bare `yolov8n_hailo_model.hef` file, which
`format="hailo"` never produces — so on a correctly-exported install the
accelerator was never found and the Pi silently ran on CPU while
reporting `"backend": "cpu"`. `_find_hailo_model()` now checks, in order:

```
models/yolov8n_hailo_model/     <- the normal case
models/yolo11n_hailo_model/
yolov8n_hailo_model/
yolov8n_hailo_model.hef         <- legacy hand-placed file
```

Backend selection remains `tensorrt > hailo > cpu`. Verify:

```bash
curl localhost:8000/api/detection/status     # "backend": "hailo"
```

---

## 3. Hitting 8–12 FPS

8–12 FPS is a comfortable target on this hardware, but **not because of
YOLO** — and knowing where the time actually goes is what keeps you from
optimising the wrong stage.

Published throughput for YOLOv8n at 640px:

| Accelerator | FPS |
|-------------|-----|
| Hailo-8L (13 TOPS) | 60–80 |
| Hailo-8 (26 TOPS) | 130–160 |
| Hailo-10H (AI HAT+ 2) | higher still |

Real-world numbers run below the published ones — users report roughly
24 FPS with YOLO11m on the 26 TOPS board — but every figure here clears
12 FPS by a wide margin.

### The actual bottleneck is ArcFace

Offloading YOLO to the Hailo does not offload the rest of the cascade:

| Stage | Runs on | Notes |
|-------|---------|-------|
| Motion | CPU | Cheap; frame differencing |
| Objects (YOLO) | **Hailo** | The stage that gets fast |
| Threat (YOLO11) | **Hailo** once compiled | Compile it too, or it stays on CPU |
| **Faces (ArcFace)** | **CPU** | InsightFace has no Hailo path here |

ArcFace is the expensive CPU stage, and it is gated on a human being
detected — which, at a booth full of people, is most frames. **That is
what will decide whether you hold 12 FPS**, not object detection.

Levers, in order of preference:

1. **Raise `detect_every`** on the stream (default 3). Boxes persist
   between runs, so the video stays smooth while the cascade runs less
   often. Cheapest win by far.
2. **Compile the threat model to Hailo too.** Otherwise it is a second
   CPU YOLO pass at ~36 ms/frame, undoing much of the benefit.
3. **Lower the ArcFace detector size.** `det_size` is set at construction
   in `FaceRecognitionService`.
4. Leave `THREAT_IMGSZ` at 416 — 320 misses threats entirely, measured.

Benchmark on the actual Pi before assuming any of this:

```bash
python scripts/benchmark_classes.py --images <folder of stills>
```

---

## 4. One Pi per camera changes the federation

This topology is better for the federated-learning story than a single
multi-camera host, and it is worth saying why on the poster.

- **Each node trains on its own footage.** Data never leaves the Pi it
  was captured on. That is the privacy claim, structurally rather than
  by policy.
- **Three camera nodes is three FL contributors.** The Byzantine-robust
  aggregation in `src/federated/robust.py` needs **three** contributors
  before consensus filtering means anything — with two, if they disagree
  there is no way to tell which is wrong. A per-camera topology reaches
  that threshold naturally; a single host with two cameras never does,
  because it is one node.

See [`FEDERATED_ARCHITECTURE.md`](FEDERATED_ARCHITECTURE.md).

---

## 5. Per-node setup

```bash
# 1. Runtime for your board
sudo apt install hailo-h10-all && sudo reboot
hailortcli fw-control identify

# 2. System packages (OpenCV and picamera2 from apt, not pip)
sudo apt install -y python3-picamera2 python3-opencv python3-venv \
                    python3-dev build-essential cmake

# 3. Venv that can see them
python3 -m venv --system-site-packages .venv
source .venv/bin/activate && pip install --upgrade pip

# 4. Dependencies
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv \
            httpx pytest numpy insightface onnxruntime ultralytics

# 5. Models: fetch the .pt, copy the compiled Hailo exports from the
#    workstation
python scripts/fetch_models.py
# scp -r models/yolov8n_hailo_model  pi@<host>:~/argus/models/

# 6. Verify
python -m pytest tests/ -q
python main.py
curl localhost:8000/api/detection/status
```

---

## 6. Troubleshooting

| Symptom | Cause |
|---------|-------|
| `"backend": "cpu"` with a HAT installed | No Hailo export in `models/`, or only a bare `.hef` where the directory is expected. |
| `hailortcli` finds nothing | Wrong runtime package for the board, PCIe ribbon not seated, or no reboot after install. |
| Export fails on the Pi | Expected — the DFC is x86_64 Linux only. Compile on a workstation. |
| HEF loads but detects nothing | Compiled for the wrong `--hw-arch`. Hailo-8 and Hailo-10H are not interchangeable. |
| FPS below 8 with a working Hailo | ArcFace on CPU. Raise `detect_every`, compile the threat model, or reduce the face detector size. |

---

## 7. What has not been tested

No Hailo hardware has been run against this code. Verified here:

- The probe logic, including the directory-vs-`.hef` regression
  (`tests/test_jetson.py::TestHailoProbe`).
- The export script's platform refusal.

Not verified: that a compiled HEF loads and infers correctly through
Ultralytics on a Pi, and the real end-to-end frame rate. Both need the
hardware.
