# Jetson Orin Migration

> ## Parked — the project targets Raspberry Pi
>
> The build is one Raspberry Pi 5 + AI HAT+ 2 per camera. See
> [`HAILO_PIPELINE.md`](HAILO_PIPELINE.md) for the live path.
>
> Nothing here has been deleted, because none of it is dead weight: the
> platform detection this document describes is what makes the camera
> layer pick the right backend per board, and it fixed a real bug that
> would have broken a Jetson deployment. It stays as a working migration
> path, not a maintained target.

Moving ARGUS from the Raspberry Pi 5 to a Jetson Orin Nano as the host,
with two CSI cameras and federated learning against the Pi.

```
   [cam 0] ──CSI──┐
                  ├── Jetson Orin Nano ── ethernet
   [cam 1] ──CSI──┘    (M.2 Key E wifi)
                              │
                              │  Flower, every 14 days
                              ▼
                       Raspberry Pi 5
                       (FL client)
```

The Orin runs capture, inference, the API, the dashboard, **and** the
Flower server. The Pi stays alive as the second FL client, which is what
makes the federation real rather than a single node talking to itself.

---

## 1. The bug this fixes first

ARGUS decided its platform with:

```python
IS_RASPBERRY_PI = platform.machine().startswith('aarch') or ...
```

**A Jetson Orin is aarch64.** That expression is true on one, so the
camera service would import `picamera2` — which does not exist on
JetPack — fail, and report "camera initialization failed". A software
bug wearing a hardware error's clothes.

`platform.machine()` describes the CPU, not the board. Detection now
reads the device tree (`src/camera/platform_detect.py`):

| Signal | Meaning |
|--------|---------|
| `/etc/nv_tegra_release` exists | Jetson (written by the L4T BSP) |
| `/proc/device-tree/model` contains "jetson"/"orin"/"tegra" | Jetson |
| ...contains "Raspberry Pi" | Pi |
| neither | generic (laptop, CI) |

Confirm on the Orin:

```bash
python -c "from src.camera.platform_detect import BOARD, board_description; print(BOARD, '|', board_description())"
# jetson | NVIDIA Jetson Orin Nano Developer Kit (aarch64)
```

---

## 2. Camera hardware: read this before ordering

You chose to reuse the **Pi Camera Module 3 (IMX708)**. Be aware of what
that costs, because it is the single most likely thing to block this
migration:

| Sensor | JetPack support |
|--------|-----------------|
| IMX219 (Pi Camera v2) | **In-tree.** Works out of the box. |
| IMX477 (Pi HQ Camera) | **In-tree.** Works out of the box. |
| **IMX708 (Pi Camera Module 3)** | **No NVIDIA driver.** Third-party ports exist and are inconsistent across JetPack releases. |

There is no IMX708 driver in JetPack. Options, in order of effort:

1. **Swap to IMX219 or IMX477.** Zero driver work. The 12MP/autofocus
   advantage of Module 3 does not survive being downscaled to 416–640px
   for inference anyway, so you lose little that ARGUS actually uses.
2. **Vendor driver.** Arducam and others sell IMX708 modules with
   Jetson driver packages. This is the path if you specifically want
   autofocus.
3. **Port the driver.** Real kernel work, and it breaks on JetPack
   upgrades.

The capture layer is sensor-agnostic — nothing in ARGUS names IMX708 —
so switching sensors is a config change, not a rewrite. **Verify the
sensor enumerates before building the rest of the demo on it.**

```bash
ls /dev/video*
v4l2-ctl --list-devices
dmesg | grep -i -E "imx|argus"
```

If `nvarguscamerasrc` cannot see it, no amount of ARGUS configuration
helps.

---

## 3. Two CSI cameras

The Orin Nano dev kit has two MIPI CSI-2 connectors. Each is one
`CameraService`; the only difference is `sensor-id` in its GStreamer
pipeline.

```bash
export CAMERA_SENSORS="0:front,1:back"
```

Format is `sensor_id:name`, comma-separated. Empty means one camera at
`CAMERA_INDEX`, so existing single-camera installs are unchanged.

The generated pipeline:

```
nvarguscamerasrc sensor-id=0
  ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,
    format=(string)NV12, framerate=(fraction)30/1
  ! nvvidconv flip-method=0
  ! video/x-raw, format=(string)BGRx
  ! videoconvert ! video/x-raw, format=(string)BGR
  ! appsink drop=true max-buffers=1 sync=false
```

Four parts of that matter:

- **`nvarguscamerasrc`** is the only way to reach the Orin's ISP. A plain
  `cv2.VideoCapture(0)` opens V4L2 devices and **will not see a CSI
  camera**.
- **`memory:NVMM`** keeps frames in hardware memory; **`nvvidconv`** is
  the hardware converter out of it. Skip either and every frame takes a
  CPU copy, discarding the point of CSI.
- **`drop=true max-buffers=1`** means a slow consumer gets the *latest*
  frame, not a backlog. Without it, latency grows without bound whenever
  inference falls behind capture — and a late security frame is worthless.
- **`format=BGR`** because everything downstream in ARGUS is BGR.

Test a sensor outside Python before blaming ARGUS:

```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
  nvvidconv ! nvegltransform ! nveglglessink
```

OpenCV must also be built **with** GStreamer, which the pip wheel is not:

```bash
python -c "import cv2; print('GStreamer' in cv2.getBuildInformation())"
```

If that prints `False`, use the JetPack-provided OpenCV
(`sudo apt install python3-opencv`) and create the venv with
`--system-site-packages`, exactly as on the Pi.

### Endpoints

| Method | Path |
|--------|------|
| `GET` | `/api/cameras` — all cameras and the detected board |
| `POST` | `/api/cameras/init` — open all, per-camera results |
| `GET` | `/api/cameras/{name}/status` |
| `POST` | `/api/cameras/{name}/init` |
| `GET` | `/api/cameras/{name}/snapshot` |
| `GET` | `/api/cameras/{name}/stream` — annotated MJPEG |
| `POST` | `/api/cameras/{name}/shutdown` |

The single-camera endpoints (`/api/camera/...`) still act on the primary
sensor, so the dashboard and existing scripts keep working.

One caveat: all cameras share one `DetectionService`, so they share its
motion history and FPS window. That is right for the demo — one pipeline,
several views — but two simultaneous annotated streams will interleave
motion state. Run one at a time for a clean FPS reading.

---

## 4. TensorRT

The Orin's advantage over the Pi is its GPU, and reaching it means
compiling to a TensorRT engine.

```bash
python scripts/export_tensorrt.py            # both models, FP16
python scripts/export_tensorrt.py --check    # what exists
```

**Engines are not portable.** TensorRT compiles for the exact GPU,
driver, and TensorRT version present at build time. You cannot build on a
laptop and copy; an engine built before a JetPack upgrade will refuse to
load after one. Build on the Orin, and rebuild after any JetPack change.

The build takes minutes per model. Do it during setup, never at a booth.

`DetectionService.initialize()` then selects, in order:

```
models/yolov8n.engine        -> backend "tensorrt"
yolov8n_hailo_model.hef      -> backend "hailo"
yolov8n.pt                   -> backend "cpu"
```

Verify:

```bash
curl localhost:8000/api/detection/status   # "backend": "tensorrt"
```

Also set the power mode — the Orin ships throttled:

```bash
sudo nvpmodel -m 0     # MAXN
sudo jetson_clocks
```

Benchmark honestly afterwards; do not assume the Pi's numbers transfer:

```bash
python scripts/benchmark_classes.py --images <folder of stills>
```

---

## 5. Federated learning every 14 days

### Why the schedule is persisted

With a 14-day interval and an in-memory timer, **any reboot inside the
window restarts the countdown**. A host rebooted weekly would run FL
*never*, while looking completely healthy. So the last-round timestamp is
written to `data/fl_state.json` and reloaded at startup.

An overdue round (host powered off across the due date) runs shortly
after startup rather than waiting another fortnight — after a 5-minute
grace, so it does not fire during the boot storm following a power cut.

A **failed** round does not advance the schedule. An unreachable server
is retried on the next wake, not deferred for two weeks.

### Configuration

```bash
export FL_ENABLED=true
export FL_ROUND_INTERVAL_DAYS=14
export FL_ROUND_HOUR=2                 # local time, idle window
export FL_SERVER_URL="<orin-ip>:8080"
```

Rounds run only when idle — no motion in the last 10 minutes — so
training never competes with live surveillance.

### Checking it

```bash
curl localhost:8000/api/federated/status
```

```json
{"enabled": true, "running": true, "interval_days": 14,
 "last_round_at": "2026-09-01T02:00:00",
 "next_due_at": "2026-09-15T02:00:00",
 "seconds_until_next": 1209600, "idle": true}
```

`next_due_at` is the point. "It runs every two weeks" is unfalsifiable
without a visible next-due time, and a judge can reasonably ask.

### Topology

The Orin runs the Flower server (`central_server/fl_aggregator.py`) and
participates as a client. The Pi runs `run_client.py` pointed at the
Orin. Two nodes is the minimum for FedAvg to mean anything.

Both nodes must run the **same model architecture** — `set_model_weights`
matches weights by shape and order, so you cannot average a YOLOv8n on
one node against anything else on the other. Note this includes the
TensorRT question: FL exchanges PyTorch `.pt` weights, so **the Pi's
`.pt` model is what federates**, and the Orin's `.engine` is an inference
artifact rebuilt from the updated `.pt` after a round.

---

## 6. Setup order

```bash
# 1. JetPack 6.x, then confirm the board
python -c "from src.camera.platform_detect import board_description; print(board_description())"

# 2. System packages (OpenCV from apt, for GStreamer support)
sudo apt update
sudo apt install -y python3-opencv python3-venv python3-dev build-essential cmake

# 3. Venv that can see the system OpenCV
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip

# 4. Dependencies. Use NVIDIA's torch wheel for Jetson, not PyPI's --
#    the PyPI build has no CUDA support for aarch64.
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv httpx pytest numpy
pip install insightface onnxruntime
#    torch: follow NVIDIA's Jetson install guide for your JetPack version
pip install ultralytics

# 5. Cameras
ls /dev/video*
export CAMERA_SENSORS="0:front,1:back"

# 6. Models
python scripts/fetch_models.py
python scripts/export_tensorrt.py

# 7. Verify
python -m pytest tests/ -q
python main.py
curl localhost:8000/api/cameras
curl localhost:8000/api/detection/status
```

---

## 7. Troubleshooting

| Symptom | Cause |
|---------|-------|
| "picamera2 not installed" on the Orin | Stale build predating the platform fix. `BOARD` should read `jetson`. |
| CSI camera does not open | Sensor not enumerating (`ls /dev/video*`), or OpenCV built without GStreamer. Test with `gst-launch-1.0` first. |
| `nvarguscamerasrc` not found | GStreamer plugins missing; reinstall JetPack multimedia packages. |
| IMX708 never enumerates | Expected — no JetPack driver. See section 2. |
| Only one of two cameras opens | Check both connectors with `sensor-id=0` and `=1` via `gst-launch-1.0`. Ribbon seating is the usual cause. |
| `"backend": "cpu"` on the Orin | No `models/yolov8n.engine`. Run the export. |
| Engine fails to load after an update | Rebuild it — engines do not survive a JetPack upgrade. |
| FL never runs | `curl /api/federated/status`. If `last_round_at` keeps resetting, `data/fl_state.json` is not writable. |
| FL round skipped every time | The idle gate. Something is generating continuous motion. |

---

## 8. What has not been tested on hardware

Stated plainly, because none of this has run on an actual Orin yet:

- The CSI pipeline is verified as a **string**, not against a sensor.
- The TensorRT export path is verified to **refuse correctly** on
  non-Jetson hardware; the build itself is untested.
- Two-camera operation is tested through the registry and the API, with
  no real second sensor.
- IMX708 on Jetson is, per section 2, **expected not to work** without
  vendor driver support.

The platform detection, camera registry, schedule arithmetic, and
persistence are all covered by tests that run anywhere
(`tests/test_jetson.py`, `tests/test_fl_schedule.py`).
