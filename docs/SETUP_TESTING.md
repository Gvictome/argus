# Setup and Testing on a New Device

How to get ARGUS running on a machine that has never had it, or update a
clone that predates the face recognition work.

> ### Read this first: the branch trap
>
> **This repository's default branch on GitHub is `master`, but all
> development happens on `main`.** `master` is a stale mirror and contains
> **no** face recognition code.
>
> A plain `git clone` checks out `master` and you will get none of it.
> **Always `git checkout main`.**

---

## 1. Get the code

### 1a. Fresh clone

```bash
git clone https://github.com/Gvictome/argus.git
cd argus
git checkout main            # REQUIRED - clone lands you on master
```

Confirm you actually have the work:

```bash
ls scripts/verify_enrollment.py src/detection/face_recognition.py
```

Both must exist. If they do not, you are on `master`.

### 1b. Updating a clone made earlier

```bash
cd path/to/argus
git status                   # check for local changes first
git stash                    # only if git status showed changes
git fetch origin
git checkout main
git pull origin main
git stash pop                # only if you stashed
```

If `git checkout main` errors with "pathspec did not match", the clone
never had the branch:

```bash
git fetch origin main:main
git checkout main
```

Verify you are current:

```bash
git log --oneline -1
# expect: Merge pull request #4 ... face-recognition-hardening (or later)
```

---

## 2. Environment

### 2a. Raspberry Pi 5 (Pi OS Bookworm or newer, 64-bit)

Confirm 64-bit first. `insightface` and `onnxruntime` have no 32-bit ARM
wheels, so a 32-bit OS is a dead end:

```bash
uname -m                     # must print aarch64
```

System packages. `picamera2` and OpenCV come from apt, not pip - building
either under pip on a Pi is slow and often fails:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev python3-picamera2 \
                    python3-opencv build-essential cmake
```

Create the venv **with `--system-site-packages`**. Without it the venv
cannot see the apt-installed `picamera2` and camera init fails:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
```

Then pick an install.

**Minimal (recommended for testing face recognition).** Skips torch,
torchvision, streamlit, and flwr, which are large and slow on a Pi:

```bash
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv \
            httpx pytest numpy insightface onnxruntime
```

Object detection will be unavailable without `ultralytics`, and
`/api/detection/status` will report `"object_detection": false`. Face
recognition still runs: the pipeline falls back to running faces on
motion alone when no object model is loaded.

**Full (needed for the YOLO cascade and federated learning).** Expect a
long install:

```bash
pip install -r requirements.txt
```

### 2b. Laptop or desktop (Windows / macOS / Linux)

No camera-specific packages. The camera service falls back to OpenCV
automatically on any non-ARM platform.

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Pre-download the face model

`buffalo_l` is roughly 300MB and downloads on first use. **Do this once
while you have network.** The demo unit runs offline, and a cold download
at a booth table is a guaranteed failure:

```bash
python -c "from insightface.app import FaceAnalysis; a = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); a.prepare(ctx_id=0, det_size=(640,640)); print('buffalo_l ready')"
```

Confirm it cached:

```bash
ls ~/.insightface/models/buffalo_l
```

---

## 4. Test without hardware

Run this first on any device. It needs no camera, no InsightFace, and no
model download - both InsightFace and ultralytics are faked at the import
boundary:

```bash
python -m pytest tests/test_face_recognition.py tests/test_face_api.py tests/test_detection_service.py -v
```

Expected: **35 passed**.

Full suite:

```bash
python -m pytest tests/ -q
```

Expected: **1 failed, 67 passed**. The one failure is
`test_camera_status`, which is pre-existing and unrelated to this work
(the camera status payload has no `status` key).

**What this proves:** matching math, threshold behavior, bbox coordinate
conversion, reset, and the HTTP contracts.
**What it does not prove:** that a real camera plus real ArcFace produce
a usable score on a real face. That is section 5.

---

## 5. Test with the camera (G-3)

Confirm the camera is visible to the OS first:

```bash
rpicam-hello --list-cameras      # older Pi OS: libcamera-hello --list-cameras
```

Then run the verification:

```bash
python scripts/verify_enrollment.py --name "Giovanny"
```

Sit facing the camera. It initializes hardware, waits for auto-exposure,
enrolls from a live frame, confirms the database write, re-captures, and
matches:

```
[PASS] Camera initialized (platform=picamera2)
[PASS] ArcFace loaded (threshold=0.4)
[PASS] Enrolled 'Giovanny' as 3f2a...
[PASS] Embedding persisted to the database

  best similarity: 0.7314
  threshold:       0.4000
  margin:          +0.3314

[PASS] Recognized 'Giovanny' at 0.7314
VERIFICATION PASSED
```

**Record the margin.** It is the point of the exercise - how much room
there is before a miss, and the raw data for threshold tuning (G-6). A
margin near `+0.30` is comfortable; near `+0.03` means one bad-lighting
frame from failing.

| Flag | Effect |
|------|--------|
| `--name` | Label to enroll under (default `VERIFY_TEST`) |
| `--keep` | Keep the enrollment; default deletes it so repeat runs do not pollute the database |
| `--settle` | Seconds to wait for auto-exposure (default 2.0) |

Exit codes: `0` pass, `1` verification failed, `2` prerequisite missing.

---

## 6. Run the live system

```bash
python main.py               # serves on 0.0.0.0:8000
```

Find the device address:

```bash
hostname -I
```

From any machine on the same network:

```bash
curl http://<ip>:8000/api/detection/status
curl -X POST "http://<ip>:8000/api/faces?name=Giovanny"
curl http://<ip>:8000/api/faces
curl -X POST http://<ip>:8000/api/faces/reset
```

Open `http://<ip>:8000/api/camera/stream` in a browser for the annotated
video. Green boxes are recognized people, red are unknown faces.

Set `DEBUG=true` for interactive docs at `/docs`; they are off by default.

### Reading the status response

```json
{"status":"running","fps":12.4,"backend":"hailo","motion_detection":true,
 "object_detection":true,"face_recognition":true,"known_faces":3}
```

| Field | Meaning if wrong |
|-------|------------------|
| `"face_recognition": false` | InsightFace failed to load. Check startup logs for `Face recognition unavailable`. |
| `"object_detection": false` | `ultralytics` missing or YOLO failed to load. Faces still run via the motion fallback. |
| `"backend": "none"` | No object model loaded at all. |
| `"fps": 0.0` | Fewer than two frames processed; nothing has hit the pipeline yet. |

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `error: externally-managed-environment` | Pi OS blocks system-wide pip (PEP 668). Use the venv. |
| `picamera2 not installed` inside the venv | The venv was created without `--system-site-packages`. Delete `.venv` and recreate it with that flag. |
| `insightface` build fails | `sudo apt install -y build-essential cmake python3-dev`, then retry. |
| No `onnxruntime` wheel | You are on a 32-bit OS. `uname -m` must say `aarch64`. |
| `Face recognition unavailable: ...` at startup | InsightFace or onnxruntime missing. The API still starts with identity disabled by design. |
| `Failed to load YOLO model` with a traceback | Object detection is off; faces still run on the motion fallback. Install `ultralytics` for the full cascade. |
| Enrollment finds no face | Lighting. The face must be well lit and fill a reasonable part of the frame. |
| Old faces still matching | `curl -X POST http://<ip>:8000/api/faces/reset`, or delete `data/database.db`. |
| `scripts/verify_enrollment.py` not found | You are on `master`. See section 1. |

`data/` and `media/` are gitignored and created automatically on first
run - no manual setup needed.
