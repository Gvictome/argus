# 3 · Computer vision: camera to boxes to clips

This section covers every module between the image sensor and the saved
footage. The threat model is in section 4; turning detections into
training data is in section 5.

---

## 3.1 The frame path

```
CaptureBackend.read()          src/camera/backends.py     one BGR frame
        │
CameraService.get_frame_array  src/camera/__init__.py     rotation, colour swap
        │
DetectionWorker._capture_loop  src/detection/worker.py    latest-frame slot ──► EventRecorder.process
        │                                                                       (every frame, for pre-roll)
DetectionWorker._detect_loop   src/detection/worker.py    newest frame only
        │
DetectionService.process_frame src/detection/__init__.py
        ├─ detect_motion       always
        ├─ detect_objects      on motion, or every frame if MOTION_GATE=false
        ├─ detect_threats      same gate as objects, if a ThreatClassifier is attached
        └─ collector.observe   every frame (section 5)
        │
stream_from_worker             src/detection/worker.py    draw_detections ─► resize ─► JPEG ─► MJPEG
```

Every frame inside ARGUS is a numpy array in **BGR** order, OpenCV's
convention. Boxes are `(x, y, width, height)` in pixels unless a function
says otherwise.

---

## 3.2 `src/camera/platform_detect.py`: which board is this?

`detect_board()` reads `/proc/device-tree/model` and `/etc/nv_tegra_release`
and returns `Board.RASPBERRY_PI`, `Board.JETSON` or `Board.GENERIC`.

Why it exists: the old check, `platform.machine().startswith('aarch')`, is
also true on a Jetson Orin. The node would then import `picamera2` on a
Jetson, fail, and report a camera fault that was really a software
mistake. CPU architecture is not the board.

Module-level constants `BOARD`, `IS_RASPBERRY_PI`, `IS_JETSON` are
re-exported from `src.camera`.

---

## 3.3 `src/camera/backends.py`: one class per board family

Every backend implements the same three methods, `open() -> bool`,
`read() -> ndarray | None`, `close()`, and none of them raise.

| Backend | Board | How it reads |
|---|---|---|
| `Picamera2Backend` | Raspberry Pi | `picamera2` / libcamera; must come from `apt`, needs a venv with `--system-site-packages` |
| `JetsonCSIBackend` | Jetson Orin | GStreamer `nvarguscamerasrc` pipeline built by `build_csi_pipeline()`; the only way to reach a CSI sensor through the Orin's ISP |
| `OpenCVBackend` | anything else, and any file | `cv2.VideoCapture`; a video file loops at the end, and with `realtime=True` is paced to its own frame rate |

`build_backend(board, sensor_id, config)` picks one. **An explicit
`CAMERA_SOURCE` wins over the board**: a file path or a numeric index
always gets `OpenCVBackend`, which is how the whole pipeline runs against a
recording on a laptop.

`rotation_to_flip_method()` maps 0/90/180/270 degrees onto the GStreamer
flip codes.

---

## 3.4 `src/camera/__init__.py`: the camera service

| Name | Role |
|---|---|
| `CameraConfig` | resolution, framerate, rotation, `sensor_id`, name, `swap_rb`, `source`, `realtime` |
| `CameraService` | Wraps one backend. `initialize()`, `get_frame_array()` (BGR array), `get_frame()` (JPEG bytes), `capture_snapshot()`, `start_recording()` / `stop_recording()`, `get_status()` |
| `CameraRegistry` | Several `CameraService`s by name, for the Orin's two CSI connectors. `build_registry_from_settings()` parses `CAMERA_SENSORS="0:front,1:back"` |
| `camera_service` | Module singleton for camera 0, used by the detection worker and the single-camera routes |

`CAMERA_SWAP_RB=true` swaps red and blue. Use it only if people look blue
on the live feed; the Pi camera already returns BGR through the backend.

---

## 3.5 `src/detection/worker.py`: the background loop

`DetectionWorker(camera, detector, recorder)` owns the camera. Nothing
else may read it: a second reader gets empty frames. Anything that needs a
picture asks the worker via `latest()`.

Two daemon threads:

* **`argus-capture`** opens the camera on its own thread (so startup never
  blocks on hardware), then loops: read a frame, store it in a one-frame
  slot under a lock, pass it to `EventRecorder.process()` together with the
  most recent detections. The recorder needs every frame, not only the
  detected ones, or its pre-roll would have gaps.
* **`argus-detect`** takes the newest frame, runs `process_frame()`, stores
  the detections. Frames that arrived while it was busy are **dropped**,
  not queued, so the boxes always describe what is in front of the camera
  now rather than seconds ago.

`status()` reports `capture_fps`, `detect_fps` (30-sample sliding window),
`detect_ms` (exponential average), `frames_captured`, `frames_detected`,
`frames_not_detected`, and `camera_error` if the camera never opened.

`stream_from_worker(worker, jpeg_quality=80, max_fps=15, stream_width=960)`
is the MJPEG generator behind `/api/camera/stream`. It never runs the
detector: it draws the worker's latest boxes on the worker's latest frame,
downscales to 960 px wide (a 1080p JPEG per frame was the most expensive
CPU stage left), encodes, and yields multipart parts. A viewer therefore
costs only draw plus encode.

`src/detection/stream.py` holds the older generator,
`stream_annotated_mjpeg()`, which detected inside the stream loop. It is
kept for its tests and for `BOUNDARY`; the running node uses the worker.

---

## 3.6 `src/detection/__init__.py`: the detection cascade

### Types

* `DetectionType`: `MOTION`, `HUMAN`, `FACE`, `ANIMAL`, `VEHICLE`,
  `PACKAGE`, `UNKNOWN`, `DANGEROUS_PERSON` (produced only by the threat
  stage).
* `Detection`: `type`, `confidence`, `bbox (x, y, w, h)`, `label`,
  `face_id`, `coco_id`. The raw COCO id is kept because "package" is
  recognised by id downstream.
* `DetectionConfig`: thresholds, `object_imgsz`, `motion_width`,
  `motion_gate`, `cpu_export`, `model_name`, `faces_enabled`. The app
  builds it from `Settings` at startup; before that fix, the dataclass
  defaults silently beat every environment variable.

### `DetectionService.initialize()`: choosing a backend

Probes files, fastest first, and loads the first one present:

1. `models/<model>.engine` (TensorRT, Jetson) → backend `tensorrt`
2. A Hailo export (`models/yolov8n_hailo_model`, `models/yolo11n_hailo_model`,
   or a bare `.hef`) → backend `hailo`
3. A CPU export (`models/<model>_ncnn_model` first on ARM,
   `models/<model>_openvino_model` first on x86) → backend `cpu-ncnn` /
   `cpu-openvino`
4. `<model>.pt` with PyTorch → backend `cpu`

For Hailo and CPU exports it reads the export's `metadata.yaml` and
overrides `object_imgsz` with the export's fixed input size, logging a
warning. A failed load is logged at ERROR with the traceback and leaves
`backend="none"`; it is loud on purpose because a silent fallback disables
detection without saying so.

If `ultralytics` is not installed, no model loads and object detection
returns nothing. This is the first thing to check when "nothing is ever
detected".

### `detect_motion(frame)`

Frame differencing on a downscaled copy (`MOTION_WIDTH`, default 320 px):
grayscale, Gaussian blur (kernel scaled with the image), `absdiff` against
the previous blurred frame, threshold at `MOTION_SENSITIVITY` (25), dilate,
external contours. Each contour bigger than 30×30 px (in full-resolution
coordinates) becomes a `MOTION` detection whose confidence is contour area
over box area. The first call returns nothing because there is no previous
frame.

Downscaling and reusing the previous blurred frame took 1080p motion from
17.8 to 22.9 FPS on the development laptop.

### `detect_objects(frame)`

One YOLO call with `imgsz=object_imgsz`, `conf=DETECTION_THRESHOLD` (0.5)
and `classes=` the 16 COCO ids ARGUS maps, so other classes are dropped
before NMS rather than after. Each box is converted from centre format to
`(x, y, w, h)` and typed through `_YOLO_CLASS_MAP` (table in section 1).

### `detect_threats(frame)`

Delegates to the attached `ThreatClassifier` (section 4). Any exception is
caught and logged; a failure in the newest stage must not take down object
detection.

### `process_frame(frame)`

```
motion = detect_motion(frame)
if motion or not motion_gate:
    objects = detect_objects(frame)
    if faces_enabled and (humans in objects or no object model):
        faces = recognize_faces(frame)          # deprecated, off by default
    threats = detect_threats(frame)             # not gated on a COCO person
collector.observe(all detections, frame)        # wrapped; never raises
return all detections
```

Two decisions worth knowing before changing this:

* **The motion gate saves CPU but never classifies a subject that holds
  still.** `MOTION_GATE=false` (`run_node.sh --every-frame`) runs YOLO on
  every frame, which is what requirement FR-19 asks for.
* **The threat stage is not gated on a COCO person box.** Gating it that
  way lost 20% of true positives on the 314-image test set (recall 0.800
  to 0.612), because COCO never found those people.

`status()` reports what actually loaded (backend, model, gate, the share of
frames where motion opened the gate, input size, whether threat detection
is attached), not what was configured.

### Face recognition (deprecated)

`src/detection/face_recognition.py` (`FaceRecognitionService`, ArcFace
512-d embeddings via InsightFace) and the Haar-cascade fallback remain in
the code behind `FACE_RECOGNITION_ENABLED=false`. They were removed from
the design on 2026-09-14: on CPU they took the pipeline from 23.4 to 0.86
FPS. The measured verification accuracy (0.9996 on LFW pairs,
`scripts/benchmark_face_accuracy.py`) is still a valid result to report,
with that cost stated alongside it.

---

## 3.7 `src/detection/annotate.py`: drawing the boxes

`draw_detections(frame, detections)` returns a **copy** with boxes and
labels burned in; the clean frame is still needed for recording.

| Colour (BGR) | Meaning |
|---|---|
| Orange `(0,165,255)` | `DANGEROUS_PERSON` |
| Red `(0,0,255)` | A face with no identity |
| Green `(0,255,0)` | Everything else |

Motion boxes are never drawn; they cover most of the frame. When a threat
box overlaps a COCO person box at IoU ≥ 0.5, only the threat box is drawn,
so one person does not carry two stacked labels. The detection list itself
is untouched.

Drawing happens server-side, before encoding, so a label can never drift
out of sync with the frame it describes.

---

## 3.8 `src/detection/event_recorder.py`: clips around events

The camera runs 24/7; recording does not. `EventRecorder.process(frame,
detections)` is called for every captured frame and decides on its own
when to open and close a clip.

| Behaviour | Default | Why |
|---|---|---|
| Triggers | person, dangerous person, face, vehicle, package (confidence ≥ 0.5) | Motion alone is excluded: clouds and branches would bring back 24/7 recording |
| Pre-roll | 5 s ring buffer | A clip that starts at the detection has already missed the approach |
| Post-roll | 10 s after the last trigger | Someone walking out of frame does not truncate the clip |
| Cooldown | continued detection extends the open clip | A loiterer gives one clip, not hundreds |
| Hard stop | 120 s | A stuck trigger cannot fill the disk |

Clips are written as `media/events/<camera>_<trigger>_<YYYYmmdd_HHMMSS>.mp4`
(mp4v, 15 FPS). On close, `on_clip` puts a row in the `events` table
(`event_type="clip"`), which is how the dashboard finds them. The recorder
is fed the **clean** frame: saved footage is evidence, not a screenshot of
the overlay.

Known gap: nothing deletes old clips. A long-running node fills its card.

---

## 3.9 Other detection files

| File | What it is |
|---|---|
| `src/detection/models.py` | `ModelManager`: load YOLOv8n and get / set / diff its weights as numpy lists. Used only by the legacy YOLO-weights FL path (section 7) |
| `src/detection/benchmark.py` | A bare YOLO FPS loop from the early project. Use `scripts/benchmark_classes.py` or `sim/run_node.py` instead |

---

## 3.10 Measured performance

Development laptop, four cores, pedestrian video:

| Change | Result |
|---|---|
| Removing face recognition | 0.86 → 23.4 FPS |
| Downscaled motion | 17.8 → 22.9 FPS at 1080p |
| Smaller YOLO input | about 60% faster, about 5% fewer detections |

On the Pi itself: the Hailo runs a large detector at 76 FPS through its own
benchmark tool, and the NCNN export works. Per the 2026-09-27 status
report, CPU-only on the Pi clears the revised 8-12 FPS target. **No
end-to-end frame rate through ARGUS on the Hailo has been recorded**,
because no `.hef` has been compiled yet. Do not quote laptop numbers as Pi
numbers.
