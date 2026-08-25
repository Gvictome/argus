# Face Recognition

ArcFace-based identity matching for the ARGUS detection pipeline.
Showcase lane G-1 through G-8 (`docs/SHOWCASE_SPRINT_PLAN.md`).

---

## 1. How it works

| Stage | Component | Notes |
|-------|-----------|-------|
| Face detection | RetinaFace, via InsightFace `buffalo_l` | Bundled in the same model pack |
| Embedding | ArcFace, 512-d, L2-normalized | `face.normed_embedding` |
| Matching | Cosine similarity against enrolled faces | In-memory cache, rebuilt from SQLite |
| Storage | `faces` table, embedding as a pickled BLOB | `src/database/__init__.py` |

Implementation lives in `src/detection/face_recognition.py`. It is wired into
`DetectionService` at startup by `src/api/app.py` via `attach_face_recognizer()`.

### Where it sits in the cascade

`DetectionService.process_frame()` runs three stages in increasing cost order:

```
motion (frame differencing)  ->  YOLOv8n objects  ->  ArcFace faces
```

Faces run only after YOLO places a **human** in the frame, because ArcFace is
by far the most expensive stage. There is one deliberate exception: when the
object model failed to load (`self.object_model is None`), motion alone is
enough to reach face recognition. Without that fallback, any YOLO load failure
silently disables the headline feature with no visible error.

---

## 2. Setup

```bash
pip install -r requirements.txt
```

`insightface` and `onnxruntime` are declared but imported lazily. If they are
missing, the API still starts: `create_app()` logs
`Face recognition unavailable: ...`, leaves `face_recognizer` as `None`, and
detection degrades to Haar-cascade face *detection* with no identity.

The `buffalo_l` model pack (~300MB) downloads on first construction of
`FaceRecognitionService`. **Do this once with network access before the
showcase** - the demo unit runs offline and a cold model download at a booth
table is a guaranteed failure.

Verify the pack is cached:

```bash
ls ~/.insightface/models/buffalo_l
```

---

## 3. API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/faces` | List enrolled faces |
| `POST` | `/api/faces?name=<name>` | Enroll from the current camera frame |
| `DELETE` | `/api/faces/{face_id}` | Remove one face |
| `POST` | `/api/faces/reset` | Clear all enrolled faces (P1-6) |
| `GET` | `/api/detection/status` | FPS, backend, which models are live |

All face routes return `503` when the recognizer failed to initialize, so a
caller can never mistake "identity is off" for "nobody was recognized".

### Resetting between judges

```bash
curl -X POST http://<pi>:8000/api/faces/reset
```

Over a multi-hour showcase the known-faces database grows with every
passer-by. An unpruned database degrades match quality and eventually
resolves the *wrong* identity in front of a judge, which is worse than
recognizing nobody. Reset between sessions.

### Detection status

```json
{
  "status": "running",
  "fps": 12.4,
  "backend": "hailo",
  "motion_detection": true,
  "object_detection": true,
  "face_recognition": true,
  "known_faces": 3
}
```

`backend` is `hailo` when `yolov8n_hailo_model.hef` was found and loaded,
`cpu` for stock YOLOv8n, `none` when no object model loaded. `fps` is a
rolling mean over the last 30 processed frames, so it reflects current
behavior rather than a lifetime average; it reads `0.0` until two frames
have been processed.

---

## 4. Hardware verification (G-3)

Enrollment logic is covered by tests, but the tests fake the camera and
InsightFace. Run this **on the Pi** to exercise the real path:

```bash
python scripts/verify_enrollment.py --name "Giovanny"
```

It initializes the camera and ArcFace, enrolls from a live frame, confirms
the row hit the database, re-captures, matches, and prints the similarity
score with its margin over the threshold:

```
  best similarity: 0.7314
  threshold:       0.4000
  margin:          +0.3314
```

The margin is the number worth recording. "It worked" tells you nothing about
how close you were to a miss.

The test enrollment is deleted afterwards unless you pass `--keep`. Exit code
is `0` on success, `1` on a verification failure, `2` on a missing
prerequisite (camera, InsightFace, or database).

---

## 5. Threshold

`FACE_SIMILARITY_THRESHOLD` defaults to `0.4` (`src/config.py`, overridable
via the environment). Cosine similarity, so higher is stricter.

**This value is not yet tuned.** Tuning and its written justification are
G-6, due Sep 27. The tradeoff to document: a false reject is a shrug, a false
accept in front of a judge is a credibility problem. Tune toward rejection.

---

## 6. Tests

```bash
python -m pytest tests/test_face_recognition.py tests/test_face_api.py tests/test_detection_service.py -q
```

InsightFace is faked at the module boundary (`_FaceAnalysis`,
`_INSIGHTFACE_AVAILABLE`), so the suite runs without ONNX runtimes or the
model download. The database is the **real** sqlite `Database` against a
tmp file, so the schema and the pickle round-trip are exercised for real.
