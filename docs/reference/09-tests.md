# 9 · Tests

## 9.1 Running them

```bash
python -m pytest -q                       # everything
python -m pytest -q -m "not slow"         # skip process spawns, video decode, network timeouts
python -m pytest tests/test_threat.py -q  # one file
python -m pytest -q -k "gate"             # by name
```

**Last full run: 2026-10-02, branch `feat/threat-detection` at `2aaf327`,
Windows 11, Python 3.12, CPU-only torch: 303 passed, 0 failed, 0 skipped,
97 s.**

Prerequisites, all in `requirements.txt` and all needed for a clean run:

| Missing | Symptom |
|---|---|
| `pytest-asyncio` | 4 failures in `test_fl_schedule.py::TestRoundOutcome` ("async def functions are not natively supported") |
| `ultralytics` | `test_a_clip_becomes_labelled_events` fails: the ingest finds 0 events because no detector loads |
| `media/vtest.avi` | `test_a_clip_becomes_labelled_events` is skipped, and `test_an_unknown_label_is_rejected` **fails** (it checks the video before the label) |

Fetch the video with the `curl` line in section 2.1.

Gate any push on the exit code (`$?` is 0), never on reading the output. A
grep for "failed" also matches a passing summary line, which is how two
failing tests were once pushed.

## 9.2 How the suite stays fast and hermetic

`tests/conftest.py` sets two variables before anything imports
`src.config`:

* `DETECTION_AUTOSTART=false`, so no test grabs the developer's webcam.
* `OBJECT_BACKEND=pt`, so an export lying in `models/` cannot change which
  backend a test sees.

`pytest.ini` runs asyncio in strict mode, so an async test missing its
marker fails loudly instead of passing while asserting nothing. Models
are replaced by stubs almost everywhere, so most tests need no weights.

## 9.3 What each file covers

| File | Tests | Covers |
|---|---|---|
| `test_pipeline_changes.py` | 46 | The September changes, each pinned: face stage deprecated, downscaled motion, the detection worker, file camera source, **the model gate** (first model accepted and persisted, regressing candidate rejected, head restored at startup), dashboard corrections to labels, motion segments, head weight isolation, the motion gate, snapshots while the worker owns the camera, camera colour order, the ARGUS class taxonomy, model selection, stills and clips, fixed-shape exports |
| `test_threat.py` | 41 | Class index pinning, construction errors, whole-frame threat detection, crop padding and clamping, the cascade stage, status, overlay colour and dedup |
| `test_jetson.py` | 35 | Board detection from the device tree, backend selection, the GStreamer CSI pipeline, the camera registry, Hailo export probing, camera state |
| `test_auth.py` | 33 | Login, protected endpoints refuse without a token and accept with one, open endpoints stay open, logout, auth disabled mode, PBKDF2 hashing |
| `test_fl_schedule.py` | 24 | Interval and due-time arithmetic, persistence across restarts, the idle gate, status, round outcomes (failure does not consume the interval, success advances it, disabled FL does not run) |
| `test_robust_aggregation.py` | 21 | Structural, norm and direction defences, the consensus floor below 3 clients, aggregation outcome, `validate_candidate` |
| `test_face_recognition.py` | 20 | Deprecated face path: reset, enroll, recognise, remove, availability |
| `test_event_recorder.py` | 18 | Triggers, pre-roll, post-roll and cooldown, output files |
| `test_annotate.py` | 11 | Box drawing and label text |
| `test_api.py` | 11 | Health, camera, detection, automation and security endpoints respond |
| `test_dashboard.py` | 11 | `/dashboard` is served and contains what the page needs |
| `test_detection_service.py` | 11 | Initialisation, cascade fallbacks, status |
| `test_stream_annotated.py` | 9 | MJPEG framing, detection cadence, resilience to a failing detector |
| `test_federated_infra.py` | 6 | Round guards (unreachable aggregator fails cleanly), video ingest end to end, scheduler lifecycle |
| `test_face_api.py` | 4 | Face reset endpoint, detection status |
| `test_stream_endpoint.py` | 2 | The stream route |

Slowest tests (each 8 to 14 s): the three `TestModelGate` cases (app
startup), the video ingest, and the unreachable-aggregator timeout.

## 9.4 What is not tested

* Real hardware: Picamera2, the Hailo, the Jetson CSI path. Covered by
  stubs; verified by hand on the Pi.
* `central_server/fl_aggregator.py` has no unit tests. It is exercised by
  `scripts/fl_trial.sh` and the hardware runs.
* `sim/` has no unit tests. It is exercised by running it.
* Auth on the federated and dashboard routers, because there is none
  (section 6.4). A test that those routes return 401 with
  `AUTH_REQUIRED=true` should be written together with the fix.
* No coverage report is configured. Add `pytest-cov` and
  `--cov=src --cov-report=term-missing` to measure it.
