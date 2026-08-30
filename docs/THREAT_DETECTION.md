# Threat Detection

Dangerous-person classification for the ARGUS detection cascade.

Flags a person in frame who appears to be carrying a firearm or knife.
This is an **additive** stage: the existing COCO object detection and
ArcFace face recognition are unchanged, and the whole stage can be turned
off with one environment variable.

---

## 1. How it works

| Stage | Component | Runs when |
|-------|-----------|-----------|
| Motion | Frame differencing | Every frame |
| Objects | YOLOv8n, 80 COCO classes | Motion found |
| Faces | ArcFace / `buffalo_l` | A human was found |
| **Threat** | **YOLO11n, 2 classes** | **A human was found** |

```
motion (frame diff)
  └─ yolov8n COCO  →  MOTION / HUMAN / ANIMAL / VEHICLE
       └─ if HUMAN:
            ├─ ArcFace identity
            └─ YOLO11n threat, on each person crop
```

Threat classification runs **per person**, not per frame, so its cost
scales with how many people are present rather than with frame rate.
Implementation is `src/detection/threat.py`; it is wired into
`DetectionService` at startup by `src/api/app.py`.

The model is not ours. See [`NOTICE.md`](../NOTICE.md) for provenance,
license, and the changes we made.

---

## 2. Setup

```bash
python scripts/fetch_models.py
```

Downloads YOLO11n from the upstream release, verifies it against the
published SHA-256, and writes `models/threat-yolo11n.pt`. Weights are
gitignored — they are fetched, never committed.

```bash
python scripts/fetch_models.py --list          # other variants
python scripts/fetch_models.py --variant yolo11s
```

**Do this once with network access before the showcase.** The demo unit
runs offline.

Requires `ultralytics`, which the minimal install in
`docs/SETUP_TESTING.md` deliberately skips. Without it — or without the
weights — the API still starts, logs `Threat detection unavailable: ...`,
and reports `"threat_detection": false`. Everything else runs normally.

---

## 3. Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `THREAT_ENABLED` | `true` | Master switch |
| `THREAT_MODEL_PATH` | `models/threat-yolo11n.pt` | Weights |
| `THREAT_CONFIDENCE` | `0.35` | Minimum detection confidence |
| `THREAT_IMGSZ` | `416` | Inference input size |

`THREAT_ENABLED=false` returns the pipeline to exactly its previous
behavior with no code change. That is the intended escape hatch if the
stage misbehaves near a demo.

---

## 4. Two findings that are easy to regress

Both were found by running the real weights, and both fail *silently* —
the system detects nothing and looks like a broken camera rather than
raising an error. Each has a regression test in `tests/test_threat.py`.

### The class names are Vietnamese

The upstream `config/data.yaml` documents `normal_person` and
`potentially_dangerous_person`. The shipped checkpoint actually contains:

```python
{0: 'nguoi_binh_thuong', 1: 'doi_tuong_nguy_hiem'}
```

Class identity is therefore resolved **by index**, never by name.
`ThreatClassifier` compares the loaded checkpoint's names against
`KNOWN_CLASS_NAMES` and **warns** if they differ, because a model whose
indices mean something else inverts every verdict.

The warning level is deliberate: `main.py` runs uvicorn at
`log_level="warning"` unless `DEBUG` is set, so an informational line
would never reach the operator on the demo unit.

### A tight crop detects nothing

The model was trained on people in full scenes, so a crop tight to the
COCO person box cuts off the arm and the weapon. Measured on the upstream
handgun sample, for the person actually holding the gun:

| Crop | Result |
|------|--------|
| Tight person box | **no detection at all** |
| +15% padding | dangerous, 0.699 |
| +30% padding | dangerous, 0.533 |
| +50% padding | dangerous, 0.436 |

`CROP_PADDING = 0.15` in `src/detection/threat.py`. It also beats
classifying the whole frame (0.436 at 416px), because the person fills
more of the input.

The tradeoff: padding can pull a neighbour into the crop, so someone
standing beside an armed person can inherit the flag. 15% keeps that
narrow while still restoring the context the model needs.

---

## 5. Any dangerous detection outranks every normal one

On the upstream handgun sample the unarmed bystander scores 0.86 and the
armed person 0.62. Reducing the boxes with a plain `argmax` over
confidence reports the bystander, and the threat vanishes.

So `_classify()` prefers *any* dangerous detection over every normal one,
and the reported confidence describes the thing being reported rather than
the most obvious person in frame.

---

## 6. Performance

Measured on x86 (Ryzen AI 7 350), one person crop, YOLO11n:

| `imgsz` | ms/person | Detection |
|--------:|----------:|-----------|
| 320 | 15.8 | **missed** |
| **416** | **20.1** | **dangerous 0.699** |
| 512 | 27.5 | dangerous 0.515 |
| 640 | 32.4 | dangerous 0.596 |
| 832 (trained) | 45.6 | dangerous 0.664 |

416 is both the fastest usable size and the most confident, which is why
it is the default. 320 misses entirely — do not go below 416.

**A Pi 5 CPU is materially slower than this**, and the per-frame cost is
this number times the number of people in frame. Two people at 416px is
~40ms of threat work on top of YOLOv8n and ArcFace. Benchmark on the Pi
before assuming the demo's frame rate survives; the Hailo export is the
lever if it does not.

---

## 7. Status endpoint

```bash
curl http://<pi>:8000/api/detection/status
```

```json
{"status":"running","fps":12.4,"backend":"cpu","motion_detection":true,
 "object_detection":true,"face_recognition":true,"threat_detection":true,
 "known_faces":3}
```

`"threat_detection": false` means the model or `ultralytics` failed to
load. Check startup logs for `Threat detection unavailable`.

---

## 8. Reading the stream

A flagged person draws an **orange** box labelled `dangerous`, replacing
the green `person` box for that same detection — both would otherwise
stack two labels on one pixel and imply two subjects.

| Color | Meaning |
|-------|---------|
| Orange | Person flagged as carrying a weapon |
| Green | Recognized person, or any ordinary object |
| Red | A face with no identity match |

---

## 9. Tests

```bash
python -m pytest tests/test_threat.py -q
```

Ultralytics is faked at the module boundary, so no weights or ONNX
runtime are needed. Covers class-index mapping, crop padding,
dangerous-outranks-normal, graceful degradation when the stage raises,
the status field, and the overlay rules.

---

## 10. What this is not

`potentially_dangerous_person` means a person appeared alongside a
weapon-like object in a single frame. It is not a claim about intent,
guilt, or threat.

Upstream recall is **0.69** — roughly a third of true instances are
missed — and false positives occur on small, occluded, or poorly lit
objects, including books, bottles, phones, and umbrellas. Those hard
negatives are in the training data precisely because they are confusable.

A detection should only mark a frame for human review. If a judge asks
about accuracy, quote the recall; it is a more credible answer than a
demo that appears to work perfectly.
