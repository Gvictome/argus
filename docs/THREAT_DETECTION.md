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
| **Threat** | **YOLO11n, 2 classes** | **Motion found** |

```
motion (frame diff)
  ├─ yolov8n COCO  →  MOTION / HUMAN / ANIMAL / VEHICLE
  │    └─ if HUMAN: ArcFace identity
  └─ YOLO11n threat  →  DANGEROUS_PERSON   (whole frame, own boxes)
```

One whole-frame inference, **not** one per person, and deliberately not
gated behind a COCO human box. The threat model is itself a two-class
person detector, so it locates and classifies in a single pass.

Gating it on a COCO detection cost 20% of true positives outright --
recall 0.612 against 0.800 -- because the object model never found those
people at all. Full numbers in
[`DETECTION_ACCURACY.md`](DETECTION_ACCURACY.md).

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
| `THREAT_CONFIDENCE` | `0.55` | Tuned; see DETECTION_ACCURACY.md |
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

This is why the cascade does not crop. `classify_region()` remains for
callers wanting a verdict on one specific person and pads by
`CROP_PADDING = 0.15`; the cascade uses `detect_threats()`, which
measured better on every axis. See `DETECTION_ACCURACY.md` section 3.

Padding has its own tradeoff, which is the other reason the cascade
avoids it: a padded crop can pull a neighbour in, so someone standing
beside an armed person can inherit the flag.

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

Measured on x86 (Ryzen AI 7 350), YOLO11n, whole frame:

| `imgsz` | ms/frame | Note |
|--------:|---------:|------|
| 320 | ~16 | **misses entirely, do not use** |
| **416** | **~36** | default: fastest usable, and most confident |
| 512 | ~28-45 | |
| 832 (trained) | ~46+ | no accuracy gain here to justify it |

The cost is a **constant per frame** now, not per person, so a crowded
booth no longer slows detection down.

**A Pi 5 CPU is materially slower than this.** Benchmark before assuming
the demo's frame rate survives:

```bash
python scripts/benchmark_classes.py --images <folder of stills>
```

If it does not hold, in order: raise `detect_every` on the stream, drop
`THREAT_IMGSZ` (never below 416), then export to Hailo.

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

Measured on ARGUS at the shipped threshold, recall is **0.768** — about
one armed person in four is missed — and roughly **19 of every 100
ordinary scenes** flag someone who is not armed. False positives cluster
on small, occluded, or poorly lit objects: books, bottles, phones,
umbrellas. Those hard negatives are in the training data precisely
because they are confusable.

This is a review aid, not a guard. Say so before a judge asks.

A detection should only mark a frame for human review. If a judge asks
about accuracy, quote the recall; it is a more credible answer than a
demo that appears to work perfectly.
