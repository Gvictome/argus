# 4 · Classification: the threat model

ARGUS classifies at two levels:

1. **What is it?** YOLOv8n on COCO, folded into person / vehicle /
   animal / package. Covered in section 3.
2. **Is this person dangerous?** A two-class YOLO11 model, covered here.
3. **Is this routine for this site?** The federated head, covered in
   section 5.

---

## 4.1 Where the model comes from

| | |
|---|---|
| Source | `Nambekai/dangerous-person-detection-yolo11`, a graduation-thesis model by Tran Si Nam |
| Licence | AGPL-3.0. ARGUS is AGPL-3.0 as a result; see `NOTICE.md` |
| Classes | index 0 `nguoi_binh_thuong` (normal person), index 1 `doi_tuong_nguy_hiem` (dangerous person) |
| Variants | `yolo11n`, `yolo11s` (used), `yolo11s_preprocessing`, `yolo11m`, `yolo11l` |
| Weights | Not in git. `python scripts/fetch_models.py [--variant yolo11s]` downloads the upstream release into `models/` and verifies the published SHA-256 |
| Default path | `models/threat-yolo11s.pt` (`THREAT_MODEL_PATH`) |

The model is itself a person detector with two classes. It does not need
COCO to find people first.

---

## 4.2 `src/detection/threat.py`

### `ThreatClassifier(model_path, confidence=0.35, imgsz=416, crop_padding=0.15)`

The app constructs it with `THREAT_CONFIDENCE` (0.70) and `THREAT_IMGSZ`
(416) from `Settings`, then attaches it with
`detection_service.attach_threat_classifier()`. It raises a clear
`RuntimeError` if `ultralytics` or the weights are missing; the app catches
that and runs without the stage.

**Classes are pinned by index, never by name.** The upstream
documentation calls them `normal_person` / `potentially_dangerous_person`,
but the checkpoint ships Vietnamese names. Matching on the English strings
silently detects nothing. At load time the class names are compared with
`KNOWN_CLASS_NAMES` and a mismatch is logged at WARNING (INFO never reaches
the demo unit's log, because uvicorn runs at warning level unless `DEBUG`).

### Methods

| Method | Used by | What it does |
|---|---|---|
| `detect_threats(frame)` | **The live cascade** | One inference over the whole frame; returns a `ThreatResult` with its own box for every class-1 detection |
| `classify_region(frame, bbox)` | Tests, tools | Pads the box by 15% of its longer side, crops, classifies |
| `classify_frame(frame)` | Tests, tools | Classifies the whole frame and reduces it to one verdict |
| `_classify(image)` | the two above | Any dangerous box beats every normal box, whatever the confidences |

### Why whole-frame, not per-person crops

Measured on the 314-image upstream test split:

| Strategy | Precision | Recall | F1 | Cost |
|---|---|---|---|---|
| Crop each COCO person | 0.879 | 0.612 | 0.722 | 26.5 ms × number of people |
| **Whole frame** | **0.901** | **0.800** | **0.847** | 35.6 ms flat |

Per-crop lost 20% of true positives outright, because the COCO detector
never found those people. It also gets slower exactly when a scene gets
busy.

### Why padding matters, when cropping is used

The model was trained on people in full scenes. A crop tight to the person
box cuts off the arm and the weapon:

| Crop | Result for the armed person in the upstream sample |
|---|---|
| Tight | no detection |
| **15% padding** | dangerous, 0.699 |
| 30% padding | dangerous, 0.533 |
| 50% padding | dangerous, 0.436 |

### Why "any dangerous box wins"

On the upstream sample frame the bystander scores 0.86 normal and the armed
person 0.62 dangerous. A plain argmax reports "nothing to see". Confidence
therefore means "how sure we are about what we are reporting".

---

## 4.3 How the operating point was chosen

`scripts/sweep_threat_config.py` swept 48 configurations (2 variants × 4
input sizes × 6 thresholds) against the 314-image labelled test split plus
128 ordinary COCO images as negatives.

**Read the baseline first.** The test split is 250 positive / 64 negative,
so a model that fires on every frame scores 0.796 plain accuracy. Plain
accuracy is close to meaningless here; compare balanced accuracy, which is
0.500 for that always-fire baseline.

| Configuration | Balanced accuracy | Precision | Recall | False alarms on ordinary scenes |
|---|---|---|---|---|
| yolo11n @ 416 @ 0.55 (old) | 0.751 | 0.919 | 0.768 | 18.8% |
| **yolo11s @ 416 @ 0.70 (current)** | **0.773** | **0.926** | **0.796** | **14.8%** |

The larger variant wins on every axis at once, including a lower
false-alarm rate, and at a higher threshold. Nothing in the sweep reached
0.90 plain accuracy (best 0.847, which falsely flagged 53.9% of ordinary
scenes). The defensible headline is precision: **when ARGUS flags someone,
it is right more than nine times in ten.**

The tradeoff the threshold controls:

* A false negative misses an armed person, the failure the system exists
  to prevent.
* A false positive flags an innocent visitor as dangerous, in public, at a
  booth. That is a credibility and an ethical problem.

`scripts/tune_threat_threshold.py` prints precision, recall, F1 and both
error counts at each threshold so the choice is made on evidence and
written down.

---

## 4.4 Cost, and when to turn it on

Measured on x86 (Ryzen AI 7 350), YOLO11n, whole frame
(`docs/THREAT_DETECTION.md`):

| Input size | ms / frame | Note |
|---|---|---|
| 320 | about 16 | misses entirely, do not use |
| **416** | **about 36** | the default: fastest usable, and most confident |
| 512 | 28 to 45 | |
| 832 (trained size) | 46 and up | no accuracy gain to justify it |

On the Pi 5 CPU, `scripts/run_node.sh` budgets about 100 ms per frame.

That is why `scripts/run_node.sh` defaults it **off** and `--threat` turns
it on. Once the detectors run on the Hailo, this constraint mostly goes
away (`sim/run_node.py --profile pi5-threat-on-cpu` models the risky case
where the threat model stays on the CPU).

---

## 4.5 Tools for this stage

| Script | Use |
|---|---|
| `scripts/fetch_models.py` | Download and checksum the weights. `--list` shows variants |
| `scripts/sweep_threat_config.py --dataset <split> [--negatives <dir>] [--weights a.pt b.pt]` | Size × threshold sweep, optionally across variants |
| `scripts/tune_threat_threshold.py --dataset <split> [--negatives <dir>]` | Threshold-only sweep with both error counts |
| `scripts/benchmark_classes.py --images <dir>` or `--dataset <split>` | Per-class confidence spread for the whole cascade; with a dataset, true precision and recall for the threat stage |

---

## 4.6 The tests

`tests/test_threat.py` (41 tests) covers class pinning by index, the
mismatch warning, padding geometry at frame edges, "dangerous beats
normal", whole-frame detection returning per-person boxes, the cascade
running threats without a COCO person, a threat-stage exception not
killing object detection, overlay colour, and the threat box superseding
the overlapping person box. None of them load the real weights; the model
is replaced by a stub, so they run without `models/`.
