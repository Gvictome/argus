# Notices and Attribution

ARGUS incorporates third-party models, datasets, and libraries. This file
records what they are, where they came from, and what their licenses
require.

---

## 1. Threat-detection model (YOLO11)

The dangerous-person classification stage
(`src/detection/threat.py`) runs weights we did not train.

| Field | Value |
|-------|-------|
| Work | YOLO11 Dangerous-Person Detection in Public Areas, v1.0.0 |
| Author | Tran Si Nam |
| Source | https://github.com/Nambekai/dangerous-person-detection-yolo11 |
| Release | `v1.0.0` — "Thesis Artifact Release" |
| License | **AGPL-3.0** |
| Origin | Graduation thesis artifact |

The weights are **not vendored into this repository.** They are fetched at
setup time by `scripts/fetch_models.py`, which verifies each download
against the SHA-256 checksums the upstream release publishes.

### Changes we made

CC BY 4.0 and good practice both ask that modifications be stated. We did
not retrain, fine-tune, or alter the weights in any way. We changed only
how they are *invoked*:

- Inference runs at **416px**, not the 832px used for training. Measured
  on x86, 832px costs 45.6ms per person against 20.1ms at 416px, and
  416px also scored *higher* on the upstream sample (0.699 vs 0.664).
- Classification runs on a **padded crop of a single person** located by
  our own COCO detector, rather than on the whole frame.
- Class identity is resolved **by index** (0 normal, 1 dangerous), because
  the published checkpoint ships Vietnamese class names
  (`nguoi_binh_thuong`, `doi_tuong_nguy_hiem`) rather than the English
  names its `config/data.yaml` documents.

### Reported performance

From the upstream model card, measured by its author on their own test
set — not reproduced by us, and not measured on ARGUS hardware:

| Variant | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---------|----------:|-------:|---:|------:|---------:|
| YOLO11n | 0.81 | 0.69 | 0.74 | 0.78 | 0.60 |
| YOLO11s | 0.87 | 0.69 | 0.77 | 0.80 | 0.64 |
| YOLO11l | 0.80 | 0.72 | 0.76 | 0.79 | 0.63 |

ARGUS ships **YOLO11n** by default.

---

## 2. Training dataset

We do not redistribute this dataset, and ARGUS does not require it at
runtime. It is recorded here because it is what the model above learned
from, and its terms are separate from the model's.

| Field | Value |
|-------|-------|
| Recorded source | Roboflow project `Do an` — https://universe.roboflow.com/062transinam-s-workspace/do-an-ahpc0 |
| License | **CC BY 4.0** |
| Size | 3,140 images, 10,595 annotated instances |
| Classes | `normal_person`, `potentially_dangerous_person` |
| Recorded export | 2026-04-22 |

Known limitations, taken from the upstream dataset card:

- The class distribution is imbalanced — 72.5% normal, 27.5% dangerous.
- Public-source imagery carries geographic, camera, lighting, and
  demographic sampling biases.
- A single frame cannot establish intent, and cannot reliably separate
  every harmless object from a weapon.

---

## 3. Ultralytics

`ultralytics` (YOLOv8 and YOLO11) is licensed **AGPL-3.0**. Both the COCO
object detector and the threat classifier run through it.

Commercial or closed-source distribution of a work built on Ultralytics
requires a separate Ultralytics enterprise license.

---

## 4. InsightFace / ArcFace

Face recognition (`src/detection/face_recognition.py`) uses `insightface`
with the `buffalo_l` model pack, downloaded at first run to
`~/.insightface/models/`. InsightFace's models are published for
**non-commercial research purposes**.

---

## 5. Why ARGUS is AGPL-3.0

Sections 1, 3, and 4 above are the reason.

AGPL-3.0 extends copyleft to network use: running AGPL software as a
network service counts as conveying it, and obliges you to offer the
complete corresponding source to its users. ARGUS is a FastAPI server
that streams video over a network and links Ultralytics in-process, so
that clause applies to the combined work.

This project therefore adopts **AGPL-3.0**. See `LICENSE`.

An earlier `README.md` line described the project as MIT. That predates
the model integration, was never backed by a LICENSE file, and is
superseded by this notice.

### If ARGUS is ever commercialized

The AGPL weights cannot ship in a closed product. The path is to keep the
`ThreatClassifier` interface and retrain on the CC BY 4.0 dataset — which
only requires attribution — from a permissively licensed base. The class
taxonomy and `src/detection/threat.py` would carry over unchanged.

---

## 6. Responsible use

The upstream dataset card names uses its authors consider inappropriate.
Two of them bear directly on ARGUS and are worth stating plainly:

- **This is not biometric threat assessment.** ARGUS runs face
  recognition and threat detection in the same pipeline, but they are
  independent stages. A `dangerous_person` result describes an object in
  the frame; it is never combined with, or attributed to, an enrolled
  identity.
- **A detection is not a conclusion.** `potentially_dangerous_person`
  means a person appeared alongside a weapon-like object in one frame. It
  is not a statement about intent, guilt, or threat, and at a recall of
  0.69 roughly a third of true instances are missed. The upstream model
  card's framing is the right one: a detection should only mark a frame
  for human review.
