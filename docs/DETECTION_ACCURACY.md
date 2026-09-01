# Detection Accuracy

Measured numbers for every detection class, so claims about the system
can be checked rather than asserted.

Reproduce with:

```bash
python scripts/benchmark_classes.py --dataset <split>      # with ground truth
python scripts/benchmark_classes.py --images <dir>         # confidence only
python scripts/tune_threat_threshold.py --dataset <split> --negatives <dir>
```

---

## 0. If you only read one section

**The test split is 250 positive / 64 negative — 80% positive. A model
that fires on every single frame scores 0.796 plain accuracy without
looking at anything.**

Quote plain accuracy on this data and the first competent question you
get is "compared to what?". Use one of these instead:

| Metric | Value | Why this one |
|--------|------:|--------------|
| Face verification accuracy (LFW) | **0.9996** | Standard benchmark, real genuine + impostor pairs |
| Threat **precision** | **0.926** | When ARGUS flags someone, it is right >9 times in 10 |
| Threat **balanced accuracy** | **0.773** | 0.500 for the always-fire baseline; cannot be gamed |
| Threat plain accuracy | 0.787 | Only ever with the 0.796 baseline beside it |

**Nothing reaches 0.90 plain accuracy on threat detection.** The best of
48 configurations is 0.847, and it falsely flags 53.9% of ordinary
scenes. That is not a usable system, and the number should not be quoted.

---

## 1. The bar

Face recognition sets it. On clean stills, ArcFace scores **0.986** for
the same person against **0.073** for a different one — a separation of
0.91, with zero false accepts across five impostors. Nothing else in the
cascade is that clean, because nothing else is comparing a 512-d
embedding against an enrolled template.

The useful question for the object classes is not "does it match 0.99"
but **"does its confidence sit far enough from the threshold to stop
flickering between frames"**.

---

## 2. Per-class confidence

`near` is the share of detections within 0.10 of the decision threshold.
Those are the ones that flip on and off frame to frame, and they matter
more than the mean — a box that appears and vanishes reads as broken
even when the average confidence looks fine.

### On COCO128 (128 general-purpose photos)

| Class | n | mean | p05 | stdev | near | Verdict |
|-------|--:|-----:|----:|------:|-----:|---------|
| human | 136 | 0.772 | 0.521 | 0.135 | 18% | Stable |
| vehicle | 18 | 0.751 | 0.601 | 0.130 | **6%** | Most stable |
| animal | 11 | 0.794 | 0.568 | 0.128 | 18% | Stable, thin sample |
| unknown | 214 | 0.729 | 0.514 | 0.138 | 24% | Noise, see §4 |

### On the threat test set (314 labeled images, people-heavy)

| Class | n | mean | p05 | stdev | near |
|-------|--:|-----:|----:|------:|-----:|
| human | 370 | 0.763 | 0.542 | 0.114 | 12% |
| dangerous_person | 240 | 0.812 | 0.494 | 0.141 | **4%** |
| vehicle | 34 | 0.698 | 0.523 | 0.132 | 35% |
| animal | 7 | 0.612 | 0.513 | 0.087 | 43% |

The vehicle and animal rows in the second table look bad and should be
read with care: this dataset is street scenes of people, so those are
incidental background objects at the edge of frame. The COCO128 numbers
are the fair measure for those classes.

**MOTION** has no confidence in the usual sense — it reports the ratio of
contour area to bounding-box area, and it is a gate for the expensive
stages rather than a claim about the world.

**Conclusion:** HUMAN, VEHICLE, ANIMAL, and DANGEROUS_PERSON are all
stable enough for the demo. None reaches face recognition's separation,
and none needs to.

---

## 3. Threat stage against ground truth

Image-level: *did this frame contain an armed person*. Not directly
comparable to the upstream box-level mAP.

### Why it is one whole-frame pass, not one per person

The first implementation cropped to each COCO person box and classified
the crop. Measured over the same 314 images:

| Strategy | Precision | Recall | F1 | ms/frame | Missed |
|----------|----------:|-------:|---:|---------:|-------:|
| Per-person crop | 0.879 | 0.612 | 0.722 | 26.5 × N people | 97 |
| **Whole frame** | **0.901** | **0.800** | **0.847** | **35.6 flat** | **50** |
| Union of both | 0.881 | 0.828 | 0.854 | 62.0 | 43 |

Cropping lost 31% of recall. The breakdown of its 97 misses:

- **51 (20%)** — the COCO model never found the person, so the crop stage
  never ran on them. A structural loss, not a model limitation.
- **46 (18%)** — the crop ran and the model called it normal.

The threat model is itself a two-class *person detector*, so it locates
and classifies in a single pass. Making it wait for someone else's person
box threw away recall for nothing. Whole-frame also costs a **constant**,
where per-crop scales with crowd size and gets slower exactly when a
booth gets busy.

Union buys +0.028 recall for 1.7× the compute. Not worth it on a Pi.

---

## 4. Model and threshold: why YOLO11s at 0.70

Swept across 48 configurations — two trained variants, four input sizes,
six thresholds — against the 314 labelled images plus 128 ordinary COCO
photos as negatives:

```bash
python scripts/sweep_threat_config.py --dataset <split> --negatives <dir> \
    --weights models/threat-yolo11n.pt models/threat-yolo11s.pt
```

### The result

| Config | Bal. acc | Precision | Recall | False alarms | ms |
|--------|---------:|----------:|-------:|-------------:|---:|
| yolo11n @416 @0.55 *(previous)* | 0.751 | 0.919 | 0.768 | 18.8% | 102 |
| **yolo11s @416 @0.70** *(now)* | **0.773** | **0.926** | **0.796** | **14.8%** | **79** |

The larger variant wins on **every axis at once**, including a *lower*
false-alarm rate — normally the thing you trade away for recall.

Two things worth understanding rather than just copying:

**It wins at a higher threshold, not a lower one.** A more confident
model can be asked for more before it fires. Moving 11n from 0.55 to 0.70
costs recall; moving to 11s and *then* raising the threshold gains recall
and cuts false alarms together.

**Bigger input size is not free accuracy.** 832px reaches the highest
plain accuracy in the sweep (0.847) but flags **53.9%** of ordinary
scenes and costs 266 ms. 416px is both the fastest and the least
false-alarm-prone usable size. Do not raise it expecting a free win.

### Why not the best plain accuracy

The top plain-accuracy row is yolo11s @832 @0.30: accuracy 0.847, recall
0.900. Both above 0.90-adjacent territory, and both misleading:

- It falsely flags **53.9%** of ordinary scenes. At a booth, that is an
  accusation aimed at every second visitor.
- Its **balanced accuracy is 0.770** — *lower* than the shipped config's
  0.773. The plain-accuracy gain comes entirely from firing more often on
  a test set that is 80% positive.

This is what the §0 baseline warning is about, shown in one row.

### Tuning it yourself

```bash
THREAT_CONFIDENCE=0.55 python main.py     # more sensitive, more false alarms
THREAT_IMGSZ=832 python main.py           # slower; not more accurate in practice
```

Recall at the shipped config is **0.796** — about one armed person in
five is missed — and roughly **15 ordinary scenes in 100** raise a false
flag. This is a review aid, not a guard.

---

## 4b. Face verification on LFW

Labelled Faces in the Wild is the standard face-verification benchmark,
so this number is directly comparable to published results rather than
self-reported.

```bash
python scripts/benchmark_face_accuracy.py --faces <lfw-root> --max-identities 120
```

120 identities, 1,200 genuine pairs, 178,500 impostor pairs:

| Threshold | Accuracy | TAR | FAR |
|----------:|---------:|----:|----:|
| 0.30 | 0.9995 | 0.9633 | 0.0002 |
| 0.35 | 0.9996 | 0.9633 | 0.0002 |
| **0.40** *(shipped)* | **0.9996** | **0.9617** | **0.0001** |
| 0.50 | 0.9995 | 0.9392 | 0.0001 |
| 0.60 | 0.9985 | 0.7900 | 0.0001 |

Genuine mean 0.654, impostor mean 0.006 — a separation of **0.648** over
real-world images, versus the 0.913 measured earlier on clean studio
stills. The clean-still figure was optimistic; this is the honest one.

**The shipped threshold of 0.40 is already optimal.** No change needed —
which is worth stating, because a tuning pass that changes nothing is
still a result.

At 0.40 the false-accept rate is **1 in 10,000** (25 of 178,500). The
error that actually occurs is a false *reject*: 3.8% of the time the same
person is not matched, and the recovery for that is walking back into
frame.

---

## 5. The `unknown` class

214 detections across 128 COCO images, 24% of them near the threshold.

`_YOLO_CLASS_MAP` maps 11 COCO ids to ARGUS types; the other 69 fall
through to `UNKNOWN`. So backpacks, phones, chairs, and handbags all draw
green boxes labelled `unknown 0.63`.

That is noise on a demo screen and it competes for attention with the
boxes that matter. It is left in deliberately for now — filtering it is a
one-line change in `draw_detections`, but it is a product decision about
what the demo should show, not a bug to fix quietly.

---

## 6. Latency

x86 (Ryzen AI 7 350), median per image:

| Stage | ms |
|-------|---:|
| YOLOv8n COCO | ~90 |
| Threat, whole frame @416 | ~36 |
| Both | ~124 |

**A Pi 5 CPU is several times slower.** These are a ceiling, not a
prediction. Before the showcase, run on the Pi:

```bash
python scripts/benchmark_classes.py --images <a folder of stills>
```

If the frame rate does not hold, in order: raise `detect_every` on the
stream, lower `THREAT_IMGSZ` (but **not below 416** — 320 misses
entirely), then export to Hailo.
