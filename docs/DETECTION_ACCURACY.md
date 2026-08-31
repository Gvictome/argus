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

## 4. Threshold: why 0.55, not the upstream 0.35

Swept against the 314 labeled images, plus 128 ordinary COCO photos as
negatives — an approximation of "people walking past a booth".

| Threshold | Precision | Recall | F1 | False alarms per 100 ordinary scenes |
|----------:|----------:|-------:|---:|--------------------:|
| 0.30 | 0.898 | 0.808 | **0.851** | 30.5 |
| 0.35 (upstream) | 0.901 | 0.800 | 0.847 | 29.7 |
| **0.55 (ours)** | **0.919** | 0.768 | 0.837 | **18.8** |
| 0.70 | 0.931 | 0.700 | 0.799 | 10.2 |

F1 peaks near 0.30. **F1 is the wrong objective here.**

At 0.35, nearly one ordinary scene in three flags somebody as armed.
Pointed at a visitor walking past the booth, that is both a credibility
problem and an ethical one — the system would be publicly accusing
innocent people roughly every third frame.

0.55 gives up 0.03 recall to cut false alarms by 37%. Raise toward 0.70
for a crowded room. Lower only with a deliberate argument that a missed
detection costs more than a false accusation.

Override without editing code:

```bash
THREAT_CONFIDENCE=0.70 python main.py
```

### What 0.55 looks like on real frames

The upstream sample images, whole-frame at 416px:

| Sample | Score | At 0.55 |
|--------|------:|---------|
| `street-scene.jpg` — clear, close, unobstructed | 0.925 | **flags** |
| `handgun-scene.jpg` — subject bent over, partly occluded | 0.436 | below |
| `public-area-scene.jpg` — distant, small in frame | 0.383 | below |

This is the tradeoff working as designed, and it predicts live behavior
usefully: **a person holding something clearly, facing the camera at
booth distance, scores high.** The marginal cases are the occluded and
distant ones — which are also the ones most likely to be a false alarm.

If you want those marginal frames to flag during a rehearsal, drop the
threshold for that session and put it back:

```bash
THREAT_CONFIDENCE=0.35 python main.py
```

### The honest caveat

Recall at 0.55 is **0.768**, so roughly one armed person in four is
missed, and about 19 ordinary scenes in 100 flag someone who is not
armed. This is a review aid, not a guard. Say that out loud before a
judge asks.

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
