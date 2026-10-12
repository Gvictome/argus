# Datasets: replacing CIFAR-10

**Decision (2026-10-12):** the federated head's benchmark moves from CIFAR-10
to **MEVA**, a CC-BY security-camera dataset with 29 cameras at one site.
Each camera becomes one federated client. The vehicle-make classifier keeps
Stanford Cars for now, fine-tuned on MEVA's own vehicle crops so it sees the
downward security-camera angle. **ShanghaiTech Campus** is the anomaly
test set. CIFAR-10 survives only as an optional comparison with published FL
results, Dirichlet-partitioned, never IID.

## Why CIFAR-10 has to go

| Problem | Effect |
|---|---|
| Natural images of 10 everyday classes, not security footage | Says nothing about ARGUS's task, deciding whether a detection is routine at a site |
| Split evenly at random across nodes (`IidPartitioner`) | The one case where federation has nothing to overcome |
| Classifies pixels | The head never sees pixels; it reads 19 numbers per tracked event (`src/federated/features.py`) |

The 88.86% from April proves the Flower plumbing works. It is not evidence about ARGUS.

## The combination

| Role | Dataset | Why this one | License | Get it |
|---|---|---|---|---|
| **FL benchmark: train and federate the head** | [MEVA](https://mevadata.org) | Real security cameras, 29 cameras = 29 non-IID clients, people and vehicles with tracked boxes, about 22 h of public annotations plus 328 h of video | CC-BY-4.0 | `aws s3 ls --no-sign-request s3://mevadata-public-01/` ([AWS registry](https://registry.opendata.aws/mevadata/)) |
| **Anomaly test set** | [ShanghaiTech Campus](https://svip-lab.github.io/dataset/campus_dataset.html) | 13 scenes, 437 videos, 130 abnormal events with pixel-level ground truth: bikes, skateboards and vehicles on walkways, fighting, chasing | No license stated; treat as research use | OneDrive link on the project page |
| **Vehicle make (crop classifier)** | [Stanford Cars](https://huggingface.co/datasets/Donghyun99/Stanford-Cars) + MEVA vehicle crops | Stanford gives make labels; MEVA crops give the camera angle Stanford lacks | Stanford: research only | Hugging Face |
| Make, commercial-safe later | [VMMRdb](https://github.com/faezetta/VMMRdb) | Replaces Stanford Cars before any node is sold | MIT | GitHub |
| Threat stage evaluation (separate track) | [OD-WeaponDetection: Knife](https://github.com/ari-dasci/OD-WeaponDetection), [Weapons in Images](https://datasetninja.com/weapons-in-images) | Weapon + person images as positives for `tune_threat_threshold.py` | CC BY-SA 4.0, DbCL | `dataset-tools`; see `threat-eval.md` |
| Literature comparison only | CIFAR-10, Dirichlet α=0.5 | Comparable with published FL robustness results | Public | Flower Datasets |

**Not chosen:**
- **VIRAT:** its ground footage needs a signed protection agreement, and MEVA (same Kitware lineage) is CC-BY.
- **UCF-Crime:** research-only, 96 GB, labelled per video rather than per frame. Possible later for the threat stage.

## How MEVA becomes head training data

The head's input is a tracked event, not an image. MEVA already ships the
tracks, so detection can be skipped for the benchmark:

1. **Tracks to events.** For each actor or vehicle track in a MEVA annotation file, build the same 19 numbers `features.extract()` builds from a live track: class one-hot, box geometry, velocity, dwell, hour and weekday (from the clip timestamp), confidence = 1.0. Reuse `features.py` rather than re-implementing it, or the benchmark drifts from the node.
2. **Clients.** One client per camera ID. That is the non-IID split for free: a parking-lot camera and a doorway camera see different class mixes and different normal.
3. **Labels.** MEVA labels activities, not "routine". Define anomaly per camera from the data, then write the rule down before running anything:
    - an object class or activity that is rare at that camera, below a frequency you fix in advance;
    - or presence outside that camera's active hours, the same rule the synthetic workload uses.
4. **Holdout.** Hold out whole clips, never random events from the same clip. Neighbouring events from one clip are near-duplicates and leak.
5. **Second pass, after the benchmark works:** run the clips through ARGUS's own YOLO (`scripts/ingest_video.py`) instead of the ground-truth tracks. The gap between the two runs shows how much detector error costs the head.

ShanghaiTech is test-only: its labelled abnormal events score whether the head flags the abnormal events in scenes it was not trained on.

## Running it

You build the converter; this is the shape of it.

| Step | Where | Check |
|---|---|---|
| Download one camera's clips + annotations first | `data/datasets/meva/` (gitignored) | Annotation track count matches the video |
| `scripts/meva_to_events.py` → one `samples.jsonl` per camera | Reuse `src/federated/store.py` `SampleStore` | Class mix differs per camera; anomaly rate 2-10% |
| Point `sim/fl.py client` at a camera's store | New `--store` flag next to `--site` | Same 19-feature vectors the node produces |
| 3 rounds × 3 epochs, 5 to 10 cameras | `sim/fl.py serve --clients N --aggregation robust` | Balanced accuracy and per-class recall per round |
| Same rounds with `--aggregation fedavg`, and each client local-only | Same holdout | The three-arm comparison |

Report balanced accuracy and per-class recall with confidence intervals. Never quote plain accuracy: person dominates every security dataset.

## Where the real Pi data fits

The 322 labelled events from the Pi are the only data from ARGUS's own camera. They are one more client, the most important one. MEVA trains the shared model; the Pi events test whether it transfers to a camera it never saw. That is the claim the showcase needs.
