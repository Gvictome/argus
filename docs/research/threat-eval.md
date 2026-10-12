# Threat stage: a larger test set (in progress)

**Goal:** re-tune the dangerous-person threshold on more than the 314-image
upstream split, then decide whether to retrain.

The threat model labels a whole scene's person as dangerous; it does not
box weapons. So weapon datasets are used as **image-level test data**, not
training data:

| Set | Source | Rule |
|---|---|---|
| Positives | [OD-WeaponDetection: Knife](https://github.com/ari-dasci/OD-WeaponDetection) (2,078 images, CC BY-SA 4.0), [Weapons in Images](https://datasetninja.com/weapons-in-images) (5,695 images, DbCL v1.0) | Weapon labelled **and** a person found by YOLO. A knife on a table is not an armed person |
| Negatives | COCO val2017 | Has a person, no knife, scissors or baseball bat |

## Status

- `scripts/build_threat_eval.py` is written, not yet run.
- The 2026-10-11 download (`dataset-tools`, about 1.8 GB) was stopped when the laptop ran low on memory. Download one dataset at a time.

## Run

```bash
pip install dataset-tools
python -c "import dataset_tools as d; d.download(dataset='OD-WeaponDetection: Knife Detection', dst_dir='data/datasets/weapons')"
python -c "import dataset_tools as d; d.download(dataset='Weapons in Images', dst_dir='data/datasets/weapons')"
python scripts/build_threat_eval.py data/datasets/weapons
python scripts/tune_threat_threshold.py --dataset data/datasets/threat_eval/positives --negatives data/datasets/threat_eval/negatives
THREAT_MODEL_PATH=models/threat-yolo11n.pt python scripts/tune_threat_threshold.py ...   # compare the variant
```

What to look for: recall on the new positives at the current 0.70 threshold,
and false alarms on ordinary scenes. If recall collapses on the new sets,
the model is overfit to the thesis split and retraining is the next step.
Then measure the Pi frame-rate cost with `./scripts/run_node.sh --threat`.
