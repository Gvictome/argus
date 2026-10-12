# Fine taxonomy simulation (2026-10-03)

**Result:** adding vehicle make, carrier and package size to the head (19 → 35
features) raised FedAvg balanced accuracy from 0.860 to 0.894, using a make
classifier actually trained on Stanford Cars (0.832 test accuracy, 49 makes).
**Local-only training beat FedAvg in every run**, by 0.04 to 0.06, so the
comparison now includes FedAvg followed by per-site fine-tuning.

Code: branch `feat/fine-taxonomy` (not yet on GitHub; on the second dev
machine). Full write-up: `Dropbox/argus/argus/ARGUS_Fine_Taxonomy_2026-10-03_v2.pdf`.
All site behaviour is simulated; see [datasets.md](datasets.md) for the move to MEVA.

## What trains on what

| Model | Trained on | Proves |
|---|---|---|
| Crop classifier (MobileNetV3-Large, make) | [Stanford Cars](https://huggingface.co/datasets/Donghyun99/Stanford-Cars), 16,185 photos → 49 makes | The make labels are learnable |
| Federated head (35 features) | Synthetic events, `sim/workload_fine.py`, 4 sites | Federation on ARGUS-shaped data |

The head never sees an image. The classifier's measured accuracy enters the
simulation as `--classifier-acc`: the simulator reports the true subtype with
that probability, otherwise a wrong one.

## Results

Balanced accuracy, 6,000 events per site, 6 rounds × 2 epochs, seed 1:

| Classifier acc | FedAvg 19 feat | FedAvg 35 feat | Local 19 feat | Local 35 feat |
|---|---|---|---|---|
| 1.00 | 0.860 | 0.908 | 0.921 | 0.962 |
| 0.92 (trained, all slots) | 0.860 | 0.909 | 0.921 | 0.952 |
| 0.90 | 0.860 | 0.913 | 0.921 | 0.949 |
| 0.76 (trained, tracked makes) | 0.860 | 0.894 | 0.921 | 0.933 |
| 0.75 | 0.860 | 0.893 | 0.921 | 0.937 |
| 0.50 | 0.860 | 0.871 | 0.921 | 0.919 |

Below about 75% classifier accuracy the gain mostly disappears. Full Flower
run, 4 clients over gRPC, 5 rounds: balanced accuracy 0.849 → 0.864.

Why FedAvg loses: the driveway household drives a Nissan and a Toyota, the
curbside one a Honda and a Ford. One shared model cannot hold both rules;
household cars at night score 0.105 under FedAvg and 1.000 locally.

Frame rate with the extra crop stage: 13.33 FPS while training (target 12),
from budgeted stage times. Measured, one crop takes 15.8 ms on the laptop CPU,
so on the Pi it must run on the Hailo.

## Run it yourself

From the `feat/fine-taxonomy` checkout:

```bash
# 1. The make classifier on Stanford Cars (the "is it trainable" proof)
python training/crop/prepare_stanford_cars.py --out D:/argus-data/crops/make
python training/crop/train.py --help          # find the epochs flag; the study used 12
python training/crop/train.py --task make --data D:/argus-data/crops/make --out D:/argus-data/models/crop_make
# read metrics.json: val accuracy per epoch, test accuracy, confusion matrix

# 2. Three FL rounds with that accuracy plugged in
python sim/run_node.py --taxonomy fine --clients 4 --rounds 3 --epochs 3 --samples 8000 --classifier-acc <test acc>
python sim/compare_taxonomy.py --acc <test acc> --rounds 3   # coarse vs fine, FedAvg vs local
```

Expect about 0.5 val accuracy after 3 classifier epochs (the study's epoch 3
was 0.530).

## Limits

- Delivery windows, household cars and per-site mixes are assumptions.
- Stanford Cars is eye-level; ARGUS looks down. Fine-tune on MEVA vehicle crops ([datasets.md](datasets.md)).
- Stanford Cars is research-only; retrain on [VMMRdb](https://github.com/faezetta/VMMRdb) (MIT) before any sale.
- The live head stays at 19 features until after the showcase; the plan is shadow mode (subtype logged, not used) by the 11/08 freeze.
