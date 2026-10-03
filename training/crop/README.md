# ARGUS crop classifier: datasets and training

The second detection stage. YOLO finds a vehicle or package; this stage
looks inside the box and reports **carrier** (UPS, USPS, FedEx, Amazon,
none), **vehicle make** and **package size** (mail, small, large). Those
answers fill the fine-taxonomy slots the simulator studies
(`sim/workload_fine.py`, `sim/compare_taxonomy.py`).

One small classifier per task, MobileNetV3-Large from ImageNet weights.

## Datasets

| Task | Dataset | Size | Labels | License | Status |
|---|---|---|---|---|---|
| make | [Stanford Cars](https://huggingface.co/datasets/Donghyun99/Stanford-Cars) (HF parquet mirror) | 8,144 train / 8,041 test | 196 make-model-year, collapsed to 49 makes | Not stated on the mirror; original Stanford release is research use. Treat as non-commercial. | Downloaded to `D:\argus-data\stanford-cars`, prepared, trained |
| make (bigger, optional) | [VMMRdb](https://github.com/faezetta/VMMRdb) | 291,752 images | 9,170 make-model-year classes, 1950-2016 | MIT (repo); cite Tafazzoli et al., CVPRW 2017 | Not downloaded |
| carrier | [USPS Merge](https://universe.roboflow.com/david-rimer-3ch55/usps-merge) (Roboflow Universe) | not verified | Amazon, FedEx, Other-Vehicles, UPS, USPS-Truck | not verified (Roboflow blocks automated reads) | Needs a Roboflow login to download |
| carrier (extra) | [Delivery Trucks](https://universe.roboflow.com/on-time-delivery/delivery-trucks) (Roboflow Universe) | not verified | delivery truck classes | not verified | Needs a Roboflow login |
| package size | [Packages](https://public.roboflow.com/object-detection/packages-dataset) (Roboflow public) | 26 raw / 250 augmented | one class "packages" (envelopes, small and large boxes, unlabelled by size) | Public domain | Too small and has no size labels; useful only as doorstep test images |
| package size | [mailers and boxes 2](https://universe.roboflow.com/lukmik-e4qay/mailers-and-boxes-2), [Envelope](https://universe.roboflow.com/parceldetection/envelope), [Parcels and Boxes](https://universe.roboflow.com/abl-okoo2/parcels-and-boxes-dv7sg) | not verified | mailer / box / envelope | not verified | Needs a Roboflow login |

**Gaps no public dataset fills:**
- **Large vs small parcel.** Public sets label "box", not size. Either
  relabel a few hundred crops by hand, or derive size from the box's real
  extent once camera height is known.
- **Carrier on a package** (a UPS label on a box at the door). No dataset
  found. The `carrier` model is trained on vehicles; package carrier stays
  "none" until the team labels its own doorstep footage.
- **Camera angle.** Stanford Cars is mostly side and 3/4 views at eye level;
  a doorbell or driveway camera looks down. Expect lower accuracy on real
  footage until ARGUS's own clips are added (Phase 4 plan).

## Pipeline

```bash
# 1. make: Stanford Cars -> 49 make folders (done once, ~2 min)
python training/crop/prepare_stanford_cars.py --out D:/argus-data/crops/make

# 2. carrier / size: download each Roboflow dataset in COCO format, unzip under
#    D:/argus-data/roboflow/<name>/, then crop per ARGUS slot
python training/crop/prepare_roboflow.py --task carrier \
    --src D:/argus-data/roboflow/usps-merge --out D:/argus-data/crops/carrier
python training/crop/prepare_roboflow.py --task size \
    --src D:/argus-data/roboflow/mailers-and-boxes-2 D:/argus-data/roboflow/envelope \
    --out D:/argus-data/crops/size

# 3. train (CPU works; ~5 min an epoch for make on 12 threads)
python training/crop/train.py --task make    --data D:/argus-data/crops/make    --out D:/argus-data/models/crop_make
python training/crop/train.py --task carrier --data D:/argus-data/crops/carrier --out D:/argus-data/models/crop_carrier
python training/crop/train.py --task size    --data D:/argus-data/crops/size    --out D:/argus-data/models/crop_size
```

Each run writes `model.pt`, `metrics.json` (test accuracy, confusion
matrix, and for make the accuracy after collapsing to the 7 ARGUS slots)
and `model.onnx`.

**Feed the measured accuracy back into the sim:**
```bash
python sim/compare_taxonomy.py --acc <test_acc_argus_slots>
```

## On the node

`src/detection/crop_classifier.py` loads `crop_<task>/model.pt` from a
models directory and `DetectionService.attach_crop_classifier()` hooks it
in after YOLO. It fills `Detection.subtype`; any failure leaves it `None`.

Not done yet:
- **Hailo build.** `model.onnx` has to go through the Hailo Dataflow
  Compiler (x86 Linux or WSL2, see `scripts/export_hailo.py`) to become a
  `.hef`. Until then the crop stage runs on the Pi CPU, which costs far more
  than the 6 ms the `pi5-hailo-fine` profile budgets.
- **Federated features.** The live head still reads the 19 coarse features.
  Switching `src/federated/features.py` to the 35-dim fine layout changes
  the model shape on every node and the aggregator, so it is a team
  decision, not a drop-in.

Datasets live on `D:\argus-data` (C: is nearly full). Nothing under it is
committed.
