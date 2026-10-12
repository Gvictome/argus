# Base detector comparison (2026-10-11)

**Result:** YOLO26n and YOLO11n tie on the classes ARGUS uses (0.458 and
0.457 mAP50-95), ahead of the current YOLOv8n (0.441). YOLOv8n is the
fastest on a laptop CPU. All three compile for the Hailo.
**Pick YOLO26n if the Hailo compile works by 10/18; otherwise keep
YOLOv8n on the CPU**, since it is already exported and measured on the Pi.

COCO val2017, first 500 images sorted (never coco8/coco128, which are
training images), imgsz 640, laptop CPU (AMD, PyTorch eager).

| Model | Params (M) | mAP50-95 | mAP50 | ARGUS classes mAP50-95 | person | vehicle | animal | package | CPU ms @640 | CPU ms @416 | Hailo export |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n | 3.16 | 0.404 | 0.555 | 0.441 | 0.505 | 0.383 | 0.592 | 0.235 | 45.8 | 23.2 | yes |
| yolo11n | 2.62 | 0.417 | 0.574 | 0.457 | 0.515 | 0.404 | 0.600 | 0.257 | 68.1 | 57.7 | yes |
| yolo26n | 2.57 | 0.434 | 0.599 | 0.458 | 0.514 | 0.412 | 0.593 | 0.261 | 66.6 | 44.9 | yes |

Package is the weakest class for every model: COCO has no parcel class, so
"package" means backpack, handbag or suitcase. Hailo support per the
[Ultralytics Hailo guide](https://docs.ultralytics.com/integrations/hailo).

## Still to do

Laptop times are not Pi times. On the Pi:

```bash
python scripts/compare_detectors.py --skip-accuracy --models yolov8n yolo26n
```

Compare NCNN exports (`scripts/export_cpu_models.py --format ncnn`), since that is what the node runs.

## Reproduce

```bash
python scripts/compare_detectors.py --prepare       # COCO val2017 into data/datasets/coco (~0.8 GB)
python scripts/compare_detectors.py --subset 500    # writes data/benchmarks/detector_compare_<date>.md
```
