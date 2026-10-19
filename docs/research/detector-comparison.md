# Base detector comparison (2026-10-11)

**Decision (2026-10-18): the base detector is YOLOv8n.** The 10/11 rule was
"YOLO26n if it compiles for the Hailo by 10/18, else YOLOv8n". It did not
compile: the Hailo Dataflow Compiler was never installed (it needs an x86-64
Linux host and a Hailo developer account), so no custom `.hef` exists for any
of the three models. The rule's fallback applies and is now locked.

**Result of the accuracy comparison (unchanged):** YOLO26n and YOLO11n tie on
the classes ARGUS uses (0.458 and 0.457 mAP50-95), ahead of YOLOv8n (0.441).
YOLOv8n is the fastest on a laptop CPU. All three are Hailo-exportable on
paper.

## Why YOLOv8n and not the +0.017 mAP of YOLO26n

| Reason | Detail |
|---|---|
| The rule said so | Tie-break was fixed on 10/11, before the result was known; honouring it keeps the decision from being made after the fact |
| Already the system default | `DetectionConfig.model_name = "yolov8n"` in `src/detection/__init__.py`; `yolov8n.pt` and the NCNN/OpenVINO exports are what every Pi measurement so far used |
| Fastest of the three on CPU | 45.8 ms @640 vs 66.6 (YOLO26n) and 68.1 (YOLO11n) on the laptop; CPU is the only backend that will run at the showcase, and the target is 8-12 FPS |
| The gain does not touch the federated claim | The head reads 19 numbers per tracked event, not pixels; 0.017 mAP moves detection quality slightly and the FL comparison not at all |
| Code freeze is 11/08 | Switching detectors now means re-measuring FPS on the Pi, re-tuning the threat threshold and re-checking the head's input distribution, with the Pi off and no compile path |

## Hailo status

- **Custom compile (ONNX to HEF with the Dataflow Compiler): future work.** Not attempted before the showcase.
- **Prebuilt model-zoo files are on disk** (`models/hailo10h/yolov8n_v5.1.0.hef`, `yolov8n_v5.2.0.hef`, `yolov11n_v5.1.0.hef`, `yolov11n_v5.2.0.hef`, not tracked by git). They are Hailo's own builds for the Hailo-10H. They are **not wired in**: `DetectionService` looks for an Ultralytics export directory (`models/yolov8n_hailo_model/`), and a zoo HEF has no such metadata or NMS wrapper. Using one means a HailoRT inference path in `src/detection/`. That is an optional post-freeze experiment, and if it is ever tried, YOLOv8n is the right first model (it matches the CPU path class-for-class).
- **Face detection stays out of the demo** (decided 2026-09-14, unchanged).
- The Hailo hardware itself is verified (76 FPS on `yolov8m_h10.hef` with `hailortcli benchmark`, see the demo runbook), so the future-work item is software, not hardware.

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
