"""
Second-stage crop classifier: carrier, vehicle make and package size.

YOLO says "vehicle" or "package". This stage looks inside that box and
says which one: a UPS truck, a Nissan, a large parcel. The answers become
the fine-taxonomy slots (sim/workload_fine.py) that let the federated head
tell the resident's car at 03:00 from a stranger's.

Each task is its own small classifier (training/crop/train.py), so a site
can run only the ones it has a model for. Backends:
    torch   model.pt from training/crop/train.py (dev boxes, CPU)
    onnx    model.onnx via onnxruntime, when installed
On the Pi the Hailo backend is the target; that needs the .hef compiled on
an x86 workstation (scripts/export_hailo.py) and is not wired here yet.

Every failure degrades to "no subtype", never to a dropped detection.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

IMG = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAD = 0.1  # must match training/crop/taxonomy.crop_box

# Slot names, mirrored from sim/workload_fine.py.
CARRIERS = ("none", "ups", "usps", "fedex", "amazon")
MAKES = ("nissan", "toyota", "honda", "ford", "chevrolet", "tesla", "other")
SIZES = ("mail", "small", "large")

# Which tasks run on which coarse class.
TASKS_FOR = {"vehicle": ("carrier", "make"), "package": ("carrier", "size")}

Backend = Callable[[np.ndarray], np.ndarray]  # (N,3,224,224) float32 -> (N,C) logits


@dataclass(frozen=True)
class TaskModel:
    task: str
    classes: Tuple[str, ...]
    run: Backend


def to_slot(task: str, label: str) -> str:
    """A model's class name -> the ARGUS slot value for that task."""
    key = label.lower()
    if task == "make":
        return key if key in MAKES else "other"
    if task == "carrier":
        return key if key in CARRIERS else "none"
    if task == "size":
        return key if key in SIZES else "small"
    raise ValueError(f"unknown task {task!r}")


def padded_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                min_side: int = 16) -> Optional[np.ndarray]:
    """Crop an (x, y, w, h) box with PAD on each side; None if too small."""
    x, y, w, h = bbox
    if w < min_side or h < min_side:
        return None
    H, W = frame.shape[:2]
    px, py = w * PAD, h * PAD
    x1, y1 = max(0, int(round(x - px))), max(0, int(round(y - py)))
    x2, y2 = min(W, int(round(x + w + px))), min(H, int(round(y + h + py)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    """BGR crop -> normalised (3, 224, 224) float32, matching training eval."""
    import cv2
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def torch_backend(path: Path) -> TaskModel:
    import torch
    from torch import nn
    from torchvision import models

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    classes = tuple(ckpt["classes"])
    net = models.mobilenet_v3_large(weights=None)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(classes))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    def run(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net(torch.from_numpy(batch)).numpy()

    return TaskModel(task=ckpt["task"], classes=classes, run=run)


def onnx_backend(path: Path, task: str, classes: Sequence[str]) -> TaskModel:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return TaskModel(task=task, classes=tuple(classes),
                     run=lambda batch: sess.run(None, {name: batch})[0])


class CropClassifier:
    """Runs the per-task models over vehicle and package boxes."""

    def __init__(self, models: Sequence[TaskModel]):
        self.models: Dict[str, TaskModel] = {m.task: m for m in models}
        unknown = set(self.models) - {"carrier", "make", "size"}
        if unknown:
            raise ValueError(f"unknown crop tasks: {sorted(unknown)}")

    @classmethod
    def from_dir(cls, root: Path) -> "CropClassifier":
        """Load every crop_<task>/model.pt found under root."""
        found = []
        for task in ("carrier", "make", "size"):
            p = Path(root) / f"crop_{task}" / "model.pt"
            if p.exists():
                found.append(torch_backend(p))
                logger.info("crop classifier: loaded %s from %s", task, p)
        if not found:
            raise FileNotFoundError(f"no crop_<task>/model.pt under {root}")
        return cls(found)

    def classify(self, frame: np.ndarray, kind: str,
                 bboxes: Sequence[Tuple[int, int, int, int]]) -> List[Optional[Dict]]:
        """One result per bbox: {carrier, make, size, confidence} or None."""
        tasks = [t for t in TASKS_FOR.get(kind, ()) if t in self.models]
        out: List[Optional[Dict]] = [None] * len(bboxes)
        if not tasks:
            return out
        keep, batch = [], []
        for i, b in enumerate(bboxes):
            crop = padded_crop(frame, b)
            if crop is not None:
                keep.append(i)
                batch.append(preprocess(crop))
        if not batch:
            return out
        x = np.stack(batch).astype(np.float32)
        results = [{"confidence": 1.0} for _ in keep]
        for task in tasks:
            m = self.models[task]
            p = _softmax(m.run(x))
            idx = p.argmax(axis=1)
            for r, k, pr in zip(results, idx, p.max(axis=1)):
                r[task] = to_slot(task, m.classes[k])
                r["confidence"] = min(r["confidence"], float(pr))
        for i, r in zip(keep, results):
            out[i] = r
        return out
