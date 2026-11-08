"""Run a Hailo model-zoo YOLO HEF on the AI HAT+ 2 (Hailo-10H) via HailoRT.

Ultralytics cannot load a model-zoo HEF: it has no metadata.yaml, and NMS is
compiled into the HEF itself. This wrapper talks to HailoRT directly and
returns plain tuples, so DetectionService can treat it like any other backend.

Requires the `hailo_platform` Python bindings (apt: hailo-h10-all). A venv
sees them only when it was created with --system-site-packages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Where the zoo HEFs live (not tracked by git, like everything in models/).
HAILO_ZOO_DIR = ("models", "hailo10h")
# Inference timeout. The first call can be slow while the model loads.
_RUN_TIMEOUT_MS = 10_000
# Grey padding, the same value Ultralytics letterboxes with.
_PAD_VALUE = 114
_BOX_FIELDS = 5  # ymin, xmin, ymax, xmax, score (normalised to the input)

# (coco_id, score, x, y, w, h) in frame pixels
RawBox = Tuple[int, float, int, int, int, int]


def find_zoo_hef(base_dir: Path, model_name: str) -> Optional[Path]:
    """Newest `<model_name>_v*.hef` under models/hailo10h, or None.

    The zoo spells YOLO11 as "yolov11n", so both spellings are tried.
    """
    folder = base_dir.joinpath(*HAILO_ZOO_DIR)
    if not folder.is_dir():
        return None
    names = {model_name, model_name.replace("yolo11", "yolov11")}
    hits = sorted(p for n in names for p in folder.glob(f"{n}_v*.hef"))
    return hits[-1] if hits else None


def letterbox(frame: np.ndarray, size: Tuple[int, int]):
    """Resize keeping aspect ratio, pad to `size` (h, w). Returns img, scale, pad."""
    th, tw = size
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (tw - nw) // 2, (th - nh) // 2
    out = np.full((th, tw, 3), _PAD_VALUE, dtype=np.uint8)
    out[py:py + nh, px:px + nw] = resized
    return out, scale, (px, py)


def _per_class(buffer) -> Sequence[np.ndarray]:
    """Normalise HailoRT's NMS-by-class output to one (N, 5) array per class.

    HailoRT hands this back either as a list (one array per class) or as a
    flat float buffer laid out per class as [count, box*count].
    """
    if isinstance(buffer, (list, tuple)) or (
        isinstance(buffer, np.ndarray) and buffer.dtype == object
    ):
        return [np.asarray(c, dtype=np.float32).reshape(-1, _BOX_FIELDS) for c in buffer]
    flat = np.asarray(buffer, dtype=np.float32).ravel()
    classes, i = [], 0
    while i < flat.size:
        n = int(flat[i])
        i += 1
        classes.append(flat[i:i + n * _BOX_FIELDS].reshape(-1, _BOX_FIELDS))
        i += n * _BOX_FIELDS
    return classes


def decode(buffer, conf: float, classes: Optional[set], input_hw, scale, pad,
           frame_hw) -> List[RawBox]:
    """NMS-by-class buffer -> boxes in original frame pixels."""
    ih, iw = input_hw
    fh, fw = frame_hw
    px, py = pad
    out: List[RawBox] = []
    for cls_id, dets in enumerate(_per_class(buffer)):
        if classes is not None and cls_id not in classes:
            continue
        for ymin, xmin, ymax, xmax, score in dets:
            if score < conf:
                continue
            x0 = max(0.0, (xmin * iw - px) / scale)
            y0 = max(0.0, (ymin * ih - py) / scale)
            x1 = min(float(fw), (xmax * iw - px) / scale)
            y1 = min(float(fh), (ymax * ih - py) / scale)
            if x1 <= x0 or y1 <= y0:
                continue
            out.append((cls_id, float(score), int(x0), int(y0),
                        int(x1 - x0), int(y1 - y0)))
    return out


class HailoYolo:
    """One HEF, configured once, run synchronously frame by frame."""

    def __init__(self, hef_path: Path):
        from hailo_platform import FormatType, VDevice  # ImportError -> caller falls back

        self.hef_path = Path(hef_path)
        self._vdevice = VDevice(VDevice.create_params())
        model = self._vdevice.create_infer_model(str(self.hef_path))
        model.set_batch_size(1)
        model.input().set_format_type(FormatType.UINT8)
        model.output().set_format_type(FormatType.FLOAT32)
        self._ctx = model.configure()
        self._configured = self._ctx.__enter__()
        shape = tuple(model.input().shape)  # (h, w, 3)
        self.input_hw = (int(shape[0]), int(shape[1]))
        self._out_shape = tuple(model.output().shape)
        self._bindings = self._configured.create_bindings()
        logger.info("HailoRT ready: %s, input %s, output %s %s",
                    self.hef_path.name, shape, model.output().name, self._out_shape)

    def infer(self, frame_bgr: np.ndarray, conf: float,
              classes: Optional[set] = None) -> List[RawBox]:
        img, scale, pad = letterbox(frame_bgr, self.input_hw)
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self._bindings.input().set_buffer(rgb)
        self._bindings.output().set_buffer(np.empty(self._out_shape, dtype=np.float32))
        self._configured.run([self._bindings], _RUN_TIMEOUT_MS)
        buf = self._bindings.output().get_buffer()
        return decode(buf, conf, classes, self.input_hw, scale, pad,
                      frame_bgr.shape[:2])

    def close(self) -> None:
        try:
            self._ctx.__exit__(None, None, None)
        finally:
            self._vdevice.release()
