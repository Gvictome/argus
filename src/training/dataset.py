"""
Local detection dataset for ARGUS Federated Learning training.

Stores detection crops on disk and maintains a JSON index so that samples
survive process restarts.  The dataset can export itself to a YOLO-format
directory (images/ + labels/ + data.yaml) for ultralytics model.train().
"""

import json
import logging
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from torch.utils.data import Dataset as _TorchDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TorchDataset = object  # fallback base class
    _TORCH_AVAILABLE = False
    logger.warning("torch not installed — LocalDetectionDataset won't be a torch Dataset")


def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to an image for training diversity.

    Operations applied:
    - Random horizontal flip (50% probability)
    - Random brightness shift (+/- 30)
    - Gaussian noise (sigma 0–15)

    Args:
        image: BGR numpy array.

    Returns:
        Augmented copy of the image.
    """
    img = image.copy()

    # Random horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Random brightness
    delta = random.randint(-30, 30)
    img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    # Gaussian noise
    sigma = random.uniform(0, 15)
    if sigma > 0:
        noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


class LocalDetectionDataset(_TorchDataset):
    """
    Filesystem-backed dataset of detection crop images.

    Directory layout::

        data_dir/
            index.json          ← sample metadata list
            images/
                <timestamp>_<n>.jpg
            labels/
                <timestamp>_<n>.txt   ← YOLO format: cls cx cy w h (normalised)

    The index.json is the source of truth; images not in the index are ignored.
    """

    INDEX_FILE = "index.json"
    IMAGES_DIR = "images"
    LABELS_DIR = "labels"

    def __init__(self, data_dir: Path) -> None:
        """
        Args:
            data_dir: Root directory for storing dataset files.
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / self.IMAGES_DIR
        self.labels_dir = self.data_dir / self.LABELS_DIR
        self.index_path = self.data_dir / self.INDEX_FILE

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self._index: List[Dict] = self._load_index()

        # Collect all unique label names for class mapping
        self._label_set: List[str] = self._build_label_set()

    # ------------------------------------------------------------------
    # Torch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Return (image, metadata) for a sample.

        Image is returned as a BGR numpy array (H, W, 3).
        Metadata is the raw index entry dict.
        """
        entry = self._index[idx]
        image_path = self.data_dir / entry["image_path"]

        img = cv2.imread(str(image_path))
        if img is None:
            # Return a blank image if file is missing (handles disk corruption)
            img = np.zeros((64, 64, 3), dtype=np.uint8)

        return img, entry

    # ------------------------------------------------------------------
    # Sample management
    # ------------------------------------------------------------------

    def add_sample(
        self,
        image: np.ndarray,
        label: str,
        bbox: Tuple[int, int, int, int],
    ) -> str:
        """
        Save a detection crop to disk and record it in the index.

        The image is resized to 128x128 before saving to keep disk usage low.

        Args:
            image: Crop as BGR numpy array.
            label: String class name (e.g. "person", "face").
            bbox: (x, y, w, h) in the ORIGINAL frame — stored as metadata.

        Returns:
            Filename stem of the saved sample (without extension).
        """
        ts = int(time.time() * 1000)
        n = len(self._index)
        stem = f"{ts}_{n}"

        img_filename = f"{stem}.jpg"
        lbl_filename = f"{stem}.txt"

        # Resize crop to standard size
        resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(self.images_dir / img_filename), resized)

        # Write YOLO-format label (single object, centred, fills frame)
        cls_id = self._get_or_add_label(label)
        with open(self.labels_dir / lbl_filename, "w") as f:
            f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")

        entry = {
            "stem": stem,
            "image_path": f"{self.IMAGES_DIR}/{img_filename}",
            "label_path": f"{self.LABELS_DIR}/{lbl_filename}",
            "label": label,
            "bbox": list(bbox),
            "timestamp": datetime.now().isoformat(),
        }
        self._index.append(entry)

        # Update label set if new
        if label not in self._label_set:
            self._label_set.append(label)

        self._save_index()
        return stem

    def get_stats(self) -> Dict[str, int]:
        """
        Return the distribution of labels in the dataset.

        Returns:
            Dict mapping label name to sample count.
        """
        stats: Dict[str, int] = {}
        for entry in self._index:
            lbl = entry.get("label", "unknown")
            stats[lbl] = stats.get(lbl, 0) + 1
        return stats

    def purge_before(self, cutoff: datetime) -> int:
        """
        Delete all samples with timestamps older than cutoff.

        Args:
            cutoff: datetime threshold; samples before this are removed.

        Returns:
            Number of samples deleted.
        """
        removed = 0
        surviving: List[Dict] = []

        for entry in self._index:
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
            except (KeyError, ValueError):
                surviving.append(entry)
                continue

            if ts < cutoff:
                # Delete files
                for path_key in ("image_path", "label_path"):
                    p = self.data_dir / entry.get(path_key, "")
                    if p.exists():
                        p.unlink(missing_ok=True)
                removed += 1
            else:
                surviving.append(entry)

        self._index = surviving
        self._label_set = self._build_label_set()
        self._save_index()
        return removed

    # ------------------------------------------------------------------
    # YOLO export
    # ------------------------------------------------------------------

    def export_yolo_format(self) -> Path:
        """
        Write a data.yaml file that points ultralytics at this dataset.

        The images and labels are already stored in the expected layout
        (images/ and labels/ sibling directories), so this just writes the
        yaml manifest.

        Returns:
            Path to the data.yaml file.
        """
        import yaml  # standard library fallback handled below

        yaml_path = self.data_dir / "data.yaml"
        names = {i: name for i, name in enumerate(self._label_set)}

        data = {
            "path": str(self.data_dir),
            "train": "images",
            "val": "images",   # use same set for local fine-tuning
            "names": names,
            "nc": len(self._label_set),
        }

        try:
            import yaml as _yaml
            with open(yaml_path, "w") as f:
                _yaml.dump(data, f, default_flow_style=False)
        except ImportError:
            # Fallback: write yaml manually
            with open(yaml_path, "w") as f:
                f.write(f"path: {self.data_dir}\n")
                f.write("train: images\n")
                f.write("val: images\n")
                f.write(f"nc: {len(self._label_set)}\n")
                f.write("names:\n")
                for i, name in enumerate(self._label_set):
                    f.write(f"  {i}: {name}\n")

        return yaml_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> List[Dict]:
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load dataset index: %s", exc)
        return []

    def _save_index(self) -> None:
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _build_label_set(self) -> List[str]:
        seen: List[str] = []
        for entry in self._index:
            lbl = entry.get("label", "unknown")
            if lbl not in seen:
                seen.append(lbl)
        return seen

    def _get_or_add_label(self, label: str) -> int:
        if label not in self._label_set:
            self._label_set.append(label)
        return self._label_set.index(label)
