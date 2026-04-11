"""
Local training infrastructure for ARGUS Federated Learning.

LocalTrainer collects high-confidence detection crops from the live camera
stream and uses them to fine-tune the YOLOv8n detection head locally before
contributing updates to the Flower FL server.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    _TORCH_AVAILABLE = False
    logger.warning("torch not installed — LocalTrainer.train_epoch will be unavailable")

try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _YOLO = None
    _ULTRALYTICS_AVAILABLE = False


class LocalTrainer:
    """
    Collects detection samples from the live stream and fine-tunes the
    YOLOv8n detection head using local data.

    Training freezes the backbone (all layers except the detection head)
    so that only task-specific weights are updated — this preserves the
    general feature extractor while adapting to the local environment.
    """

    def __init__(self, model_manager, config) -> None:
        """
        Args:
            model_manager: ModelManager instance (src.detection.models).
            config: Settings instance (src.config.Settings).
        """
        self.model_manager = model_manager
        self.config = config

        from src.training.dataset import LocalDetectionDataset  # local import to avoid circularity
        self.dataset = LocalDetectionDataset(data_dir=config.FL_TRAINING_DIR)

    # ------------------------------------------------------------------
    # Sample collection
    # ------------------------------------------------------------------

    def collect_sample(
        self,
        frame: np.ndarray,
        detections: List,
    ) -> int:
        """
        Save high-confidence detection crops as training data.

        Only detections with confidence >= 0.7 are kept; lower-confidence
        detections are too noisy to be reliable training signal.

        Args:
            frame: Full camera frame (BGR numpy array).
            detections: List of Detection objects from DetectionService.

        Returns:
            Number of samples actually saved in this call.
        """
        saved = 0
        for det in detections:
            if det.confidence < 0.7:
                continue

            x, y, w, h = det.bbox
            # Guard against out-of-bounds crops
            fh, fw = frame.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(fw, x + w)
            y2 = min(fh, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            label = det.label or det.type.value

            self.dataset.add_sample(
                image=crop,
                label=label,
                bbox=(x1, y1, x2 - x1, y2 - y1),
            )
            saved += 1

        return saved

    def has_enough_samples(self) -> bool:
        """
        Check whether the local dataset has enough samples to train.

        Returns:
            True if sample count >= config.FL_MIN_SAMPLES.
        """
        return len(self.dataset) >= self.config.FL_MIN_SAMPLES

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        model,
        epochs: int = 5,
    ) -> Dict[str, float]:
        """
        Fine-tune the YOLOv8n detection head on the local dataset.

        The backbone is frozen (requires_grad = False) so that only the
        detection head adapts.  Uses the ultralytics Trainer under the hood
        via model.train() with the local dataset exported as a YOLO-format
        data directory.

        Args:
            model: YOLO model instance.
            epochs: Number of fine-tuning epochs.

        Returns:
            Metrics dict containing at least {"loss": float}.
            Returns {"loss": 0.0, "error": "..."} on failure.
        """
        if not _ULTRALYTICS_AVAILABLE or not _TORCH_AVAILABLE:
            logger.warning("Cannot train: ultralytics or torch not available")
            return {"loss": 0.0, "error": "dependencies missing"}

        if not self.has_enough_samples():
            logger.info(
                "Not enough samples to train (%d / %d)",
                len(self.dataset),
                self.config.FL_MIN_SAMPLES,
            )
            return {"loss": 0.0, "error": "insufficient samples"}

        try:
            pytorch_model = model.model

            # Freeze backbone — unfreeze only the detection head (model[-1])
            for param in pytorch_model.parameters():
                param.requires_grad = False
            for param in pytorch_model.model[-1].parameters():
                param.requires_grad = True

            # Export dataset to a temporary YOLO-format directory
            data_yaml = self.dataset.export_yolo_format()

            logger.info("Starting local fine-tuning for %d epoch(s)", epochs)
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=640,
                batch=8,
                workers=2,
                verbose=False,
                exist_ok=True,
            )

            # Extract loss from results
            loss_value = 0.0
            if hasattr(results, "results_dict"):
                loss_value = float(
                    results.results_dict.get("train/box_loss", 0.0)
                )

            metrics = {
                "loss": loss_value,
                "epochs": float(epochs),
                "samples": float(len(self.dataset)),
            }
            logger.info("Local training complete: %s", metrics)
            return metrics

        except Exception as exc:
            logger.error("Training failed: %s", exc)
            return {"loss": 0.0, "error": str(exc)}

    def evaluate(self, model) -> Tuple[float, float]:
        """
        Evaluate the model on the local dataset.

        Returns a (loss, accuracy) tuple.  When the dataset is too small
        or dependencies are missing, returns (0.0, 0.0).
        """
        if not _ULTRALYTICS_AVAILABLE or not _TORCH_AVAILABLE:
            return 0.0, 0.0

        if len(self.dataset) == 0:
            return 0.0, 0.0

        try:
            data_yaml = self.dataset.export_yolo_format()
            metrics = model.val(data=str(data_yaml), verbose=False)

            loss = float(getattr(metrics, "box_loss", 0.0))
            # mAP50 used as accuracy proxy
            accuracy = float(getattr(metrics, "map50", 0.0))
            return loss, accuracy
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # Dataset access
    # ------------------------------------------------------------------

    def get_training_data(self):
        """
        Return the underlying LocalDetectionDataset.

        Used by the FL client to report num_examples to the server.
        """
        return self.dataset

    def clear_old_samples(self, max_age_days: int = 7) -> int:
        """
        Delete training samples older than max_age_days.

        Args:
            max_age_days: Samples older than this are removed.

        Returns:
            Number of samples deleted.
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        removed = self.dataset.purge_before(cutoff)
        logger.info("Cleared %d samples older than %d days", removed, max_age_days)
        return removed
