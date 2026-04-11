"""
Model management utilities for ARGUS detection models.

Provides weight extraction, injection, and delta computation used by the
Federated Learning layer to exchange model updates without shipping full
model files over the network.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO as _YOLO
    import torch as _torch
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _YOLO = None
    _torch = None
    _ULTRALYTICS_AVAILABLE = False
    logger.warning(
        "ultralytics / torch not installed — ModelManager will be unavailable"
    )


class ModelManager:
    """
    Manages a YOLOv8n model instance and provides utilities for Federated
    Learning weight exchange.

    Weight lists are ordered lists of numpy arrays — one array per named
    parameter tensor in the model.  The order is deterministic (dict insertion
    order, Python 3.7+) so clients can exchange weight lists without metadata.
    """

    def __init__(self) -> None:
        self.model: Optional[object] = None  # YOLO instance

    # ------------------------------------------------------------------
    # Model loading / saving
    # ------------------------------------------------------------------

    def load_model(self, model_path: Optional[Path] = None) -> object:
        """
        Load a YOLOv8n model.

        If model_path is given and exists, loads from that checkpoint;
        otherwise falls back to the default pretrained yolov8n.pt weights.
        """
        if not _ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "ultralytics is required for ModelManager. "
                "Install it with: pip install ultralytics"
            )

        # Default fallback logic
        default_model_name = "yolov8n.pt"
        # 1. Try absolute path provided
        # 2. Try current directory
        # 3. Try src/ directory relative to this file
        src_path = Path(__file__).parent.parent / default_model_name

        if model_path is not None and Path(model_path).exists():
            logger.info("Loading YOLOv8n from checkpoint: %s", model_path)
            self.model = _YOLO(str(model_path))
        elif Path(default_model_name).exists():
            logger.info("Loading %s from current directory", default_model_name)
            self.model = _YOLO(default_model_name)
        elif src_path.exists():
            logger.info("Loading %s from src directory: %s", default_model_name, src_path)
            self.model = _YOLO(str(src_path))
        else:
            logger.info("Model not found locally, downloading pretrained %s", default_model_name)
            self.model = _YOLO(default_model_name)

        return self.model

    def save_model(self, model: object, path: Path) -> None:
        """
        Save the model's PyTorch state_dict to disk.

        Args:
            model: YOLO model instance (must have .model attribute).
            path: Destination .pt file path.
        """
        if _torch is None:
            raise RuntimeError("torch is required to save model weights")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        pytorch_model = model.model  # type: ignore[attr-defined]
        _torch.save(pytorch_model.state_dict(), str(path))
        logger.info("Model state_dict saved to %s", path)

    def load_state_dict(self, path: Path) -> dict:
        """
        Load a PyTorch state_dict from a .pt file.

        Args:
            path: Path to the .pt file produced by save_model().

        Returns:
            State dict mapping parameter names to tensors.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        if _torch is None:
            raise RuntimeError("torch is required to load state_dict")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"State dict not found: {path}")

        state_dict = _torch.load(str(path), map_location="cpu")
        logger.info("State dict loaded from %s (%d keys)", path, len(state_dict))
        return state_dict

    # ------------------------------------------------------------------
    # Weight extraction / injection (for Federated Learning)
    # ------------------------------------------------------------------

    def get_model_weights(self, model: object) -> List[np.ndarray]:
        """
        Extract model parameters as a list of numpy arrays.

        The order matches the model's named parameters, which is stable
        across identical architectures.

        Args:
            model: YOLO model instance.

        Returns:
            Ordered list of numpy arrays — one per parameter tensor.
        """
        if _torch is None:
            raise RuntimeError("torch is required for weight extraction")

        pytorch_model = model.model  # type: ignore[attr-defined]
        return [
            param.detach().cpu().numpy()
            for param in pytorch_model.parameters()
        ]

    def set_model_weights(self, model: object, weights: List[np.ndarray]) -> None:
        """
        Inject a list of numpy arrays into the model as its parameters.

        The weight list must have the same length and shapes as the model's
        current parameters (i.e. from get_model_weights on the same arch).

        Args:
            model: YOLO model instance to modify in-place.
            weights: Ordered list of numpy arrays.

        Raises:
            ValueError: If weight count or shapes don't match the model.
        """
        if _torch is None:
            raise RuntimeError("torch is required for weight injection")

        pytorch_model = model.model  # type: ignore[attr-defined]
        params = list(pytorch_model.parameters())

        if len(params) != len(weights):
            raise ValueError(
                f"Weight count mismatch: model has {len(params)} params, "
                f"got {len(weights)} weight arrays"
            )

        with _torch.no_grad():
            for param, weight_array in zip(params, weights):
                tensor = _torch.from_numpy(weight_array).to(param.device)
                if tensor.shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch: param shape {param.shape}, "
                        f"weight shape {tensor.shape}"
                    )
                param.copy_(tensor)

    def get_weight_delta(
        self,
        old_weights: List[np.ndarray],
        new_weights: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Compute the element-wise delta between two weight lists.

        delta[i] = new_weights[i] - old_weights[i]

        Useful for sending only the change to the FL server rather than
        the full parameter set.

        Args:
            old_weights: Weights before local training.
            new_weights: Weights after local training.

        Returns:
            List of delta arrays (same shapes as inputs).
        """
        if len(old_weights) != len(new_weights):
            raise ValueError(
                f"Weight list length mismatch: {len(old_weights)} vs {len(new_weights)}"
            )
        return [new - old for old, new in zip(old_weights, new_weights)]

    def apply_weight_delta(
        self,
        base_weights: List[np.ndarray],
        delta: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Apply a delta to a base weight list.

        result[i] = base_weights[i] + delta[i]

        Args:
            base_weights: Starting weights.
            delta: Delta arrays from get_weight_delta().

        Returns:
            Updated weight list (same shapes as inputs).
        """
        if len(base_weights) != len(delta):
            raise ValueError(
                f"Weight list length mismatch: {len(base_weights)} vs {len(delta)}"
            )
        return [base + d for base, d in zip(base_weights, delta)]
