"""
Utility functions for ARGUS Federated Learning weight transport.

These helpers handle serialization (compress/decompress) and arithmetic
(delta computation/application) on lists of numpy weight arrays.

They are intentionally standalone — no dependency on flwr or torch — so
they can be used for offline analysis or debugging without the full FL stack.
"""

import io
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def compress_weights(weights: List[np.ndarray]) -> bytes:
    """
    Serialize a list of numpy arrays to a compact byte string.

    Uses numpy's savez_compressed (zlib-compressed npz format) written to
    an in-memory buffer, so no temporary files are created on disk.

    Args:
        weights: Ordered list of numpy arrays (e.g. from ModelManager).

    Returns:
        Compressed bytes that can be sent over the network or stored.

    Example::

        data = compress_weights(model_manager.get_model_weights(model))
        # transmit `data` ...
        recovered = decompress_weights(data)
    """
    buf = io.BytesIO()
    # savez_compressed accepts keyword args; name arrays by index
    named = {str(i): arr for i, arr in enumerate(weights)}
    np.savez_compressed(buf, **named)
    return buf.getvalue()


def decompress_weights(data: bytes) -> List[np.ndarray]:
    """
    Deserialize compressed weight bytes back to a list of numpy arrays.

    The list is sorted by the integer key used during compression so that
    the original order is preserved exactly.

    Args:
        data: Bytes produced by compress_weights().

    Returns:
        Ordered list of numpy arrays.

    Raises:
        ValueError: If the data cannot be parsed as a numpy npz archive.
    """
    try:
        buf = io.BytesIO(data)
        archive = np.load(buf, allow_pickle=False)
        # Keys are string-encoded integers: "0", "1", ...
        sorted_keys = sorted(archive.files, key=lambda k: int(k))
        return [archive[k] for k in sorted_keys]
    except Exception as exc:
        raise ValueError(f"Failed to decompress weights: {exc}") from exc


def compute_delta(
    old_weights: List[np.ndarray],
    new_weights: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Compute the element-wise difference between two weight lists.

    delta[i] = new_weights[i] - old_weights[i]

    Transmitting deltas rather than full weights reduces bandwidth when
    updates are small relative to the total parameter count (e.g. when
    only the detection head is fine-tuned).

    Args:
        old_weights: Weights before local training.
        new_weights: Weights after local training.

    Returns:
        List of delta arrays with the same shapes as the input arrays.

    Raises:
        ValueError: If the two lists have different lengths or mismatched shapes.
    """
    if len(old_weights) != len(new_weights):
        raise ValueError(
            f"Weight list length mismatch: old={len(old_weights)}, new={len(new_weights)}"
        )
    deltas: List[np.ndarray] = []
    for i, (old, new) in enumerate(zip(old_weights, new_weights)):
        if old.shape != new.shape:
            raise ValueError(
                f"Shape mismatch at index {i}: old={old.shape}, new={new.shape}"
            )
        deltas.append(new - old)
    return deltas


def apply_delta(
    base_weights: List[np.ndarray],
    delta: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Apply a weight delta to a base weight list.

    result[i] = base_weights[i] + delta[i]

    Args:
        base_weights: Starting weights (e.g. global model from server).
        delta: Delta arrays from compute_delta().

    Returns:
        Updated weight list with the same shapes as the inputs.

    Raises:
        ValueError: If the two lists have different lengths or mismatched shapes.
    """
    if len(base_weights) != len(delta):
        raise ValueError(
            f"Weight list length mismatch: base={len(base_weights)}, delta={len(delta)}"
        )
    result: List[np.ndarray] = []
    for i, (base, d) in enumerate(zip(base_weights, delta)):
        if base.shape != d.shape:
            raise ValueError(
                f"Shape mismatch at index {i}: base={base.shape}, delta={d.shape}"
            )
        result.append(base + d)
    return result
