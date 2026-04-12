"""
Argus Federated Learning Module
================================
Uses Flower (flwr) for federated learning across multiple Pi nodes.
Each node trains a lightweight model on local detections, then shares
weight updates with the aggregation server.

Architecture:
  - FL Server: runs on one designated Pi (or central machine)
  - FL Client: runs on each Argus edge node alongside detection pipeline
  - Model: lightweight detection classifier fine-tuned per-node
  - Prometheus: all FL metrics exported via server.py /metrics endpoint
"""

import threading
import time
import logging
import json
import os
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import requests as http_requests

logger = logging.getLogger("argus.federated")

# ─── Configuration ───────────────────────────────────────────────────

FL_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "fl_config.json")
DEFAULT_CONFIG = {
    "server_address": "localhost:8080",
    "role": "client",  # "client" or "server"
    "min_clients": 2,
    "rounds": 10,
    "local_epochs": 3,
    "batch_size": 16,
    "learning_rate": 0.001,
    "model_dir": os.path.join(os.path.dirname(__file__), "models"),
    "data_dir": os.path.join(os.path.dirname(__file__), "fl_data"),
}


def load_config() -> dict:
    if os.path.exists(FL_CONFIG_FILE):
        with open(FL_CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        return {**DEFAULT_CONFIG, **cfg}
    return DEFAULT_CONFIG.copy()


# ─── Prometheus Metrics Bridge ───────────────────────────────────────
# These update the gauges defined in server.py

_prometheus_metrics = None


def set_prometheus_metrics(fl_round, fl_loss, fl_version, fl_peers):
    """Accept Prometheus gauge references from server.py."""
    global _prometheus_metrics
    _prometheus_metrics = {
        "round": fl_round,
        "loss": fl_loss,
        "version": fl_version,
        "peers": fl_peers,
    }


def _update_metrics(round_num=None, loss=None, version=None, peers=None):
    if _prometheus_metrics is None:
        return
    if round_num is not None:
        _prometheus_metrics["round"].set(round_num)
    if loss is not None:
        _prometheus_metrics["loss"].set(loss)
    if version is not None:
        _prometheus_metrics["version"].set(version)
    if peers is not None:
        _prometheus_metrics["peers"].set(peers)


# ─── Simple Model (numpy-based for Pi compatibility) ─────────────────

class ArgusModel:
    """Lightweight numpy-based classifier for federated learning.

    This serves as the model that gets trained locally and whose weights
    are exchanged during FL rounds. On Pi with Hailo, the actual YOLO
    detection is handled by the Hailo accelerator — this model learns
    *patterns* from detection metadata (time-of-day, frequency, bbox
    distributions) for anomaly detection and alert prioritization.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, output_dim: int = 4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Initialize weights
        np.random.seed(42)
        self.w1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.01
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def get_weights(self) -> List[np.ndarray]:
        return [self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()]

    def set_weights(self, weights: List[np.ndarray]):
        self.w1, self.b1, self.w2, self.b2 = weights

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        out = h @ self.w2 + self.b2
        # Softmax
        exp_out = np.exp(out - np.max(out, axis=-1, keepdims=True))
        return exp_out / np.sum(exp_out, axis=-1, keepdims=True)

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.001) -> float:
        """Single training step with cross-entropy loss. Returns loss value."""
        batch_size = x.shape[0]

        # Forward
        h = x @ self.w1 + self.b1
        h_relu = np.maximum(0, h)
        logits = h_relu @ self.w2 + self.b2
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Cross-entropy loss
        loss = -np.mean(np.sum(y * np.log(probs + 1e-8), axis=-1))

        # Backward
        d_logits = (probs - y) / batch_size
        d_w2 = h_relu.T @ d_logits
        d_b2 = np.sum(d_logits, axis=0)
        d_h_relu = d_logits @ self.w2.T
        d_h = d_h_relu * (h > 0).astype(np.float32)
        d_w1 = x.T @ d_h
        d_b1 = np.sum(d_h, axis=0)

        # Update
        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1
        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2

        return float(loss)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        data = np.load(path)
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        logger.info(f"Model loaded from {path}")


# ─── Detection Data Collector ───────────────────────────────────────

class DetectionDataCollector:
    """Collects detection events and converts them to training data.

    Features extracted from each detection:
      0: hour_sin (cyclic time encoding)
      1: hour_cos
      2: confidence
      3: bbox_area (normalized)
      4: bbox_center_x
      5: bbox_center_y
      6: is_person (binary)
      7: motion_intensity (from software motion detector)

    Labels (one-hot):
      0: normal_activity
      1: suspicious
      2: known_routine
      3: anomaly
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.buffer: List[dict] = []
        self.max_buffer = 1000

    def add_detection(self, detection: dict):
        self.buffer.append(detection)
        if len(self.buffer) > self.max_buffer:
            self.buffer.pop(0)

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert detection buffer to numpy training arrays."""
        if len(self.buffer) < 10:
            # Generate synthetic data for bootstrapping
            return self._generate_synthetic(100)

        features = []
        labels = []

        for det in self.buffer:
            try:
                import datetime as dt
                ts = det.get("timestamp", "")
                if isinstance(ts, str) and "T" in ts:
                    hour = int(ts.split("T")[1].split(":")[0])
                else:
                    hour = 12

                bbox = det.get("bbox", {})
                area = (bbox.get("xmax", 0.5) - bbox.get("xmin", 0.5)) * \
                       (bbox.get("ymax", 0.5) - bbox.get("ymin", 0.5))
                cx = (bbox.get("xmin", 0.5) + bbox.get("xmax", 0.5)) / 2
                cy = (bbox.get("ymin", 0.5) + bbox.get("ymax", 0.5)) / 2

                feat = [
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                    det.get("confidence", 0.5),
                    area,
                    cx,
                    cy,
                    1.0 if det.get("label") == "person" else 0.0,
                    det.get("motion_intensity", 0.5),
                ]
                features.append(feat)

                # Simple label heuristic (will be refined by FL)
                label = self._auto_label(det, hour)
                labels.append(label)

            except Exception:
                continue

        if not features:
            return self._generate_synthetic(100)

        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)

    def _auto_label(self, det: dict, hour: int) -> List[float]:
        """Heuristic labeling for bootstrap training."""
        label = det.get("label", "")
        conf = det.get("confidence", 0.5)

        # Night + person + high confidence = suspicious
        if label == "person" and (hour < 6 or hour > 22) and conf > 0.7:
            return [0.0, 1.0, 0.0, 0.0]  # suspicious
        # Daytime + common objects = normal
        if 7 <= hour <= 20 and label in ("car", "person", "dog", "cat"):
            return [1.0, 0.0, 0.0, 0.0]  # normal
        # Known patterns
        if label in ("car", "truck") and 7 <= hour <= 9:
            return [0.0, 0.0, 1.0, 0.0]  # routine
        # Default: normal
        return [1.0, 0.0, 0.0, 0.0]

    def _generate_synthetic(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for cold start."""
        x = np.random.randn(n, 8).astype(np.float32) * 0.5
        y = np.zeros((n, 4), dtype=np.float32)
        y[np.arange(n), np.random.randint(0, 4, n)] = 1.0
        return x, y


# ─── Flower FL Client ───────────────────────────────────────────────

class ArgusFlowerClient:
    """Flower-compatible FL client for Argus nodes."""

    def __init__(self, config: dict):
        self.config = config
        self.model = ArgusModel()
        self.data_collector = DetectionDataCollector(config["data_dir"])
        self._current_round = 0
        self._running = False
        self._thread = None

        model_path = os.path.join(config["model_dir"], "argus_model.npz")
        if os.path.exists(model_path):
            self.model.load(model_path)

    def add_detection(self, detection: dict):
        """Feed detection events to the FL data collector."""
        self.data_collector.add_detection(detection)

    def get_parameters(self) -> List[np.ndarray]:
        return self.model.get_weights()

    def set_parameters(self, parameters: List[np.ndarray]):
        self.model.set_weights(parameters)

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        """Local training round."""
        self.model.set_weights(parameters)
        x_train, y_train = self.data_collector.prepare_training_data()

        epochs = self.config.get("local_epochs", 3)
        lr = self.config.get("learning_rate", 0.001)
        batch_size = self.config.get("batch_size", 16)
        total_loss = 0.0

        for epoch in range(epochs):
            indices = np.random.permutation(len(x_train))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(x_train), batch_size):
                batch_idx = indices[i:i + batch_size]
                loss = self.model.train_step(x_train[batch_idx], y_train[batch_idx], lr)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            total_loss = avg_loss

        self._current_round += 1
        _update_metrics(round_num=self._current_round, loss=total_loss,
                        version=self._current_round)

        logger.info(f"FL round {self._current_round}: loss={total_loss:.4f}, "
                    f"samples={len(x_train)}")

        return self.model.get_weights(), len(x_train), {"loss": total_loss}

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        """Evaluate model on local data."""
        self.model.set_weights(parameters)
        x, y = self.data_collector.prepare_training_data()

        preds = self.model.forward(x)
        loss = -np.mean(np.sum(y * np.log(preds + 1e-8), axis=-1))
        accuracy = np.mean(np.argmax(preds, axis=-1) == np.argmax(y, axis=-1))

        return float(loss), len(x), {"accuracy": float(accuracy)}

    def start_flower_client(self):
        """Start Flower client in background thread."""
        if self._running:
            logger.warning("FL client already running")
            return

        self._running = True

        def _run():
            try:
                import flwr as fl

                class _Client(fl.client.NumPyClient):
                    def __init__(inner_self):
                        super().__init__()

                    def get_parameters(inner_self, config):
                        return self.get_parameters()

                    def fit(inner_self, parameters, config):
                        return self.fit(parameters, config)

                    def evaluate(inner_self, parameters, config):
                        return self.evaluate(parameters, config)

                logger.info(f"Connecting to FL server at {self.config['server_address']}")
                fl.client.start_numpy_client(
                    server_address=self.config["server_address"],
                    client=_Client(),
                )
            except ImportError:
                logger.warning("Flower not installed — running in standalone mode")
                self._standalone_training_loop()
            except Exception as e:
                logger.error(f"FL client error: {e}")
                self._standalone_training_loop()
            finally:
                self._running = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _standalone_training_loop(self):
        """Fallback: train locally without FL server connection."""
        logger.info("Running standalone training loop (no FL server)")
        while self._running:
            try:
                weights = self.get_parameters()
                new_weights, n_samples, metrics = self.fit(weights, {})
                self.model.set_weights(new_weights)

                model_path = os.path.join(self.config["model_dir"], "argus_model.npz")
                self.model.save(model_path)

                logger.info(f"Standalone round {self._current_round}: "
                            f"loss={metrics['loss']:.4f}")
            except Exception as e:
                logger.error(f"Standalone training error: {e}")

            time.sleep(60)  # Train every 60 seconds

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)


# ─── Flower FL Server ───────────────────────────────────────────────

class ArgusFlowerServer:
    """Flower-compatible FL aggregation server."""

    def __init__(self, config: dict):
        self.config = config
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return

        self._running = True

        def _run():
            try:
                import flwr as fl

                strategy = fl.server.strategy.FedAvg(
                    min_fit_clients=self.config.get("min_clients", 2),
                    min_evaluate_clients=self.config.get("min_clients", 2),
                    min_available_clients=self.config.get("min_clients", 2),
                )

                logger.info(f"Starting FL server on {self.config['server_address']}")
                fl.server.start_server(
                    server_address=self.config["server_address"],
                    config=fl.server.ServerConfig(
                        num_rounds=self.config.get("rounds", 10)
                    ),
                    strategy=strategy,
                )
            except ImportError:
                logger.warning("Flower not installed — FL server not started")
            except Exception as e:
                logger.error(f"FL server error: {e}")
            finally:
                self._running = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False


# ─── Central Server Registration ────────────────────────────────────

def register_with_central(config: dict):
    """Register this edge node with the central FL aggregation server."""
    central_api = config.get("central_api", "")
    if not central_api:
        return

    node_id = config.get("node_id", f"pi-{socket.gethostname()}")
    headers = {}
    if config.get("auth_token"):
        headers["Authorization"] = f"Bearer {config['auth_token']}"

    try:
        resp = http_requests.post(
            f"{central_api}/api/nodes/register",
            json={"node_id": node_id, "hostname": socket.gethostname()},
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info(f"Registered with central server: {central_api} as {node_id}")
        else:
            logger.warning(f"Central registration failed: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Could not reach central server: {e}")


def _heartbeat_loop(config: dict):
    """Send periodic heartbeats to the central server."""
    central_api = config.get("central_api", "")
    node_id = config.get("node_id", f"pi-{socket.gethostname()}")
    if not central_api:
        return

    headers = {}
    if config.get("auth_token"):
        headers["Authorization"] = f"Bearer {config['auth_token']}"

    while True:
        try:
            http_requests.post(
                f"{central_api}/api/nodes/{node_id}/heartbeat",
                headers=headers,
                timeout=5,
            )
        except Exception:
            pass
        time.sleep(30)


# ─── API Integration ────────────────────────────────────────────────
# These functions are called from server.py to wire up FL

_fl_client: Optional[ArgusFlowerClient] = None
_fl_server: Optional[ArgusFlowerServer] = None


def init_federated(prometheus_metrics: Optional[dict] = None) -> ArgusFlowerClient:
    """Initialize the FL subsystem. Call from server.py startup."""
    global _fl_client, _fl_server

    config = load_config()
    os.makedirs(config["model_dir"], exist_ok=True)
    os.makedirs(config["data_dir"], exist_ok=True)

    if prometheus_metrics:
        set_prometheus_metrics(**prometheus_metrics)

    _fl_client = ArgusFlowerClient(config)

    if config["role"] == "server":
        _fl_server = ArgusFlowerServer(config)
        _fl_server.start()

    # Register with central server if configured
    if config.get("register_on_startup") and config.get("central_api"):
        register_with_central(config)
        threading.Thread(target=_heartbeat_loop, args=(config,), daemon=True).start()

    _fl_client.start_flower_client()
    logger.info(f"Federated learning initialized (role={config['role']})")
    return _fl_client


def feed_detection(detection: dict):
    """Feed a detection event to the FL data collector."""
    if _fl_client:
        _fl_client.add_detection(detection)


def get_fl_status() -> dict:
    """Get current FL status for API responses."""
    if _fl_client is None:
        return {"fl_enabled": False}
    return {
        "fl_enabled": True,
        "fl_round": _fl_client._current_round,
        "fl_running": _fl_client._running,
        "fl_data_points": len(_fl_client.data_collector.buffer),
        "fl_role": load_config().get("role", "client"),
    }
