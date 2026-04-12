"""
Argus Central FL Aggregation Server
=====================================
Runs the Flower federated learning server with:
  - FedAvg aggregation strategy
  - Per-round epoch evaluation and metrics logging
  - Model checkpointing and versioning
  - REST API for monitoring training progress
  - Prometheus metrics for Grafana dashboards
  - Node registry for tracking connected Pi clients

Deploy via docker-compose or standalone behind Cloudflare Tunnel.
"""

import os
import sys
import json
import time
import logging
import threading
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from prometheus_client import (
    Counter, Gauge, Histogram, Info,
    generate_latest, CONTENT_TYPE_LATEST, REGISTRY,
)

# ─── Logging ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("argus.central")

# ─── Configuration ──────────────────────────────────────────────────

CONFIG_FILE = os.environ.get(
    "ARGUS_CONFIG", os.path.join(os.path.dirname(__file__), "central_config.json")
)

DEFAULT_CONFIG = {
    "fl_server_address": "0.0.0.0:8080",
    "api_port": 8090,
    "min_clients": 2,
    "rounds": 50,
    "eval_every_n_rounds": 1,
    "checkpoint_dir": os.path.join(os.path.dirname(__file__), "checkpoints"),
    "auth_token": "",  # Set via env ARGUS_AUTH_TOKEN or config
    "node_scrape_targets": [],  # List of "host:port" for Prometheus federation
}


def load_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg.update(json.load(f))
    # Env overrides
    if os.environ.get("ARGUS_AUTH_TOKEN"):
        cfg["auth_token"] = os.environ["ARGUS_AUTH_TOKEN"]
    if os.environ.get("ARGUS_FL_ADDRESS"):
        cfg["fl_server_address"] = os.environ["ARGUS_FL_ADDRESS"]
    if os.environ.get("ARGUS_MIN_CLIENTS"):
        cfg["min_clients"] = int(os.environ["ARGUS_MIN_CLIENTS"])
    if os.environ.get("ARGUS_ROUNDS"):
        cfg["rounds"] = int(os.environ["ARGUS_ROUNDS"])
    return cfg


config = load_config()

# ─── Prometheus Metrics ─────────────────────────────────────────────

FL_GLOBAL_ROUND = Gauge(
    "argus_fl_global_round", "Current global FL round"
)
FL_GLOBAL_LOSS = Gauge(
    "argus_fl_global_loss", "Aggregated global loss after evaluation"
)
FL_GLOBAL_ACCURACY = Gauge(
    "argus_fl_global_accuracy", "Aggregated global accuracy after evaluation"
)
FL_CONNECTED_CLIENTS = Gauge(
    "argus_fl_connected_clients", "Number of connected FL clients"
)
FL_ROUND_DURATION = Histogram(
    "argus_fl_round_duration_seconds", "Time per FL round",
    buckets=[1, 2, 5, 10, 30, 60, 120, 300],
)
FL_CLIENT_LOSS = Gauge(
    "argus_fl_client_loss", "Per-client loss from last round",
    ["node_id"],
)
FL_CLIENT_ACCURACY = Gauge(
    "argus_fl_client_accuracy", "Per-client accuracy from last round",
    ["node_id"],
)
FL_CLIENT_SAMPLES = Gauge(
    "argus_fl_client_samples", "Per-client training samples",
    ["node_id"],
)
FL_MODEL_VERSION = Gauge(
    "argus_fl_model_version", "Current global model version"
)
FL_CHECKPOINT_TOTAL = Counter(
    "argus_fl_checkpoints_total", "Total model checkpoints saved"
)
FL_TOTAL_ROUNDS = Counter(
    "argus_fl_rounds_total", "Total FL rounds completed"
)
SERVER_INFO = Info("argus_central", "Central server information")
SERVER_INFO.info({
    "version": "1.0.0",
    "role": "aggregator",
})

# ─── Node Registry ──────────────────────────────────────────────────


@dataclass
class NodeInfo:
    node_id: str
    hostname: str
    ip: str
    registered_at: str
    last_seen: str
    fl_rounds_participated: int = 0
    last_loss: float = 0.0
    last_accuracy: float = 0.0
    last_samples: int = 0
    status: str = "registered"  # registered, training, idle, offline


class NodeRegistry:
    def __init__(self):
        self._nodes: Dict[str, NodeInfo] = {}
        self._lock = threading.Lock()

    def register(self, node_id: str, hostname: str, ip: str) -> NodeInfo:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].last_seen = now
                self._nodes[node_id].status = "registered"
                return self._nodes[node_id]
            node = NodeInfo(
                node_id=node_id, hostname=hostname, ip=ip,
                registered_at=now, last_seen=now,
            )
            self._nodes[node_id] = node
            logger.info(f"Node registered: {node_id} ({hostname} @ {ip})")
            return node

    def update_training(self, node_id: str, loss: float, accuracy: float, samples: int):
        with self._lock:
            if node_id in self._nodes:
                n = self._nodes[node_id]
                n.last_seen = datetime.now(timezone.utc).isoformat()
                n.fl_rounds_participated += 1
                n.last_loss = loss
                n.last_accuracy = accuracy
                n.last_samples = samples
                n.status = "training"
                # Update per-client Prometheus metrics
                FL_CLIENT_LOSS.labels(node_id=node_id).set(loss)
                FL_CLIENT_ACCURACY.labels(node_id=node_id).set(accuracy)
                FL_CLIENT_SAMPLES.labels(node_id=node_id).set(samples)

    def heartbeat(self, node_id: str):
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].last_seen = datetime.now(timezone.utc).isoformat()

    def get_all(self) -> List[dict]:
        with self._lock:
            return [asdict(n) for n in self._nodes.values()]

    def count(self) -> int:
        with self._lock:
            return len(self._nodes)


registry = NodeRegistry()

# ─── Training History ───────────────────────────────────────────────


@dataclass
class RoundResult:
    round_num: int
    timestamp: str
    global_loss: float
    global_accuracy: float
    num_clients: int
    client_results: List[dict]
    duration_seconds: float
    model_version: int


class TrainingHistory:
    def __init__(self, max_history: int = 500):
        self._rounds: List[RoundResult] = []
        self._max = max_history
        self._lock = threading.Lock()

    def add(self, result: RoundResult):
        with self._lock:
            self._rounds.append(result)
            if len(self._rounds) > self._max:
                self._rounds.pop(0)

    def get_all(self) -> List[dict]:
        with self._lock:
            return [asdict(r) for r in self._rounds]

    def get_latest(self, n: int = 10) -> List[dict]:
        with self._lock:
            return [asdict(r) for r in self._rounds[-n:]]

    def get_round(self, round_num: int) -> Optional[dict]:
        with self._lock:
            for r in self._rounds:
                if r.round_num == round_num:
                    return asdict(r)
            return None


history = TrainingHistory()

# ─── Model Checkpointing ───────────────────────────────────────────

checkpoint_dir = Path(config["checkpoint_dir"])
checkpoint_dir.mkdir(parents=True, exist_ok=True)
_model_version = 0


def save_checkpoint(weights: List[np.ndarray], round_num: int, metrics: dict):
    global _model_version
    _model_version += 1
    path = checkpoint_dir / f"model_round_{round_num:04d}_v{_model_version}.npz"
    np.savez(
        str(path),
        **{f"w{i}": w for i, w in enumerate(weights)},
        round_num=round_num,
        version=_model_version,
    )
    # Also save as "latest"
    latest = checkpoint_dir / "model_latest.npz"
    np.savez(
        str(latest),
        **{f"w{i}": w for i, w in enumerate(weights)},
        round_num=round_num,
        version=_model_version,
    )
    FL_CHECKPOINT_TOTAL.inc()
    FL_MODEL_VERSION.set(_model_version)
    logger.info(f"Checkpoint saved: {path.name} (v{_model_version})")
    return str(path)


def load_latest_checkpoint() -> Optional[Tuple[List[np.ndarray], int]]:
    latest = checkpoint_dir / "model_latest.npz"
    if latest.exists():
        data = np.load(str(latest))
        weights = []
        i = 0
        while f"w{i}" in data:
            weights.append(data[f"w{i}"])
            i += 1
        return weights, int(data.get("version", 0))
    return None


# ─── Flower FL Server with Evaluation Strategy ─────────────────────

def start_flower_server():
    """Start Flower server with custom FedAvg strategy and epoch evaluation."""
    try:
        import flwr as fl
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    except ImportError:
        logger.error("Flower not installed — run: pip install flwr")
        return

    # Load initial weights if checkpoint exists
    initial_params = None
    checkpoint = load_latest_checkpoint()
    if checkpoint:
        weights, ver = checkpoint
        initial_params = ndarrays_to_parameters(weights)
        logger.info(f"Loaded checkpoint v{ver} as initial parameters")

    round_start_time = [0.0]
    current_round = [0]

    def fit_config(server_round: int):
        """Send config to clients each round."""
        current_round[0] = server_round
        round_start_time[0] = time.time()
        return {
            "server_round": server_round,
            "local_epochs": config.get("local_epochs", 3),
            "batch_size": config.get("batch_size", 16),
            "learning_rate": config.get("learning_rate", 0.001),
        }

    def evaluate_config(server_round: int):
        return {"server_round": server_round}

    def on_fit_end(server_round, aggregated_parameters, aggregated_metrics, failures):
        """Called after fit aggregation — evaluate and checkpoint."""
        duration = time.time() - round_start_time[0]
        FL_ROUND_DURATION.observe(duration)
        FL_GLOBAL_ROUND.set(server_round)
        FL_TOTAL_ROUNDS.inc()

        # Extract per-client metrics from aggregated_metrics
        client_results = []
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        if aggregated_metrics:
            for client_id, metrics in aggregated_metrics:
                loss = metrics.get("loss", 0.0) if isinstance(metrics, dict) else 0.0
                acc = metrics.get("accuracy", 0.0) if isinstance(metrics, dict) else 0.0
                samples = metrics.get("samples", 0) if isinstance(metrics, dict) else 0
                client_results.append({
                    "client": str(client_id),
                    "loss": loss,
                    "accuracy": acc,
                    "samples": samples,
                })
                total_loss += loss * max(samples, 1)
                total_acc += acc * max(samples, 1)
                total_samples += max(samples, 1)

        global_loss = total_loss / max(total_samples, 1)
        global_acc = total_acc / max(total_samples, 1)

        FL_GLOBAL_LOSS.set(global_loss)
        FL_GLOBAL_ACCURACY.set(global_acc)
        FL_CONNECTED_CLIENTS.set(len(client_results) if client_results else 0)

        # Checkpoint the model
        if aggregated_parameters is not None:
            weights = parameters_to_ndarrays(aggregated_parameters)
            save_checkpoint(weights, server_round, {
                "loss": global_loss, "accuracy": global_acc,
            })

        # Record in history
        result = RoundResult(
            round_num=server_round,
            timestamp=datetime.now(timezone.utc).isoformat(),
            global_loss=global_loss,
            global_accuracy=global_acc,
            num_clients=len(client_results),
            client_results=client_results,
            duration_seconds=duration,
            model_version=_model_version,
        )
        history.add(result)

        logger.info(
            f"Round {server_round}/{config['rounds']}: "
            f"loss={global_loss:.4f}, acc={global_acc:.4f}, "
            f"clients={len(client_results)}, time={duration:.1f}s"
        )

    class ArgusStrategy(fl.server.strategy.FedAvg):
        """FedAvg with per-round evaluation and checkpointing."""

        def aggregate_fit(self, server_round, results, failures):
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            # Collect per-client metrics
            client_metrics = []
            for client_proxy, fit_res in results:
                client_metrics.append(
                    (str(client_proxy.cid), fit_res.metrics)
                )
            on_fit_end(
                server_round, aggregated_parameters,
                client_metrics, failures,
            )
            return aggregated_parameters, aggregated_metrics

    strategy_kwargs = {
        "min_fit_clients": config["min_clients"],
        "min_evaluate_clients": config["min_clients"],
        "min_available_clients": config["min_clients"],
        "on_fit_config_fn": fit_config,
        "on_evaluate_config_fn": evaluate_config,
    }
    if initial_params:
        strategy_kwargs["initial_parameters"] = initial_params

    strategy = ArgusStrategy(**strategy_kwargs)

    logger.info(
        f"Starting Flower FL server on {config['fl_server_address']} "
        f"(rounds={config['rounds']}, min_clients={config['min_clients']})"
    )
    fl.server.start_server(
        server_address=config["fl_server_address"],
        config=fl.server.ServerConfig(num_rounds=config["rounds"]),
        strategy=strategy,
    )
    logger.info("FL training completed all rounds")


# ─── Flask REST API for Monitoring ──────────────────────────────────

api = Flask(__name__)
CORS(api)


def require_auth(f):
    """Optional auth middleware — skipped if no token configured."""
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        token = config.get("auth_token", "")
        if not token:
            return f(*args, **kwargs)
        auth = request.headers.get("Authorization", "")
        if auth == f"Bearer {token}":
            return f(*args, **kwargs)
        return jsonify({"error": "unauthorized"}), 401
    return decorated


@api.route("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)


@api.route("/api/status")
@require_auth
def status():
    return jsonify({
        "status": "running",
        "fl_server_address": config["fl_server_address"],
        "total_rounds": config["rounds"],
        "current_round": int(FL_GLOBAL_ROUND._value.get()),
        "global_loss": float(FL_GLOBAL_LOSS._value.get()),
        "global_accuracy": float(FL_GLOBAL_ACCURACY._value.get()),
        "connected_clients": int(FL_CONNECTED_CLIENTS._value.get()),
        "model_version": _model_version,
        "registered_nodes": registry.count(),
    })


@api.route("/api/nodes", methods=["GET"])
@require_auth
def get_nodes():
    return jsonify(registry.get_all())


@api.route("/api/nodes/register", methods=["POST"])
@require_auth
def register_node():
    data = request.json or {}
    node_id = data.get("node_id", f"node-{secrets.token_hex(4)}")
    hostname = data.get("hostname", "unknown")
    ip = request.remote_addr or "unknown"
    node = registry.register(node_id, hostname, ip)
    return jsonify(asdict(node))


@api.route("/api/nodes/<node_id>/heartbeat", methods=["POST"])
@require_auth
def node_heartbeat(node_id):
    registry.heartbeat(node_id)
    return jsonify({"status": "ok"})


@api.route("/api/training/history", methods=["GET"])
@require_auth
def training_history():
    limit = request.args.get("limit", 50, type=int)
    return jsonify(history.get_latest(limit))


@api.route("/api/training/round/<int:round_num>", methods=["GET"])
@require_auth
def training_round(round_num):
    result = history.get_round(round_num)
    if result is None:
        return jsonify({"error": "round not found"}), 404
    return jsonify(result)


@api.route("/api/training/summary", methods=["GET"])
@require_auth
def training_summary():
    """Summary of training progress — loss/accuracy curves for graphing."""
    rounds = history.get_all()
    return jsonify({
        "total_rounds_completed": len(rounds),
        "target_rounds": config["rounds"],
        "model_version": _model_version,
        "rounds": [
            {
                "round": r["round_num"],
                "loss": r["global_loss"],
                "accuracy": r["global_accuracy"],
                "clients": r["num_clients"],
                "duration": r["duration_seconds"],
            }
            for r in rounds
        ],
    })


@api.route("/api/model/latest", methods=["GET"])
@require_auth
def get_latest_model():
    """Download the latest model checkpoint."""
    latest = checkpoint_dir / "model_latest.npz"
    if not latest.exists():
        return jsonify({"error": "no checkpoint available"}), 404
    from flask import send_file
    return send_file(str(latest), mimetype="application/octet-stream",
                     as_attachment=True, download_name="argus_model_latest.npz")


@api.route("/api/model/checkpoints", methods=["GET"])
@require_auth
def list_checkpoints():
    files = sorted(checkpoint_dir.glob("model_round_*.npz"))
    return jsonify([
        {"name": f.name, "size": f.stat().st_size, "modified": f.stat().st_mtime}
        for f in files[-20:]  # Last 20 checkpoints
    ])


@api.route("/api/config", methods=["GET"])
@require_auth
def get_config():
    safe = {k: v for k, v in config.items() if k != "auth_token"}
    return jsonify(safe)


@api.route("/api/config", methods=["POST"])
@require_auth
def update_config():
    """Hot-reload select config values (rounds, min_clients, etc.)."""
    data = request.json or {}
    allowed = {"rounds", "min_clients", "eval_every_n_rounds",
               "local_epochs", "batch_size", "learning_rate"}
    updated = {}
    for key in allowed:
        if key in data:
            config[key] = data[key]
            updated[key] = data[key]
    logger.info(f"Config updated: {updated}")
    return jsonify({"updated": updated})


# ─── Prometheus Scrape Target Generation ────────────────────────────

@api.route("/api/prometheus/targets", methods=["GET"])
def prometheus_targets():
    """HTTP SD targets for Prometheus — auto-discovers registered nodes."""
    nodes = registry.get_all()
    targets = []
    for node in nodes:
        targets.append({
            "targets": [f"{node['ip']}:5000"],
            "labels": {
                "node_id": node["node_id"],
                "hostname": node["hostname"],
                "job": "argus-edge",
            },
        })
    return jsonify(targets)


# ─── Main ───────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("  ARGUS Central FL Aggregation Server")
    logger.info("=" * 60)
    logger.info(f"FL Server: {config['fl_server_address']}")
    logger.info(f"API Port:  {config['api_port']}")
    logger.info(f"Rounds:    {config['rounds']}")
    logger.info(f"Min Clients: {config['min_clients']}")
    logger.info(f"Auth:      {'enabled' if config.get('auth_token') else 'disabled'}")
    logger.info("=" * 60)

    # Start REST API in background thread (Flask is thread-safe)
    api_thread = threading.Thread(
        target=lambda: api.run(host="0.0.0.0", port=config["api_port"], use_reloader=False),
        daemon=True,
    )
    api_thread.start()

    # Start Flower server in main thread (requires main thread for signal handlers)
    start_flower_server()


if __name__ == "__main__":
    main()
