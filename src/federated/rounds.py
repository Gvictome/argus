"""
One federated round, run the same way however it was triggered.

The scheduled round (the cron) and the manual one must not drift apart:
if the automatic path trained something different from the button, the
demo would prove nothing about what runs at 2am. Both call this.

What a round is, concretely:

  1. Write the weights currently live to the run directory, so the client
     starts from what the node is actually using.
  2. Run the client as its own process. It trains on this node's labelled
     events and exchanges weights with the aggregator -- never footage.
  3. Take the averaged model the server sends back and put it through the
     validation gate, which only swaps it in if it does not do worse on
     this node's own held-out events.

Everything lands under DATA_DIR/fl_rounds/<timestamp>/ so a round can be
inspected after the fact: what was sent, what came back, what was decided.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from src.config import settings
from src.federated import model_state

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


def _rounds_completed(run_dir: Path) -> int:
    events = run_dir / "events.jsonl"
    if not events.exists():
        return 0
    count = 0
    for line in events.read_text(encoding="utf-8").splitlines():
        try:
            if json.loads(line).get("kind") == "global":
                count += 1
        except Exception:
            pass
    return count


def run_round_blocking(app, server: Optional[str] = None, epochs: int = 6,
                       tolerance: float = 0.01, min_samples: int = 200,
                       timeout_s: float = 1800.0) -> dict:
    """Join one round and gate the result. Blocks; call it off the event loop."""
    store = getattr(app.state, "fl_store", None)
    head = getattr(app.state, "fl_head", None)
    if store is None or head is None:
        return {"accepted": False, "reason": "federated head not initialised"}

    X, _ = store.labelled()
    if len(X) < min_samples:
        return {"accepted": False,
                "reason": f"only {len(X)} labelled samples, need {min_samples}"}

    server = server or settings.FL_SERVER_URL
    run_dir = Path(settings.DATA_DIR) / "fl_rounds" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    head.save(run_dir / "init.npz")

    cmd = [
        sys.executable, "-m", "src.federated.node_client",
        "--store", str(store.path),
        "--init", str(run_dir / "init.npz"),
        "--out", str(run_dir),
        "--server", server,
        "--epochs", str(epochs),
        "--node-name", settings.NODE_NAME,
    ]
    if getattr(settings, "FL_CENTRAL_API", ""):
        cmd += ["--central-api", settings.FL_CENTRAL_API]

    logger.info("FL round starting: server=%s samples=%d", server, len(X))
    try:
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True,
                              text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"accepted": False, "reason": f"round timed out after {timeout_s:.0f}s",
                "run_dir": str(run_dir)}

    candidate = run_dir / "global.npz"
    rounds = _rounds_completed(run_dir)

    if proc.returncode != 0 or not candidate.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
        logger.warning("FL round did not complete (exit %s)", proc.returncode)
        return {
            "accepted": False,
            "exit_code": proc.returncode,
            "rounds": rounds,
            "reason": ("could not reach the aggregator or it sent no model"
                       if proc.returncode != 0 else "server sent no global model"),
            "log": tail,
            "run_dir": str(run_dir),
        }

    decision = model_state.accept_candidate(
        app, candidate, "federated", tolerance,
        {"rounds": rounds, "run_dir": str(run_dir), "server": server},
    )
    logger.info("FL round finished: accepted=%s rounds=%d", decision.get("accepted"), rounds)
    return decision
