#!/usr/bin/env python3
"""
One-command hybrid federated demo: real node(s) plus simulated MEVA peers.

    python scripts/demo_hybrid.py --standin media/vtest.avi          # laptop stand-in node
    python scripts/demo_hybrid.py                                    # wait for the Pi to join
    python scripts/demo_hybrid.py --standin media/vtest.avi --poison signflip
    python scripts/demo_hybrid.py --real-nodes 2                     # node B has arrived

What it starts, in order:

  1. the aggregator (`sim/fl.py serve`, robust aggregation) waiting for
     peers + real nodes + the optional attacker;
  2. N simulated peers (`sim/fl.py client --store <MEVA camera>`), each one
     MEVA camera with its own non-IID events;
  3. with --standin VIDEO, a stand-in for the real node on this machine:
     the video goes through the real detection pipeline
     (`scripts/ingest_video.py`) and the result trains through the real
     node client (`src.federated.node_client`). Without it, the launcher
     prints the exact command to run on the Pi and waits for it to join.

Each round prints the aggregated model's balanced accuracy and per-class
recall on the pooled clip-level MEVA holdout, and which updates the robust
aggregator accepted or rejected. With three or more contributors the
direction and trimmed-mean filters are active; --poison adds one malicious
peer so the rejection is visible live.

Going from one real node to two changes one number: --real-nodes 2 makes
the aggregator wait for one more client, and the second node joins with the
same command the first one used (a different --node-name).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_EVENTS_ROOT = REPO / "data" / "datasets" / "meva" / "events"
DEFAULT_RUN_DIR = REPO / "sim" / "_run" / "hybrid"
MIN_CAMERA_EVENTS = 100
ATTACKER_NAME = "mallory"
# `import flwr` alone takes ~40 s on the laptop, and every process pays it.
STARTUP_TIMEOUT_S = 240.0


@dataclass(frozen=True)
class Camera:
    """A MEVA camera's converted event store."""
    name: str
    path: Path
    n_events: int


@dataclass(frozen=True)
class Plan:
    """Everything the launcher will start, as plain data."""
    cameras: Sequence[Camera]
    server_cmd: List[str]
    peer_cmds: List[List[str]]
    total_clients: int
    real_nodes: int
    address: str
    events_log: Path = field(default=DEFAULT_RUN_DIR / "events.jsonl")


# ---- planning -----------------------------------------------------------
def discover_cameras(root: Path, min_events: int = MIN_CAMERA_EVENTS) -> List[Camera]:
    """Converted cameras under `root`, biggest first, tiny ones skipped."""
    from src.federated.store import SampleStore

    out: List[Camera] = []
    for d in sorted(Path(root).glob("*")):
        path = d / "samples.jsonl"
        if not path.exists():
            continue
        n = sum(1 for r in SampleStore(path)._rows if r.get("y") is not None)
        if n >= min_events:
            out.append(Camera(d.name, d, n))
    return sorted(out, key=lambda c: -c.n_events)


def build_plan(*, events_root: Path, peers: int, rounds: int, epochs: int,
               real_nodes: int, poison: str, port: int,
               bind_all: bool = False, events_log: Optional[Path] = None,
               min_events: int = 20) -> Plan:
    """Work out the server and peer commands.

    `total_clients` is peers + real nodes + the attacker (if any). Adding
    node B changes `real_nodes` and nothing else.
    """
    cameras = discover_cameras(events_root, min_events=min_events)
    if peers > len(cameras):
        raise SystemExit(
            f"asked for {peers} simulated peers but only {len(cameras)} MEVA "
            f"cameras are converted under {events_root} "
            f"(python scripts/meva_to_events.py --camera Gxxx)")
    chosen = cameras[:peers]
    host = "0.0.0.0" if bind_all else "127.0.0.1"
    address = f"127.0.0.1:{port}"
    attackers = 0 if poison == "none" else 1
    total = peers + real_nodes + attackers
    log = events_log or DEFAULT_RUN_DIR / "events.jsonl"

    py = sys.executable
    sim_fl = str(REPO / "sim" / "fl.py")
    server_cmd = [
        py, sim_fl, "serve", "--rounds", str(rounds), "--clients", str(total),
        "--address", f"{host}:{port}", "--aggregation", "robust",
        "--eval-store", *[str(c.path) for c in chosen],
    ]
    peer_cmds = [
        [py, sim_fl, "client", "--store", str(c.path), "--epochs", str(epochs),
         "--seed", str(i + 1), "--address", address, "--node-id", c.name]
        for i, c in enumerate(chosen)
    ]
    if attackers:
        a = chosen[0]
        peer_cmds.append(
            [py, sim_fl, "client", "--store", str(a.path), "--epochs", str(epochs),
             "--seed", "99", "--address", address, "--node-id", ATTACKER_NAME,
             "--poison", poison])
    return Plan(chosen, server_cmd, peer_cmds, total, real_nodes, address, log)


# ---- rendering ----------------------------------------------------------
def format_round(ev: Dict, agg: Optional[Dict]) -> str:
    """One round as console text: accuracy, per-class recall, accept/reject."""
    r = ev.get("round", 0)
    head = ("round 0 (untrained start)" if r == 0 else f"round {r}")
    lines = [f"{head}: balanced accuracy {ev.get('balanced_accuracy', 0.0):.3f}"
             f"   loss {ev.get('loss', 0.0):.3f}"]
    recall = {k: v for k, v in (ev.get("per_class_recall") or {}).items() if v is not None}
    if recall:
        lines.append("  recall: " + "  ".join(f"{k} {v:.2f}" for k, v in recall.items()))
    cams = ev.get("per_camera_balanced_accuracy") or {}
    if cams:
        lines.append("  per camera: " + "  ".join(f"{k} {v:.2f}" for k, v in cams.items()))
    if agg:
        n_ok, n_bad = len(agg.get("accepted", [])), len(agg.get("rejected", []))
        filt = "direction + trimmed-mean filters ON" if agg.get("consensus_applied") \
            else "consensus filters off (fewer than 3 contributors)"
        lines.append(f"  updates: accepted {n_ok}, rejected {n_bad}   [{filt}]")
        for d in agg.get("detail", []):
            if not d.get("accepted"):
                lines.append(f"  REJECTED {d['node_id']}: {d.get('reason', '')}")
    return "\n".join(lines)


# ---- running ------------------------------------------------------------
def _lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "<laptop-ip>"


def _wait_port(port: int, timeout: float, proc: Optional[subprocess.Popen] = None) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            return False
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


def _spawn(cmd: List[str], log: Path, env: Dict[str, str]) -> subprocess.Popen:
    fh = open(log, "w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)


def _read_events(path: Path, offset: int):
    """New complete JSON lines since `offset`; returns (events, new_offset)."""
    if not path.exists():
        return [], offset
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    cut = data.rfind(b"\n") + 1
    out = []
    for line in data[:cut].decode("utf-8", "replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out, offset + cut


def _standin_cmds(video: str, run_dir: Path, address: str, epochs: int,
                  max_frames: int) -> List[List[str]]:
    data_dir = run_dir / "standin_data"
    py = sys.executable
    ingest = [py, str(REPO / "scripts" / "ingest_video.py"), video,
              "--label-by-class", "--data-dir", str(data_dir)]
    if max_frames:
        ingest += ["--max-frames", str(max_frames)]
    node = [py, "-m", "src.federated.node_client",
            "--store", str(data_dir / "training" / "samples.jsonl"),
            "--out", str(run_dir / "standin_out"), "--server", address,
            "--epochs", str(epochs), "--node-name", "node-A-standin"]
    return [ingest, node]


def run(args: argparse.Namespace) -> int:
    external = args.real_nodes - (1 if args.standin else 0)
    plan = build_plan(events_root=Path(args.events_root), peers=args.peers,
                      rounds=args.rounds, epochs=args.epochs,
                      real_nodes=args.real_nodes, poison=args.poison,
                      port=args.port, bind_all=external > 0,
                      events_log=Path(args.run_dir) / "events.jsonl")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if plan.events_log.exists():
        plan.events_log.unlink()
    env = dict(os.environ, ARGUS_FL_EVENTS=str(plan.events_log), PYTHONUNBUFFERED="1")
    procs: List[subprocess.Popen] = []

    print(f"Hybrid demo: {args.peers} simulated MEVA peers ({', '.join(c.name for c in plan.cameras)})"
          f" + {args.real_nodes} real node(s)"
          f"{' + 1 poisoned peer (' + args.poison + ')' if args.poison != 'none' else ''}"
          f" = {plan.total_clients} clients, {args.rounds} rounds, robust aggregation")
    try:
        standin = []
        if args.standin:
            print(f"Stand-in node: running {args.standin} through the detection pipeline ...")
            ingest, node = _standin_cmds(args.standin, run_dir, plan.address,
                                         args.epochs, args.max_frames)
            rc = subprocess.run(ingest, env=env, cwd=REPO,
                                stdout=open(run_dir / "standin_ingest.log", "w"),
                                stderr=subprocess.STDOUT).returncode
            if rc != 0:
                print(f"stand-in ingest failed (exit {rc}); see {run_dir / 'standin_ingest.log'}")
                return 1
            standin = [node]

        procs.append(_spawn(plan.server_cmd, run_dir / "server.log", env))
        if not _wait_port(args.port, STARTUP_TIMEOUT_S, procs[0]):
            print(f"aggregator did not open port {args.port}; see {run_dir / 'server.log'}")
            return 1
        for i, cmd in enumerate(plan.peer_cmds):
            procs.append(_spawn(cmd, run_dir / f"peer{i}.log", env))
        for i, cmd in enumerate(standin):
            procs.append(_spawn(cmd, run_dir / f"standin{i}.log", env))
        if external > 0:
            print(f"\nWaiting for {external} real node(s) to join. On the Pi:")
            print(f"  python -m src.federated.node_client --store data/training/samples.jsonl "
                  f"--init data/fl_head.npz --out data/fl_round "
                  f"--server {_lan_ip()}:{args.port} --node-name pi-A")
            print("Rounds start when everyone is connected.\n")

        return _watch(plan, procs[0], args.rounds, args.timeout)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def _watch(plan: Plan, server: subprocess.Popen, rounds: int, timeout: float) -> int:
    offset, agg_by_round, history = 0, {}, []
    deadline = time.time() + timeout
    done = False
    while time.time() < deadline and not done:
        events, offset = _read_events(plan.events_log, offset)
        for ev in events:
            kind = ev.get("kind")
            if kind == "aggregation":
                agg_by_round[ev["round"]] = ev
            elif kind == "round_eval":
                print(format_round(ev, agg_by_round.get(ev["round"])), flush=True)
                history.append(ev)
            elif kind == "server_done":
                done = True
        if server.poll() is not None and not done:
            # The aggregator exited. Drain whatever it wrote last; if it
            # never reported server_done it died, which is a failure.
            time.sleep(1.0)
            events, offset = _read_events(plan.events_log, offset)
            for ev in events:
                if ev.get("kind") == "round_eval":
                    print(format_round(ev, agg_by_round.get(ev["round"])), flush=True)
                    history.append(ev)
            if not any(e.get("kind") == "server_done" for e in events):
                print("aggregator exited early; see the server.log in the run directory")
                return 1
            done = True
        time.sleep(0.5)
    if not done:
        print(f"timed out after {timeout:.0f}s")
        return 1
    if history:
        first, last = history[0], history[-1]
        print(f"\nDone. Balanced accuracy {first['balanced_accuracy']:.3f} (start) -> "
              f"{last['balanced_accuracy']:.3f} (round {last['round']}).")
        rejected = sorted({n for a in agg_by_round.values() for n in a.get("rejected", [])})
        print(f"Rejected over the run: {', '.join(rejected) if rejected else 'none'}")
    (plan.events_log.parent / "summary.json").write_text(
        json.dumps({"rounds": history,
                    "aggregation": [agg_by_round[k] for k in sorted(agg_by_round)]}, indent=2),
        encoding="utf-8")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--peers", type=int, default=5, help="simulated MEVA camera clients (4-8 typical)")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3, help="local epochs per round")
    p.add_argument("--real-nodes", type=int, default=1,
                   help="real nodes in the federation, including the stand-in (2 once node B exists)")
    p.add_argument("--standin", default="",
                   help="video file run through the real node pipeline on this machine "
                        "instead of waiting for the Pi")
    p.add_argument("--max-frames", type=int, default=600, help="frames of the stand-in video to use (0 = all)")
    p.add_argument("--poison", choices=("none", "signflip", "nan"), default="none",
                   help="add one malicious peer to show robust aggregation rejecting it")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--events-root", default=str(DEFAULT_EVENTS_ROOT))
    p.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    p.add_argument("--timeout", type=float, default=3600.0, help="give up after this many seconds")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
